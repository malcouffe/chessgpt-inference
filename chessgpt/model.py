"""
ChessGPT — LLaMA-style decoder-only transformer for UCI move prediction.

Architecture: RMSNorm, RoPE, SwiGLU, no dropout, scaled residual init.
Training: pure next-token prediction on UCI move tokens (no board state).
Inference: you can optionally apply legal-move masking with python-chess.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ChessGPTConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.0
    weight_init_std: float = 0.02


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    # xq, xk: (B, n_heads, T, head_dim)
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs = freqs_cis[None, None, :xq.shape[2], :]  # (1, 1, T, head_dim//2)
    xq_out = torch.view_as_real(xq_c * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_c * freqs).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # gate
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Causal self-attention with RoPE, using PyTorch SDPA."""

    def __init__(self, config: ChessGPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # (B,T,nh,hd)

        # (B, nh, T, hd)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # QK-Norm before RoPE (Gemma / DeepSeek-V2 practice)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE to Q and K
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # PyTorch SDPA (uses flash-attn kernels when possible)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )  # (B, nh, T, hd)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):

    def __init__(self, config: ChessGPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.d_ff)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), freqs_cis)
        x = x + self.ffn(self.ln2(x))
        return x


class ChessGPT(nn.Module):

    def __init__(self, config: ChessGPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        head_dim = config.d_model // config.n_heads
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, config.max_seq_len),
        )

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = RMSNorm(config.d_model)

        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # weight tying

        self.gradient_checkpointing = False
        self.apply(self._init_weights)

        # Scale residual projections (GPT-2/3 practice)
        residual_std = self.config.weight_init_std / (2 * self.config.n_layers) ** 0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=residual_std)
            nn.init.normal_(block.ffn.w2.weight, std=residual_std)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.weight_init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.weight_init_std)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        """
        Args:
            input_ids: (B, T)
            targets: (B, T) or None
        Returns:
            logits: (B, T, vocab)
            loss: scalar or None
        """
        B, T = input_ids.shape
        if T > self.config.max_seq_len:
            raise ValueError(f"Sequence length {T} > max_seq_len {self.config.max_seq_len}")

        x = self.token_emb(input_ids)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch_checkpoint(block, x, self.freqs_cis, use_reentrant=False)
            else:
                x = block(x, self.freqs_cis)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,  # PAD
            )
        return logits, loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
