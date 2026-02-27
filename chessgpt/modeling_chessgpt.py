"""
ChessGPT -- LLaMA-style decoder-only transformer for UCI move prediction.

Architecture: RMSNorm, RoPE, SwiGLU, QK-Norm, no bias, scaled residual init.
HuggingFace-compatible implementation.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_chessgpt import ChessGPTConfig


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


def precompute_rope_freqs(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (freqs_cos, freqs_sin) as real-valued tensors."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len)
    angles = torch.outer(t, freqs)
    return angles.cos(), angles.sin()


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
):
    # xq, xk: (B, n_heads, T, head_dim)
    T = xq.shape[2]
    cos = freqs_cos[:T][None, None, :, :]  # (1, 1, T, head_dim//2)
    sin = freqs_sin[:T][None, None, :, :]

    # Split into pairs and apply rotation
    xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)

    xq_out = torch.stack([
        xq_r[..., 0] * cos - xq_r[..., 1] * sin,
        xq_r[..., 0] * sin + xq_r[..., 1] * cos,
    ], dim=-1).flatten(-2)

    xk_out = torch.stack([
        xk_r[..., 0] * cos - xk_r[..., 1] * sin,
        xk_r[..., 0] * sin + xk_r[..., 1] * cos,
    ], dim=-1).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # gate
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalSelfAttention(nn.Module):
    """Causal self-attention with RoPE and QK-Norm, using PyTorch SDPA."""

    def __init__(self, config: ChessGPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # (B, T, nh, hd)

        q = q.transpose(1, 2)  # (B, nh, T, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # QK-Norm before RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)

        # PyTorch SDPA (uses flash-attn kernels when possible)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
        )  # (B, nh, T, hd)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, config: ChessGPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn = SwiGLU(config.d_model, config.d_ff)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), freqs_cos, freqs_sin)
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# HuggingFace-compatible model classes
# ---------------------------------------------------------------------------


class ChessGPTPreTrainedModel(PreTrainedModel):
    config_class = ChessGPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]

    def _init_weights(self, module):
        std = self.config.weight_init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)


class ChessGPTModel(ChessGPTPreTrainedModel):
    """The bare ChessGPT transformer outputting raw hidden-states."""

    def __init__(self, config: ChessGPTConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        head_dim = config.d_model // config.n_heads
        freqs_cos, freqs_sin = precompute_rope_freqs(
            head_dim, config.max_seq_len, config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=True)
        self.register_buffer("freqs_sin", freqs_sin, persistent=True)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_f = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        if T > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {T} > max_seq_len {self.config.max_seq_len}"
            )

        x = self.embed_tokens(input_ids)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch_checkpoint(block, x, self.freqs_cos, self.freqs_sin, use_reentrant=False)
            else:
                x = block(x, self.freqs_cos, self.freqs_sin)

        x = self.ln_f(x)
        return x


class ChessGPTForCausalLM(ChessGPTPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: ChessGPTConfig):
        super().__init__(config)
        self.model = ChessGPTModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        hidden_states = self.model(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)
