#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate chess moves with a moves-only CausalLM (ChessGPT) + optional legal-move masking.

Usage:
  chessgpt-generate --checkpoint malcouffe/chessgpt --moves "e2e4 e7e5 g1f3"
  chessgpt-generate --checkpoint ./hf_model --set sampling.temperature=0.8 max_new_moves=40

Common knobs (via --set):
  sampling.temperature=1.0  sampling.top_k=40  sampling.top_p=0.95
  sampling.greedy=true      (deterministic, ignores temperature/top_k/top_p)
  legal_mask=true           (recommended)
  min_new_moves=20          (EOS allowed only after N generated moves)
"""

from __future__ import annotations

import argparse
import importlib.resources
import os
import random
import sys
from typing import List, Tuple

import torch
import chess

from . import (
    UCITokenizer,
    ChessGPTConfig,
    ChessGPTForCausalLM,
    resolve_device,
    set_seed,
)
from .config import GenerateConfig, load_config


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep only top_k logits (others set to -inf). logits shape: (V,)"""
    if top_k is None or top_k <= 0 or top_k >= logits.numel():
        return logits
    values, _ = torch.topk(logits, top_k)
    cutoff = values[-1]
    neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)
    return torch.where(logits < cutoff, neg_inf, logits)


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filtering. Keeps the smallest set of tokens with cumulative prob >= top_p."""
    if top_p is None or top_p <= 0.0 or top_p >= 1.0:
        return logits

    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(sorted_probs, dim=-1)

    # mask tokens above nucleus threshold
    to_remove = cumprobs > top_p
    to_remove[0] = False  # keep at least one token

    sorted_logits = sorted_logits.masked_fill(to_remove, float("-inf"))

    out = torch.full_like(logits, float("-inf"))
    out[sorted_idx] = sorted_logits
    return out


def build_legal_mask(
    board: chess.Board,
    tokenizer: UCITokenizer,
    device: torch.device,
    allow_eos: bool,
) -> torch.Tensor:
    """
    Returns a boolean mask (vocab_size,) where True means allowed.
    Allowed = legal moves in current position (+ optional EOS).
    """
    mask = torch.zeros(tokenizer.vocab_size, dtype=torch.bool, device=device)

    for mv in board.legal_moves:
        u = mv.uci()  # includes promotion letter when relevant (e.g. e7e8q)
        tid = tokenizer.move_to_id.get(u)
        if tid is not None:
            mask[tid] = True

    if allow_eos:
        mask[tokenizer.EOS_ID] = True

    # never allow special tokens as a "move"
    mask[tokenizer.PAD_ID] = False
    mask[tokenizer.BOS_ID] = False
    return mask


def uci_history_to_board(moves_uci: List[str]) -> chess.Board:
    board = chess.Board()
    for mv in moves_uci:
        board.push_uci(mv)
    return board


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_next_token(
    model: ChessGPTForCausalLM,
    input_ids: torch.Tensor,             # (1, T)
    tokenizer: UCITokenizer,
    legal_mask: torch.Tensor | None,     # (V,) or None
    temperature: float,
    top_k: int,
    top_p: float,
    greedy: bool,
) -> int:
    """
    Forward -> last logits -> optional legal mask -> (greedy or temp/top-k/top-p sampling).
    """
    use_amp = (input_ids.device.type == "cuda")
    with torch.amp.autocast(device_type=input_ids.device.type, enabled=use_amp):
        output = model(input_ids)
    logits = output.logits[0, -1, :]  # (V,)

    if legal_mask is not None:
        logits = logits.masked_fill(~legal_mask, float("-inf"))

    # Greedy mode
    if greedy or temperature == 0.0:
        tid = torch.argmax(logits).item()
        if tid in (tokenizer.PAD_ID, tokenizer.BOS_ID):
            return tokenizer.EOS_ID
        if torch.isinf(logits[tid]) and logits[tid] < 0:
            return tokenizer.EOS_ID
        return int(tid)

    # Temperature
    if temperature < 0:
        temperature = 1.0
    logits = logits / max(temperature, 1e-6)

    # Top-k then top-p
    logits = top_k_filtering(logits, top_k)
    logits = top_p_filtering(logits, top_p)

    probs = torch.softmax(logits, dim=-1)

    # If all tokens are masked, probs become NaN or sum=0
    if torch.isnan(probs).any() or probs.sum().item() == 0.0:
        return tokenizer.EOS_ID

    return int(torch.multinomial(probs, num_samples=1).item())


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_legacy_checkpoint(
    checkpoint_path: str, device: torch.device,
) -> Tuple[ChessGPTForCausalLM, UCITokenizer, ChessGPTConfig]:
    """Load a legacy .pt checkpoint (backward compatibility)."""
    from .modeling_chessgpt import ChessGPTForCausalLM as _Model

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model config from checkpoint
    if "config" in ckpt and "model" in ckpt.get("config", {}):
        model_d = ckpt["config"]["model"]
    else:
        model_d = ckpt.get("train_config", {})

    tokenizer = UCITokenizer()

    config = ChessGPTConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=int(model_d.get("d_model", 256)),
        n_layers=int(model_d.get("n_layers", 8)),
        n_heads=int(model_d.get("n_heads", 8)),
        d_ff=int(model_d.get("d_ff", 1024)),
        max_seq_len=int(model_d.get("max_seq_len", 256)),
        dropout=float(model_d.get("dropout", 0.0)),
    )

    # Build the new model and remap old state dict keys
    model = _Model(config).to(device)

    old_sd = ckpt["model_state_dict"]
    new_sd = {}
    for key, value in old_sd.items():
        # Strip _orig_mod. prefix from torch.compile'd checkpoints
        clean_key = key.replace("_orig_mod.", "")

        if clean_key == "head.weight":
            new_key = "lm_head.weight"
        elif clean_key == "token_emb.weight":
            new_key = "model.embed_tokens.weight"
        elif clean_key == "freqs_cis":
            continue  # recomputed buffer
        else:
            new_key = "model." + clean_key
        new_sd[new_key] = value

    model.load_state_dict(new_sd, strict=False)
    model.eval()

    return model, tokenizer, config


def load_checkpoint(
    checkpoint_path: str, device: torch.device,
) -> Tuple[ChessGPTForCausalLM, UCITokenizer, ChessGPTConfig]:
    """Load model and tokenizer.

    checkpoint_path can be:
    - A HuggingFace model ID (e.g., "malcouffe/chessgpt")
    - A local directory containing HF-format model files
    - A legacy .pt checkpoint file (backward compatibility)
    """
    if os.path.isfile(checkpoint_path) and checkpoint_path.endswith(".pt"):
        return _load_legacy_checkpoint(checkpoint_path, device)

    # HuggingFace from_pretrained
    model = ChessGPTForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    tokenizer = UCITokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    config = model.config

    return model, tokenizer, config


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    model: ChessGPTForCausalLM,
    tokenizer: UCITokenizer,
    device: torch.device,
    cfg: GenerateConfig,
):
    # Parse history
    moves = cfg.moves
    history = [m for m in moves.strip().split() if m] if moves else []

    # Validate history tokens (better error than KeyError)
    for m in history:
        if tokenizer.move_to_id.get(m) is None:
            raise ValueError(f"Unknown/invalid UCI move in prompt: {m}")

    # Build board from history (also validates legality)
    board = uci_history_to_board(history)

    # Initial token sequence: BOS + history (no EOS)
    ids = [tokenizer.BOS_ID] + [tokenizer.move_to_id[m] for m in history]

    # Crop to max context
    max_ctx = model.config.max_seq_len
    if len(ids) > max_ctx:
        ids = ids[-max_ctx:]

    if cfg.verbose:
        print("Initial board:")
        print(board)
        print("FEN:", board.fen())
        print("History:", " ".join(history) if history else "(empty)")
        print("-" * 60)

    generated: List[str] = []

    for step in range(cfg.max_new_moves):
        if cfg.stop_on_game_over and board.is_game_over():
            if cfg.verbose:
                print(f"[stop] game over: {board.result()}  ({board.outcome()})")
            break

        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        # Allow EOS only after min_new_moves (or if game is already over)
        allow_eos = (step >= cfg.min_new_moves) or board.is_game_over()

        lm = None
        if cfg.legal_mask:
            lm = build_legal_mask(board, tokenizer, device, allow_eos=allow_eos)

        tid = sample_next_token(
            model=model,
            input_ids=input_ids,
            tokenizer=tokenizer,
            legal_mask=lm,
            temperature=cfg.sampling.temperature,
            top_k=cfg.sampling.top_k,
            top_p=cfg.sampling.top_p,
            greedy=cfg.sampling.greedy,
        )

        # EOS -> stop
        if tid == tokenizer.EOS_ID:
            if cfg.verbose:
                print("[stop] sampled EOS")
            break

        move_uci = tokenizer.id_to_move.get(tid, "")
        if not move_uci:
            # Shouldn't happen, but be safe
            if cfg.verbose:
                print("[warn] unknown token id, fallback to random legal move")
            mv = random.choice(list(board.legal_moves))
            move_uci = mv.uci()
        else:
            mv = chess.Move.from_uci(move_uci)

        # If not using legal mask (or mismatch), enforce legality with fallback
        if mv not in board.legal_moves:
            if cfg.verbose:
                print(f"[warn] illegal sampled move {move_uci} -> fallback random legal")
            mv = random.choice(list(board.legal_moves))
            move_uci = mv.uci()

        if cfg.verbose:
            try:
                san = board.san(mv)
            except Exception:
                san = "?"
            print(f"{step+1:>3d}. {move_uci}   ({san})")

        board.push(mv)
        generated.append(move_uci)

        # Append to context (must exist in tokenizer vocab)
        tid2 = tokenizer.move_to_id.get(move_uci)
        if tid2 is None:
            # Should be rare (if tokenizer doesn't include some legal move form),
            # but keep generation going: stop gracefully.
            if cfg.verbose:
                print(f"[stop] move not in tokenizer vocab: {move_uci}")
            break

        ids.append(tid2)
        if len(ids) > max_ctx:
            ids = ids[-max_ctx:]

    return history, generated, board


# ---------------------------------------------------------------------------
# Default config path
# ---------------------------------------------------------------------------

def _default_config_path() -> str:
    """Locate the bundled generate_default.yaml config file."""
    ref = importlib.resources.files("chessgpt").parent / "configs" / "generate_default.yaml"
    return str(ref)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Generate moves with ChessGPT (+ optional legal masking)")

    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: bundled generate_default.yaml)",
    )
    p.add_argument(
        "--set",
        nargs="*",
        default=[],
        dest="overrides",
        help="Override values: --set checkpoint=path.pt sampling.temperature=0.8",
    )

    # Convenience shortcuts (most common overrides)
    p.add_argument("--checkpoint", type=str, default=None, help="HF model ID, local dir, or legacy .pt path")
    p.add_argument("--moves", type=str, default=None, help="UCI history, e.g. 'e2e4 e7e5 g1f3'")

    return p.parse_args()


def main():
    args = parse_args()

    overrides = list(args.overrides)

    # Add convenience flags as overrides
    if args.checkpoint is not None:
        overrides.append(f"checkpoint={args.checkpoint}")
    if args.moves is not None:
        overrides.append(f"moves={args.moves}")

    config_path = args.config if args.config is not None else _default_config_path()
    cfg = load_config(config_path, GenerateConfig, overrides)

    if not cfg.checkpoint:
        print("Error: --checkpoint is required (or set checkpoint=... in config)")
        sys.exit(1)

    device = resolve_device(cfg.device)
    set_seed(cfg.seed)

    model, tokenizer, _ = load_checkpoint(cfg.checkpoint, device)

    history, generated, final_board = generate(
        model=model,
        tokenizer=tokenizer,
        device=device,
        cfg=cfg,
    )

    print("\n=== RESULT ===")
    print("History   :", " ".join(history) if history else "(empty)")
    print("Generated :", " ".join(generated) if generated else "(none)")
    print("Final FEN :", final_board.fen())
    print("\nFinal board:")
    print(final_board)
    print("Game over :", final_board.is_game_over())
    if final_board.is_game_over():
        print("Result    :", final_board.result())


if __name__ == "__main__":
    main()
