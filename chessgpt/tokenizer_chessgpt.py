"""UCI Tokenizer for ChessGPT, compatible with HuggingFace transformers."""

import json
import os
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer


class UCITokenizer(PreTrainedTokenizer):
    """Maps UCI move strings to integer token IDs and back.

    Vocab:
      - <PAD>=0, <BOS>=1, <EOS>=2
      - All src->dst normal moves (4032)
      - Promotions (176)
      - Total: 4209 tokens (including 1 unused slot for <UNK> alias)
    """

    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        bos_token: str = "<BOS>",
        eos_token: str = "<EOS>",
        pad_token: str = "<PAD>",
        unk_token: str = "<PAD>",
        **kwargs,
    ):
        # Build or load vocabulary
        if vocab_file is not None and os.path.isfile(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                self.encoder: Dict[str, int] = json.load(f)
        else:
            self.encoder = self._build_vocab()

        self.decoder: Dict[int, str] = {v: k for k, v in self.encoder.items()}

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            **kwargs,
        )

    @staticmethod
    def _build_vocab() -> Dict[str, int]:
        """Build the UCI move vocabulary deterministically."""
        vocab: Dict[str, int] = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2}
        idx = 3

        squares = [f + r for f in "abcdefgh" for r in "12345678"]

        # Normal moves: all src -> dst where src != dst
        for src in squares:
            for dst in squares:
                if src != dst:
                    vocab[src + dst] = idx
                    idx += 1

        # Promotion moves (including captures)
        for f_idx, f in enumerate("abcdefgh"):
            for df in (-1, 0, 1):  # capture left, forward, capture right
                nf_idx = f_idx + df
                if 0 <= nf_idx < 8:
                    nf = "abcdefgh"[nf_idx]
                    for promo in "qrbn":
                        # White promotion
                        vocab[f + "7" + nf + "8" + promo] = idx
                        idx += 1
                        # Black promotion
                        vocab[f + "2" + nf + "1" + promo] = idx
                        idx += 1

        return vocab

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.encoder)

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """Split UCI move string on whitespace. Each move is one token."""
        return text.strip().split()

    def _convert_token_to_id(self, token: str) -> int:
        return self.encoder.get(token, self.PAD_ID)

    def _convert_id_to_token(self, index: int) -> str:
        return self.decoder.get(index, "<PAD>")

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Add BOS at the start and EOS at the end."""
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            raise ValueError(f"save_directory ({save_directory}) is not a directory")
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json",
        )
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.encoder, f, ensure_ascii=False, indent=2)
        return (vocab_file,)

    # ------------------------------------------------------------------
    # Backward-compatibility helpers (used by generate.py)
    # ------------------------------------------------------------------

    @property
    def move_to_id(self) -> Dict[str, int]:
        return self.encoder

    @property
    def id_to_move(self) -> Dict[int, str]:
        return self.decoder
