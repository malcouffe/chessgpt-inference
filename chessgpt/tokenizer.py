# ---------------------------------------------------------------------------
# UCI Tokenizer
# ---------------------------------------------------------------------------


class UCITokenizer:
    """Maps UCI move strings to integer token IDs and back.

    Vocab:
      - <PAD>=0, <BOS>=1, <EOS>=2
      - All src->dst (4032)
      - Promotions (176)
    """

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2

    def __init__(self):
        self.move_to_id: dict[str, int] = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2}
        self.id_to_move: dict[int, str] = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>"}
        idx = 3

        squares = [f + r for f in "abcdefgh" for r in "12345678"]

        # Normal moves: all src -> dst where src != dst
        for src in squares:
            for dst in squares:
                if src != dst:
                    u = src + dst
                    self.move_to_id[u] = idx
                    self.id_to_move[idx] = u
                    idx += 1

        # Promotion moves: (including captures) from 7->8 for white, 2->1 for black
        for f_idx, f in enumerate("abcdefgh"):
            for df in (-1, 0, 1):  # capture left, forward, capture right
                nf_idx = f_idx + df
                if 0 <= nf_idx < 8:
                    nf = "abcdefgh"[nf_idx]
                    for promo in "qrbn":
                        # White promotion
                        u = f + "7" + nf + "8" + promo
                        self.move_to_id[u] = idx
                        self.id_to_move[idx] = u
                        idx += 1
                        # Black promotion
                        u = f + "2" + nf + "1" + promo
                        self.move_to_id[u] = idx
                        self.id_to_move[idx] = u
                        idx += 1

        self.vocab_size = idx

    def encode(self, moves_uci: str) -> list[int]:
        """Encode a space-separated UCI move string to token IDs, adding BOS/EOS."""
        tokens = [self.BOS_ID]
        if moves_uci:
            for mv in moves_uci.split():
                # dataset should already be clean; but defensive:
                tid = self.move_to_id.get(mv)
                if tid is not None:
                    tokens.append(tid)
        tokens.append(self.EOS_ID)
        return tokens

    def decode(self, token_ids: list[int]) -> list[str]:
        """Decode token IDs back to UCI move strings (skipping special tokens)."""
        out = []
        for tid in token_ids:
            if tid in (self.PAD_ID, self.BOS_ID, self.EOS_ID):
                continue
            out.append(self.id_to_move.get(int(tid), ""))
        return [x for x in out if x]