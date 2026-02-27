"""ChessGPT model configuration."""

from transformers import PretrainedConfig


class ChessGPTConfig(PretrainedConfig):
    model_type = "chessgpt"

    def __init__(
        self,
        vocab_size: int = 4211,
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        dropout: float = 0.0,
        weight_init_std: float = 0.02,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.weight_init_std = weight_init_std
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        super().__init__(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
