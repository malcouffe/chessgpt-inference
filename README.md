# chessgpt-inference

Inference package for **ChessGPT**, a 432M-parameter decoder-only transformer trained on chess games in UCI notation.

Model weights are hosted on HuggingFace: [malcouffe/chessgpt](https://huggingface.co/malcouffe/chessgpt)

## Features

- **Legal move masking** — constrain generation to only legal moves using `python-chess`
- **HuggingFace compatible** — load with `from_pretrained()` or the built-in CLI
- **Interactive notebook** — play against the model in Jupyter

## Installation

```bash
pip install git+https://github.com/malcouffe/chessgpt-inference.git
```

## Quick Start

### Generate moves (CLI)

```bash
chessgpt-generate --checkpoint malcouffe/chessgpt --moves "e2e4 e7e5"
```

Common options via `--set`:

```bash
# Greedy decoding
chessgpt-generate --checkpoint malcouffe/chessgpt --set sampling.greedy=true

# Adjust temperature and top-k
chessgpt-generate --checkpoint malcouffe/chessgpt --set sampling.temperature=0.8 sampling.top_k=20

# Verbose output (shows board + SAN notation)
chessgpt-generate --checkpoint malcouffe/chessgpt --set verbose=true

# Generate from empty board, at least 40 moves
chessgpt-generate --checkpoint malcouffe/chessgpt --set min_new_moves=40 max_new_moves=80
```

### Python API

```python
import torch
from chessgpt import ChessGPTForCausalLM, UCITokenizer
from chessgpt.generate import load_checkpoint, generate
from chessgpt.config import GenerateConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer, config = load_checkpoint("malcouffe/chessgpt", device)

cfg = GenerateConfig(
    moves="e2e4 e7e5 g1f3",
    max_new_moves=40,
)

history, generated, final_board = generate(model, tokenizer, device, cfg)
print("Generated:", " ".join(generated))
print("Final FEN:", final_board.fen())
```

### Interactive Play

Open [`notebooks/play.ipynb`](notebooks/play.ipynb) in Jupyter to play a full game against ChessGPT with a visual board.

## Model Details

See the full [model card on HuggingFace](https://huggingface.co/malcouffe/chessgpt) for architecture, training data, and hyperparameters.

| | |
|---|---|
| Parameters | ~432M |
| Architecture | Decoder-only transformer (RMSNorm, RoPE, SwiGLU) |
| Vocabulary | 4209 UCI tokens |
| Training data | Lichess games (ELO >= 1800), July 2025 -- January 2026 |
| Tokens seen | ~6.55B |
| Validation loss | 1.3314 |

## License

Apache 2.0
