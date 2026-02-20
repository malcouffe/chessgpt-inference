# chessgpt-inference

Inference package for **ChessGPT**, a 432M-parameter decoder-only transformer pre-trained on chess games in UCI notation.

Model weights are hosted on HuggingFace: [malcouffe/chessgpt](https://huggingface.co/malcouffe/chessgpt)

## Installation

```bash
pip install git+https://github.com/malcouffe/chessgpt-inference.git
```

## Quick Start

### 1. Download the model weights

```bash
pip install huggingface_hub
huggingface-cli download malcouffe/chessgpt step_25000.pt --local-dir .
```

### 2. Generate moves (CLI)

```bash
chessgpt-generate --checkpoint step_25000.pt --moves "e2e4 e7e5"
```

Common options via `--set`:

```bash
# Greedy decoding
chessgpt-generate --checkpoint step_25000.pt --set sampling.greedy=true

# Adjust temperature and top-k
chessgpt-generate --checkpoint step_25000.pt --set sampling.temperature=0.8 sampling.top_k=20

# Verbose output (shows board + SAN notation)
chessgpt-generate --checkpoint step_25000.pt --set verbose=true

# Generate from empty board, at least 40 moves
chessgpt-generate --checkpoint step_25000.pt --set min_new_moves=40 max_new_moves=80
```

### 3. Python API

```python
import torch
from chessgpt import ChessGPT, ChessGPTConfig, UCITokenizer
from chessgpt.generate import load_checkpoint, generate
from chessgpt.config import GenerateConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer, model_cfg = load_checkpoint("step_25000.pt", device)

cfg = GenerateConfig(
    moves="e2e4 e7e5 g1f3",
    max_new_moves=40,
)

history, generated, final_board = generate(model, tokenizer, device, cfg)
print("Generated:", " ".join(generated))
print("Final FEN:", final_board.fen())
```

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
