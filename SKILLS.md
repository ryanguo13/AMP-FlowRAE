# AMP-FlowRAE Development Guide

## Environment Setup

This project uses **uv** for Python environment management.

### Quick Start
```bash
# Activate virtual environment (if exists)
source .venv/bin/activate

# Or run commands directly with uv run
uv run python script.py
uv run pytest tests/
```

### Installing Dependencies
```bash
uv pip install <package>
uv pip install -r requirements.txt
```

---

## Flow Matching V2 Training

### Quick Start
```bash
# Train the model
uv run python flow_matching_v2/train.py --epochs 200 --batch-size 256

# Sample with classifier-free guidance
uv run python flow_matching_v2/sample.py --num-samples 1000 --guidance-scale 1.5

# Evaluate generated sequences
uv run python scripts/evaluate_flow_matching.py \
    --input-fasta outputs/generated_sequences.fasta \
    --conditions-csv outputs/samples_v2/sampled_conditions.csv \
    --train-fasta data/processed/filter/merged.fasta \
    --mmseqs-db data/processed/filter/filterdb \
    --output-dir outputs/evaluation_v2

# Visualize results
uv run python flow_matching_v2/visualize.py
```

### Training Options
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 200 | Number of training epochs |
| `--batch-size` | 256 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--drop-condition-prob` | 0.1 | Classifier-free guidance drop probability |
| `--hidden-dims` | 512 512 512 256 | Hidden layer dimensions |
| `--use-amp` | - | Enable automatic mixed precision |

### Sampling Options
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-samples` | 1000 | Number of samples |
| `--ode-steps` | 200 | ODE integration steps |
| `--guidance-scale` | 1.5 | Classifier-free guidance scale (1.0 = no guidance) |
| `--mode` | random | Condition sampling mode |

---

## Evaluation

### Metrics
1. **Novelty (Sequence Identity)**: Lower is better (target <40%)
2. **Diversity (Internal)**: Target 30-50%
3. **Condition Control**: Correlation >0.8

### Key Files
- `scripts/evaluate_flow_matching.py` - Main evaluation script
- `mmseqs/bin/mmseqs` - Fast sequence similarity search

---

## Project Structure
```
AMP-FlowRAE/
├── flow_matching_v2/     # Improved Flow Matching model
│   ├── model.py          # Model architecture
│   ├── train.py          # Training script
│   ├── sample.py         # Sampling script
│   └── visualize.py      # Visualization
├── scripts/
│   └── evaluate_flow_matching.py  # Evaluation
├── outputs/
│   ├── flow_v2/         # Training outputs
│   ├── samples_v2/      # Generated samples
│   ├── evaluation_v2/   # Evaluation results
│   └── analysis/        # Visualizations
└── data/
    ├── processed/filter/ # Training data
    └── embeddings/       # ESM3 embeddings
```

---

## Hardware
- GPU: MPS (Apple Silicon) or CUDA
- Check with: `torch.cuda.is_available()` or `torch.backends.mps.is_available()`
