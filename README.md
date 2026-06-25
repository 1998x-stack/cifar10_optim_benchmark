# CIFAR-10 Optimizer Benchmark

Benchmarking PyTorch optimizers (Adam, AdamW, SGD+Momentum, RMSprop) on CIFAR-10 with ResNet-18. Includes a real-time Dash dashboard and static HTML report.

**Live Results:** [1998x-stack.github.io/cifar10_optim_benchmark](https://1998x-stack.github.io/cifar10_optim_benchmark)

## Results

| # | Optimizer | Best Val Acc | Test Acc | Epochs |
|---|-----------|-------------|----------|--------|
| 1 | **AdamW** | **86.24%** | **85.06%** | 50 |
| 2 | Adam | 85.11% | 84.04% | 50 |
| 3 | RMSprop | 83.73% | 82.87% | 50 |
| 4 | SGD+Momentum | 75.93% | 75.65% | 36 (early stop) |

*8x NVIDIA A100-SXM4-80GB, PyTorch 2.6, 50 epochs, CosineAnnealingLR*

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Single optimizer run
python main.py optimizer.name=Adam

# Compare all 4 optimizers (auto-starts dashboard)
python main.py --compare

# Single run with live dashboard
python main.py optimizer.name=SGD --visualize

# Generate report from past run
python main.py --report logs/ResNet18_Adam/
```

## CLI Flags

| Flag | Description |
|------|-------------|
| `--compare` | Run all 4 optimizers + auto-start dashboard |
| `--visualize` | Enable live Dash dashboard for single runs |
| `--no-visualize` | Disable dashboard (even with --compare) |
| `--optimizers Adam SGD` | Compare specific optimizers only |
| `--report LOG_DIR` | Generate static HTML report from existing logs |
| `--config my.yaml` | Use custom config file |

Config overrides use dot notation: `optimizer.lr=0.01 training.epochs=100 model.name=ResNet50`

## Architecture

```
main.py          → CLI + orchestration
core/            → Config loader, registry/factory, metrics, logger
models/          → ResNet18/34/50 (CIFAR-adapted), SimpleCNN
optimizers/      → Adam, AdamW, SGD, RMSprop wrappers
trainers/        → CIFARTrainer with AMP, gradient clipping, early stopping
experiments/     → Single-run and comparison runners
visualization/   → MetricsStore, Plotly figures, Dash dashboard, HTML report
```

All components use a `@register("category", "name")` decorator pattern. Add new optimizers or models by decorating and importing — no config changes needed.

## Visualization

- **Live dashboard:** `python main.py --compare` starts a Plotly Dash server at `http://127.0.0.1:8050`
- **Static report:** Auto-generated after training as `logs/comparison/dashboard.html`
- **Backfill:** `python main.py --report <log-dir>` generates report from historical JSON

## Config

See `config/default.yaml` for all options. Key sections:

```yaml
model:
  name: ResNet18       # ResNet18, ResNet34, ResNet50, SimpleCNN
optimizer:
  name: Adam           # Adam, AdamW, SGD, RMSprop
  lr: 0.001
training:
  epochs: 50
  early_stop_patience: 20
  amp: false           # Automatic Mixed Precision
```
