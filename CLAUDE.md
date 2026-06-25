# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Benchmark PyTorch optimizers (Adam, AdamW, SGD+Momentum, RMSprop) on CIFAR-10 using a pluggable registry/factory architecture.

## Commands

```bash
# Single optimizer run
python main.py optimizer.name=Adam

# Compare all 4 optimizers
python main.py --compare

# Custom config + overrides
python main.py --config my_config.yaml optimizer.name=SGD optimizer.lr=0.01 training.epochs=100

# Compare specific optimizers only
python main.py --compare --optimizers Adam SGD
```

No test suite, linting, or build system exists.

## Architecture

All components use a **registry + factory pattern** (`core/registry.py`). Classes/functions are decorated with `@register("category", "name")` and built via `Registry.get("category", "name")` or `build_from_config(...)`.

### Entry point (`main.py`)
- Parses `--config`, `--compare`, `--optimizers`, and key=value CLI overrides
- CLI overrides use dot-path syntax (`optimizer.lr=0.001`) and are type-coerced (bool, int, float, str)
- Dispatches to `run_single_experiment()` or `run_comparison()`

### Config (`config/default.yaml`)
Central YAML config with sections: `experiment`, `data`, `model`, `optimizer`, `scheduler`, `training`, `logging`. The `device: auto` key resolves to `cuda` if available, else `cpu`. CLI overrides use dot-paths; `ConfigLoader.merge_configs()` does deep dict merge.

### Core module (`core/`)
- **`registry.py`** — `Registry` class with `register()`/`get()`/`list()` + the `@register(category, name)` decorator + `build_from_config()` factory
- **`config_loader.py`** — Loads YAML, applies CLI overrides, resolves `auto` device
- **`metrics.py`** — `MetricAggregator` (running averages with reset) and `MetricsTracker` (per-batch timing + throughput)
- **`logger.py`** — `ExperimentLogger`: per-experiment Python logger with console/file handlers + optional TensorBoard (`torch.utils.tensorboard.SummaryWriter`) and WandB

### Models (`models/`)
- Registered via `@register("models", "Name")`
- **`ResNet18`**, **`ResNet34`** (BasicBlock), **`ResNet50`** (Bottleneck) — CIFAR-adapted: 3×3 initial conv, no maxpool, stride 1
- **`SimpleCNN`** — 3-conv lightweight baseline for sanity checks
- Factory: `get_model("ResNet18", num_classes=10)` calls `build_from_config("models", "ResNet18", num_classes=10)`

### Optimizers (`optimizers/`)
- Registered via `@register("optimizers", "Name")`
- Each is a `BaseOptimizerWrapper` subclass with a `build()` method returning a `torch.optim.*` instance
- **Adam**, **AdamW**, **SGD** (with momentum + Nesterov), **RMSprop**
- Factory: `get_optimizer("Adam", params, config)` reads `config["optimizer"]` and sub-keys (e.g., `config["optimizer"]["adam"]["betas"]`)

### Trainers (`trainers/`)
- **`BaseTrainer`** — Abstract base with train loop, `should_stop_early()`, checkpoint logic (best/last), metric logging
- **`CIFARTrainer`** — Implements `train_epoch()` and `validate()` with optional AMP (`autocast` + `GradScaler`), gradient clipping, label smoothing. The `fit()` method attaches loaders and optionally runs final test evaluation

### Experiments (`experiments/`)
- **`run_single.py`** — End-to-end pipeline: data loaders → model → optimizer → scheduler → trainer → fit. Returns summary dict with `best_val_acc` and `test_acc`
- **`run_comparison.py`** — Iterates over optimizer list, reuses `run_single_experiment()`, prints ranked comparison table, writes `comparison_report.json`

### Data flow
`main.py` → `ConfigLoader.load(overrides)` → `run_single_experiment(config)` → `_get_cifar_loaders()` + `get_model()` + `get_optimizer()` + `_get_scheduler()` → `CIFARTrainer(model, optimizer_wrapper, scheduler, device, logger, config)` → `trainer.fit(loaders, epochs)` → returns summary dict

## Key patterns
- **Adding a new optimizer**: Create `<name>.py` in `optimizers/`, subclass `BaseOptimizerWrapper`, decorate with `@register("optimizers", "Name")`, implement `build()`. The registry picks it up automatically.
- **Adding a new model**: Create or edit a file in `models/`, decorate the factory function/class with `@register("models", "Name")`, import in `models/__init__.py`.
- **CLI overrides** use dot-notation with automatic type coercion: `training.epochs=100` → `int`, `training.amp=true` → `bool`, `optimizer.lr=0.01` → `float`.
- **Device resolution**: Set `device: auto` in YAML for automatic CUDA/CPU selection.
