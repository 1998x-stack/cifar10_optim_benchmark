#!/usr/bin/env python3
"""
CIFAR-10 Optimizer Benchmark

Compares Adam, AdamW, SGD+Momentum, and RMSprop using a pluggable factory pattern.

Usage:
    # Single optimizer
    python main.py optimizer.name=Adam

    # Comparison (all 4 optimizers)
    python main.py --compare

    # Custom config path
    python main.py --config my_config.yaml optimizer.name=SGD

    # Override multiple params
    python main.py --compare optimizer.lr=0.01 training.epochs=100 model.name=ResNet34
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json

from core.config_loader import ConfigLoader
from experiments.run_single import run_single_experiment
from experiments.run_comparison import run_comparison


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 Optimizer Benchmark")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison across all optimizers (Adam, AdamW, SGD, RMSprop)",
    )
    parser.add_argument(
        "--optimizers",
        type=str,
        nargs="+",
        default=None,
        help="Optimizers to compare (default: Adam AdamW SGD RMSprop)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="CLI overrides: optimizer.name=AdamW training.epochs=100",
    )
    args = parser.parse_args()

    # Parse overrides
    cli_overrides = {}
    for item in args.overrides:
        if "=" in item:
            key, value = item.split("=", 1)
            cli_overrides[key] = value

    # Load config
    loader = ConfigLoader(args.config)
    config = loader.load(cli_overrides)

    # Print config
    print("=" * 60)
    print("  CIFAR-10 Optimizer Benchmark")
    print("=" * 60)
    opt_name = config.get("optimizer", {}).get("name", "Adam")
    model_name = config.get("model", {}).get("name", "ResNet18")
    epochs = config.get("training", {}).get("epochs", 50)
    print(f"  Model:    {model_name}")
    print(f"  Optimizer: {opt_name if not args.compare else 'COMPARISON'}")
    print(f"  Epochs:   {epochs}")
    print(f"  Device:   {config.get('experiment', {}).get('device', 'auto')}")
    print(f"  Compare:  {'Yes' if args.compare else 'No (single run)'}")
    print("=" * 60)

    if args.compare:
        optimizers = args.optimizers or ["Adam", "AdamW", "SGD", "RMSprop"]
        print(f"\n  Comparing optimizers: {', '.join(optimizers)}")
        results = run_comparison(config, optimizers=optimizers)
    else:
        summary = run_single_experiment(config)
        print(f"\n  Single run complete: {summary.get('experiment', '?')}")
        print(f"  Best Val Acc: {summary.get('best_val_acc', 0):.2f}%")
        print(f"  Test Acc:     {summary.get('test_acc', 0):.2f}%")


if __name__ == "__main__":
    main()
