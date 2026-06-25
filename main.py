#!/usr/bin/env python3
"""
CIFAR-10 Optimizer Benchmark

Compares Adam, AdamW, SGD+Momentum, and RMSprop using a pluggable factory pattern.

Usage:
    # Single optimizer
    python main.py optimizer.name=Adam

    # Single optimizer with live dashboard
    python main.py optimizer.name=Adam --visualize

    # Comparison (all 4 optimizers) -- auto-starts dashboard
    python main.py --compare

    # Comparison without dashboard
    python main.py --compare --no-visualize

    # Backfill: generate report from historical JSON
    python main.py --report logs/ResNet18_Adam/
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
from visualization.store import MetricsStore
from visualization.dashboard import start_dashboard, stop_dashboard
from visualization.report import generate_report


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 Optimizer Benchmark")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument(
        "--compare", action="store_true",
        help="Run comparison across all optimizers (Adam, AdamW, SGD, RMSprop)",
    )
    parser.add_argument(
        "--optimizers", type=str, nargs="+", default=None,
        help="Optimizers to compare (default: Adam AdamW SGD RMSprop)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Enable live Dash dashboard (auto-enabled with --compare)",
    )
    parser.add_argument(
        "--no-visualize", action="store_true",
        help="Disable dashboard even with --compare",
    )
    parser.add_argument(
        "--report", type=str, default=None,
        help="Generate static HTML report from existing experiment log dir",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="CLI overrides: optimizer.name=AdamW training.epochs=100",
    )
    args = parser.parse_args()

    # -- Backfill mode --
    if args.report:
        _generate_backfill_report(args.report)
        return

    # Parse overrides
    cli_overrides = {}
    for item in args.overrides:
        if "=" in item:
            key, value = item.split("=", 1)
            cli_overrides[key] = value

    # Load config
    loader = ConfigLoader(args.config)
    config = loader.load(cli_overrides)

    # Dashboard: auto-enable for comparison, unless --no-visualize
    use_dashboard = (args.compare or args.visualize) and not args.no_visualize

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
    print(f"  Dashboard: {'Yes' if use_dashboard else 'No'}")
    print("=" * 60)

    # Start dashboard if requested
    store = None
    dashboard_port = None
    if use_dashboard:
        store = MetricsStore()
        dashboard_port = start_dashboard(store, open_browser=True)

    if args.compare:
        optimizers = args.optimizers or ["Adam", "AdamW", "SGD", "RMSprop"]
        print(f"\n  Comparing optimizers: {', '.join(optimizers)}")
        results = run_comparison(config, optimizers=optimizers, store=store)
        # Generate report
        if store is not None:
            from pathlib import Path
            log_dir = Path(config.get("logging", {}).get("log_dir", "./logs"))
            report_path = log_dir / "comparison" / "dashboard.html"
            # Enrich snapshot with final test results
            snap = store.snapshot()
            for opt_name, summary in results.items():
                if opt_name in snap:
                    snap[opt_name]["test_acc"] = summary.get("test_acc", "N/A")
            generate_report(snap, str(report_path))
            print(f"\n  Report saved: {report_path}")
    else:
        summary = run_single_experiment(config, store=store)
        print(f"\n  Single run complete: {summary.get('experiment', '?')}")
        print(f"  Best Val Acc: {summary.get('best_val_acc', 0):.2f}%")
        print(f"  Test Acc:     {summary.get('test_acc', 0):.2f}%")
        # Generate report for single run
        if store is not None:
            from pathlib import Path
            log_dir = Path(config.get("logging", {}).get("log_dir", "./logs"))
            exp_name = f"{model_name}_{opt_name}"
            report_path = log_dir / exp_name / "report.html"
            snap = store.snapshot()
            if exp_name in snap:
                snap[exp_name]["test_acc"] = summary.get("test_acc", "N/A")
            generate_report(snap, str(report_path))
            print(f"  Report saved: {report_path}")

    # Keep alive if dashboard running so user can view charts
    if dashboard_port is not None:
        print(f"\n  Dashboard running at http://127.0.0.1:{dashboard_port}")
        print("  Press Ctrl+C to exit.\n")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n  Shutting down.")


def _generate_backfill_report(log_dir: str):
    """Generate a static HTML report from an existing JSON metrics file."""
    from pathlib import Path
    p = Path(log_dir)
    metrics_file = p / "metrics_history.json"
    if not metrics_file.exists():
        print(f"Error: {metrics_file} not found")
        sys.exit(1)
    metrics = json.loads(metrics_file.read_text())
    # Convert per-epoch metrics list to MetricsStore format
    store = MetricsStore()
    exp_name = p.name  # e.g. "ResNet18_Adam"
    store.ensure_optimizer(exp_name)
    for entry in metrics:
        epoch = entry.get("step", 0)
        store.update(exp_name, epoch=epoch, **{k: v for k, v in entry.items() if k != "step"})
    store.set_status(exp_name, "done")
    report_path = p / "report.html"
    generate_report(store.snapshot(), str(report_path))
    print(f"Report saved: {report_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
