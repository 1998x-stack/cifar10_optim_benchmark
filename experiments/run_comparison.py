"""
Run comparison: iterate over optimizers with the same base config, collect results.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

from core.config_loader import ConfigLoader
from .run_single import run_single_experiment


def run_comparison(
    base_config: dict,
    optimizers: Optional[List[str]] = None,
    store=None,
) -> Dict[str, dict]:
    """
    Run the same experiment with each optimizer.

    Args:
        base_config: Config dict (the optimizer.name field is overridden per run).
        optimizers: List of optimizer names, e.g. ["Adam", "AdamW", "SGD", "RMSprop"].

    Returns:
        Dict mapping optimizer name -> summary dict.
    """
    if optimizers is None:
        optimizers = ["Adam", "AdamW", "SGD", "RMSprop"]

    results = {}
    log_dir = Path(base_config.get("logging", {}).get("log_dir", "./logs"))
    compare_dir = log_dir / "comparison"
    compare_dir.mkdir(parents=True, exist_ok=True)

    for opt_name in optimizers:
        print(f"\n{'='*60}")
        print(f"  Running optimizer: {opt_name}")
        print(f"{'='*60}")

        # Clone config and override optimizer
        config = ConfigLoader.merge_configs(base_config, {})
        config["optimizer"]["name"] = opt_name

        # Set a deterministic data_dir seed for fair comparison
        data_cfg = config.get("data", {})

        if store is not None:
            store.ensure_optimizer(opt_name)
            store.set_status(opt_name, "running")

        try:
            summary = run_single_experiment(config, store=store)
            results[opt_name] = summary
            if store is not None:
                store.set_status(opt_name, "done")
        except Exception as e:
            if store is not None:
                store.set_status(opt_name, "error")
            raise

        # Clean GPU cache between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Build comparison report ──
    report = _build_comparison_report(results)
    report_path = compare_dir / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary table
    print("\n")
    print("=" * 70)
    print("  CIFAR-10 OPTIMIZER COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  {'Optimizer':<12} {'Best Val Acc':<15} {'Test Acc':<12} {'Epochs':<8}")
    print(f"  {'-'*12} {'-'*15} {'-'*12} {'-'*8}")
    for opt_name, summary in sorted(
        results.items(), key=lambda x: x[1].get("best_val_acc", 0), reverse=True
    ):
        print(
            f"  {opt_name:<12} "
            f"{summary.get('best_val_acc', 0):>8.2f}%      "
            f"{summary.get('test_acc', 0):>8.2f}%    "
            f"{summary.get('epochs_trained', 0):<8}"
        )
    print("=" * 70)
    print(f"  Full report: {report_path}")
    print()

    return results


def _build_comparison_report(results: Dict[str, dict]) -> dict:
    """Build structured comparison report from all results."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "num_optimizers": len(results),
        "optimizers": list(results.keys()),
        "ranked_by_val_acc": sorted(
            [
                {
                    "optimizer": k,
                    "best_val_acc": v.get("best_val_acc", 0.0),
                    "test_acc": v.get("test_acc", 0.0),
                    "epochs_trained": v.get("epochs_trained", 0),
                }
                for k, v in results.items()
            ],
            key=lambda x: x["best_val_acc"],
            reverse=True,
        ),
        "details": results,
    }
