"""
Configuration loader: YAML + CLI overrides.

CLI overrides use dot-notation: --optimizer.lr=0.01
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (modifies base in-place)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _set_dot_path(d: dict, path: str, value: Any) -> dict:
    """Set a dotted path like 'optimizer.lr' in a nested dict."""
    keys = path.split(".")
    current = d
    for k in keys[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]
    # Try to cast to appropriate type
    raw = value
    if raw.lower() in ("true", "false"):
        current[keys[-1]] = raw.lower() == "true"
    elif raw.lower() == "none":
        current[keys[-1]] = None
    else:
        try:
            if "." in raw:
                current[keys[-1]] = float(raw)
            else:
                current[keys[-1]] = int(raw)
        except ValueError:
            current[keys[-1]] = raw  # keep as string
    return d


class ConfigLoader:
    """Load config from YAML file and apply CLI overrides."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    def load(self, cli_overrides: Optional[Dict[str, str]] = None) -> dict:
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        if cli_overrides:
            for key, value in cli_overrides.items():
                _set_dot_path(config, key, value)

        self._post_process(config)
        return config

    @staticmethod
    def _post_process(config: dict) -> None:
        """Resolve 'auto' device."""
        if config.get("experiment", {}).get("device") == "auto":
            import torch

            config["experiment"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def parse_cli() -> Dict[str, str]:
        """Parse CLI args into dot-path overrides dict."""
        parser = argparse.ArgumentParser(description="CIFAR-10 Optimizer Benchmark")
        parser.add_argument("--config", type=str, default=None, help="Config YAML path")
        parser.add_argument(
            "overrides",
            nargs="*",
            help="CLI overrides: optimizer.name=AdamW training.epochs=100",
        )
        args, _ = parser.parse_known_args()

        overrides = {}
        if args.overrides:
            for item in args.overrides:
                if "=" in item:
                    key, value = item.split("=", 1)
                    overrides[key] = value

        result = {"config_path": args.config}
        if overrides:
            result["overrides"] = overrides
        return result

    @staticmethod
    def merge_configs(base: dict, overrides: dict) -> dict:
        return _deep_merge(base.copy(), overrides)
