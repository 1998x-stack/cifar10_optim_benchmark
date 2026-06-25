"""
Centralized logging system with file + console + optional TensorBoard / WandB.
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, TextIO, Union


class ExperimentLogger:
    """Manages logging, metrics output, and optional experiment trackers."""

    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: str = "experiment",
        level: Union[int, str] = logging.INFO,
        console: bool = True,
        tensorboard: bool = True,
        wandb: bool = False,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.start_time = datetime.utcnow()
        self.metrics_history: list = []

        # ── python logging setup ──
        self.logger = logging.getLogger(f"exp.{experiment_name}")
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File handler
        log_file = self.log_dir / f"{experiment_name}.log"
        fh = logging.FileHandler(str(log_file))
        fh.setLevel(level)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Console handler
        if console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(level)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        self.info(f"Experiment '{experiment_name}' started at {self.start_time.isoformat()}")
        self.info(f"Log directory: {self.log_dir.resolve()}")

        # ── TensorBoard ──
        self._tb_writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_dir = self.log_dir / "tensorboard"
                tb_dir.mkdir(exist_ok=True)
                self._tb_writer = SummaryWriter(str(tb_dir))
                self.info(f"TensorBoard enabled: {tb_dir}")
            except ImportError:
                self.warning("TensorBoard not available (pip install tensorboard)")

        # ── WandB ──
        self._wandb_run = None
        if wandb:
            try:
                import wandb

                self._wandb_run = wandb.init(project=experiment_name, dir=str(self.log_dir))
                self.info("WandB enabled")
            except ImportError:
                self.warning("WandB not available (pip install wandb)")

    # ── Logging shortcuts ──

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    # ── Metrics ──

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        """Log a dict of metrics at a given step."""
        self.metrics_history.append({"step": step, **metrics})

        # TensorBoard
        if self._tb_writer is not None:
            for key, value in metrics.items():
                tag = f"{prefix}/{key}" if prefix else key
                self._tb_writer.add_scalar(tag, value, step)

        # WandB
        if self._wandb_run is not None:
            import wandb

            self._wandb_run.log({f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}, step=step)

    def log_model_graph(self, model, dummy_input) -> None:
        if self._tb_writer is not None:
            try:
                self._tb_writer.add_graph(model, dummy_input)
            except Exception as e:
                self.warning(f"Failed to log model graph: {e}")

    def log_hyperparams(self, hparams: dict) -> None:
        if self._tb_writer is not None:
            self._tb_writer.add_hparams(hparams, {})
        if self._wandb_run is not None:
            import wandb

            wandb.config.update(hparams)
        self.info(f"Hyperparams: {json.dumps(hparams, indent=2)}")

    def save_metrics_history(self, filename: str = "metrics_history.json") -> None:
        path = self.log_dir / filename
        with open(path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        self.info(f"Metrics history saved to {path} ({len(self.metrics_history)} entries)")

    # ── Timing ──

    def time_since_start(self) -> float:
        return (datetime.utcnow() - self.start_time).total_seconds()

    # ── Cleanup ──

    def close(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._wandb_run is not None:
            import wandb

            wandb.finish()
        elapsed = self.time_since_start()
        self.info(f"Experiment finished. Elapsed: {elapsed:.1f}s")
