"""
Abstract base trainer. Subclass to implement train/val loops.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn


class BaseTrainer(ABC):
    """Abstract trainer with common logging/checkpoint infrastructure."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        exp_logger,
        config: dict,
    ):
        self.model = model
        self.device = device
        self.log = exp_logger
        self.config = config
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_metric = -float("inf")
        self.best_epoch = -1
        self.early_stop_counter = 0
        self._stop_training = False

    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch. Returns dict of metrics."""
        ...

    @abstractmethod
    def validate(self) -> Dict[str, float]:
        """Run validation. Returns dict of metrics."""
        ...

    @abstractmethod
    def get_state_dict(self) -> dict:
        """Return checkpoint state dict."""
        ...

    @abstractmethod
    def load_state_dict(self, state: dict) -> None:
        """Load checkpoint state dict."""
        ...

    def should_stop_early(self, val_metric: float) -> bool:
        """Return True if early stopping criterion met."""
        patience = self.config.get("training", {}).get("early_stop_patience", 20)
        if val_metric > self.best_val_metric:
            self.best_val_metric = val_metric
            self.best_epoch = self.current_epoch
            self.early_stop_counter = 0
            return False
        self.early_stop_counter += 1
        if self.early_stop_counter >= patience:
            self.log.info(
                f"Early stopping triggered after {patience} epochs without improvement"
            )
            return True
        return False

    def train(self, train_loader, val_loader, num_epochs: int) -> Dict[str, list]:
        """Full training loop. Returns metrics history."""
        all_metrics = {"train": [], "val": []}

        self.log.info(f"Starting training for {num_epochs} epochs")
        self.log.info(f"Model: {self.model.__class__.__name__}")
        self.log.info(f"Device: {self.device}")

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch

            # Train
            self.model.train()
            train_metrics = self.train_epoch()
            all_metrics["train"].append({"epoch": epoch, **train_metrics})

            # Validate
            val_interval = self.config.get("training", {}).get("val_interval", 1)
            if epoch % val_interval == 0:
                self.model.eval()
                val_metrics = self.validate()
                all_metrics["val"].append({"epoch": epoch, **val_metrics})

                # Log to experiment logger
                self.log.log_metrics(train_metrics, epoch, prefix="train")
                self.log.log_metrics(val_metrics, epoch, prefix="val")

                # Log message
                val_acc = val_metrics.get("acc", 0.0)
                train_loss = train_metrics.get("loss", 0.0)
                self.log.info(
                    f"Epoch {epoch:3d}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Acc: {val_acc:.2f}%"
                )

                # Checkpoint
                self._save_checkpoint(val_metrics)

                # Early stopping
                if self.should_stop_early(val_acc):
                    break
            else:
                self.log.log_metrics(train_metrics, epoch, prefix="train")
                train_loss = train_metrics.get("loss", 0.0)
                self.log.info(
                    f"Epoch {epoch:3d}/{num_epochs} | Train Loss: {train_loss:.4f}"
                )

        self.log.info(
            f"Training complete. Best Val Acc: {self.best_val_metric:.2f}% "
            f"at epoch {self.best_epoch}"
        )
        return all_metrics

    def _save_checkpoint(self, val_metrics: dict) -> None:
        """Save checkpoint if best model."""
        save = self.config.get("logging", {}).get("checkpoints", False)
        if not save:
            return

        import os
        from pathlib import Path

        ckpt_dir = Path(self.config.get("logging", {}).get("log_dir", "./logs")) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        val_acc = val_metrics.get("acc", 0.0)
        state = self.get_state_dict()

        # Save last
        if self.config.get("logging", {}).get("save_last", True):
            torch.save(state, ckpt_dir / "last.pt")

        # Save best
        if val_acc > self.best_val_metric:
            torch.save(state, ckpt_dir / "best.pt")
            self.log.info(f"New best model saved (acc={val_acc:.2f}%)")
