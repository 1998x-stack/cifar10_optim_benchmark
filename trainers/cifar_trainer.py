"""
CIFAR-10 trainer: implements train_epoch and validate.
"""

from typing import Dict

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from core.metrics import MetricsTracker
from .base import BaseTrainer


class CIFARTrainer(BaseTrainer):
    """Trainer specifically tuned for CIFAR-10 image classification."""

    def __init__(
        self,
        model: nn.Module,
        optimizer_wrapper,
        scheduler,
        device: torch.device,
        exp_logger,
        config: dict,
        criterion=None,
    ):
        super().__init__(model, device, exp_logger, config)
        self.optimizer_wrapper = optimizer_wrapper
        self.optimizer = optimizer_wrapper.optimizer
        self.scheduler = scheduler
        self.criterion = criterion or nn.CrossEntropyLoss()

        # AMP
        self.amp_enabled = config.get("training", {}).get("amp", False)
        self.scaler = GradScaler(enabled=self.amp_enabled)

        # Label smoothing
        ls = config.get("training", {}).get("label_smoothing", 0.0)
        if ls > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=ls)

        self.grad_clip = config.get("training", {}).get("gradient_clip_norm", 0.0)

    def train_epoch(self) -> Dict[str, float]:
        tracker = MetricsTracker()

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            tracker.tick()
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            if self.amp_enabled:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.optimizer.step()

            acc = (outputs.argmax(dim=1) == targets).float().mean().item() * 100.0
            tracker.tock()
            tracker.update(loss=loss.item(), acc=acc)
            self.global_step += 1

        if self.scheduler is not None:
            if hasattr(self.scheduler, "get_last_lr"):
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = 0.0
            self.log.debug(f"LR after epoch {self.current_epoch}: {current_lr:.6f}")
            self.scheduler.step()

        metrics = tracker.epoch_averages()
        return metrics

    def validate(self) -> Dict[str, float]:
        tracker = MetricsTracker()

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                tracker.tick()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                acc = (outputs.argmax(dim=1) == targets).float().mean().item() * 100.0
                tracker.tock()
                tracker.update(loss=loss.item(), acc=acc)

        return tracker.epoch_averages()

    def fit(
        self,
        train_loader,
        val_loader,
        test_loader=None,
        num_epochs: int = 50,
    ) -> Dict[str, list]:
        """Convenience: attach loaders, run train, optionally test."""
        self.train_loader = train_loader
        self.val_loader = val_loader
        metrics = self.train(train_loader, val_loader, num_epochs)

        if test_loader is not None:
            test_metrics = self.test(test_loader)
            metrics["test"] = test_metrics
            self.log.log_metrics(test_metrics, self.current_epoch, prefix="test")
            self.log.info(
                f"Test Acc: {test_metrics.get('acc', 0.0):.2f}% | "
                f"Test Loss: {test_metrics.get('loss', 0.0):.4f}"
            )

        return metrics

    def test(self, test_loader) -> Dict[str, float]:
        tracker = MetricsTracker()
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                tracker.tick()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                acc = (outputs.argmax(dim=1) == targets).float().mean().item() * 100.0
                tracker.tock()
                tracker.update(loss=loss.item(), acc=acc)
        return tracker.epoch_averages()

    def get_state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": self.current_epoch,
            "best_val_metric": self.best_val_metric,
        }

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") and self.scheduler:
            self.scheduler.load_state_dict(state["scheduler"])
        self.current_epoch = state.get("epoch", 0)
        self.best_val_metric = state.get("best_val_metric", 0.0)
