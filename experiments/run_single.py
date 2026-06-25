"""
Run a single optimizer experiment (one config).
"""

import json
import os
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

from core.config_loader import ConfigLoader
from core.logger import ExperimentLogger
from models import get_model
from optimizers import get_optimizer
from trainers import CIFARTrainer


def _get_cifar_loaders(config: dict, device: torch.device):
    """Prepare CIFAR-10 train/val/test data loaders."""
    data_cfg = config.get("data", {})
    data_dir = data_cfg.get("data_dir", "./data")

    # CIFAR-10 normalization
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_train = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=data_cfg.get("download", True),
        transform=train_transform,
    )

    # Split train into train/val
    val_split = 1.0 - data_cfg.get("train_val_split", 0.9)
    val_len = int(len(full_train) * val_split)
    train_len = len(full_train) - val_len
    generator = torch.Generator().manual_seed(data_cfg.get("val_seed", 42))
    train_dataset, val_dataset = random_split(
        full_train, [train_len, val_len], generator=generator
    )
    # Apply eval transform to val split
    val_dataset.dataset.transform = eval_transform

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=data_cfg.get("download", True),
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.get("batch_size", 128),
        shuffle=True,
        num_workers=config.get("experiment", {}).get("num_workers", 4),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.get("test_batch_size", 256),
        shuffle=False,
        num_workers=config.get("experiment", {}).get("num_workers", 4),
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_cfg.get("test_batch_size", 256),
        shuffle=False,
        num_workers=config.get("experiment", {}).get("num_workers", 4),
        pin_memory=(device.type == "cuda"),
    )

    return train_loader, val_loader, test_loader


def _get_scheduler(optimizer_wrapper, config: dict):
    """Build learning rate scheduler."""
    sch_cfg = config.get("scheduler", {})
    name = sch_cfg.get("name", "CosineAnnealingLR")
    opt = optimizer_wrapper.optimizer

    if name == "CosineAnnealingLR":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(
            opt,
            T_max=sch_cfg.get("T_max", 200),
            eta_min=sch_cfg.get("eta_min", 0.0),
        )
    elif name == "StepLR":
        from torch.optim.lr_scheduler import StepLR
        return StepLR(opt, step_size=sch_cfg.get("step_size", 30), gamma=sch_cfg.get("gamma", 0.1))
    elif name == "MultiStepLR":
        from torch.optim.lr_scheduler import MultiStepLR
        return MultiStepLR(opt, milestones=sch_cfg.get("milestones", [60, 120, 160]), gamma=sch_cfg.get("gamma", 0.1))
    elif name == "ReduceLROnPlateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(opt, mode="max", patience=10, factor=0.5)
    elif name == "OneCycleLR":
        from torch.optim.lr_scheduler import OneCycleLR
        epochs = config.get("training", {}).get("epochs", 50)
        return OneCycleLR(opt, max_lr=sch_cfg.get("lr", 0.001), epochs=epochs, steps_per_epoch=1)
    else:
        return None


def run_single_experiment(config: dict) -> dict:
    """
    Run one experiment with the given config.
    Returns summary dict with best metrics.

    Args:
        config: Full config dict (loaded from YAML + overrides).
    """
    seed = config.get("experiment", {}).get("seed", 42)
    torch.manual_seed(seed)

    device_name = config.get("experiment", {}).get("device", "cuda")
    device = torch.device(device_name)

    exp_name = (
        f"{config.get('model', {}).get('name', 'ResNet18')}"
        f"_{config.get('optimizer', {}).get('name', 'Adam')}"
    )
    log_dir = Path(config.get("logging", {}).get("log_dir", "./logs")) / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    config["logging"]["log_dir"] = str(log_dir)

    exp_logger = ExperimentLogger(
        log_dir=log_dir,
        experiment_name=exp_name,
        level=config.get("experiment", {}).get("log_level", "INFO"),
        console=config.get("logging", {}).get("console", True),
        tensorboard=config.get("logging", {}).get("tensorboard", True),
        wandb=config.get("logging", {}).get("wandb", False),
    )

    # Log full config
    exp_logger.info(f"Config:\n{json.dumps(config, indent=2)}")

    # Data
    train_loader, val_loader, test_loader = _get_cifar_loaders(config, device)
    exp_logger.info(
        f"Data: {len(train_loader.dataset)} train, "
        f"{len(val_loader.dataset)} val, "
        f"{len(test_loader.dataset)} test"
    )

    # Model
    model_cfg = config.get("model", {})
    model = get_model(
        name=model_cfg.get("name", "ResNet18"),
        num_classes=model_cfg.get("num_classes", 10),
    ).to(device)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    exp_logger.info(f"Model: {model_cfg.get('name')} | Params: {total_params:,} | Trainable: {trainable_params:,}")

    exp_logger.log_model_graph(model, torch.randn(1, 3, 32, 32).to(device))

    # Optimizer
    opt_name = config.get("optimizer", {}).get("name", "Adam")
    optimizer_wrapper = get_optimizer(opt_name, model.parameters(), config)

    exp_logger.info(f"Optimizer: {optimizer_wrapper}")

    # Scheduler
    scheduler = _get_scheduler(optimizer_wrapper, config)

    exp_logger.log_hyperparams({
        "model": model_cfg.get("name"),
        "optimizer": opt_name,
        "lr": config.get("optimizer", {}).get("lr"),
        "weight_decay": config.get("optimizer", {}).get("weight_decay"),
        "batch_size": config.get("data", {}).get("batch_size"),
        "epochs": config.get("training", {}).get("epochs"),
        "scheduler": config.get("scheduler", {}).get("name"),
    })

    # Trainer
    trainer = CIFARTrainer(
        model=model,
        optimizer_wrapper=optimizer_wrapper,
        scheduler=scheduler,
        device=device,
        exp_logger=exp_logger,
        config=config,
    )

    # Train
    metrics = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=config.get("training", {}).get("epochs", 50),
    )

    # Save metrics history
    if config.get("logging", {}).get("metrics_history", True):
        exp_logger.save_metrics_history()

    # Summary
    best_val_acc = max(
        (m.get("acc", 0.0) for m in metrics.get("val", []) if "acc" in m),
        default=0.0,
    )
    test_acc = metrics.get("test", {}).get("acc", 0.0)

    summary = {
        "experiment": exp_name,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "epochs_trained": len(metrics.get("train", [])),
        "config": {k: v for k, v in config.items() if k != "logging"},
    }

    # Save summary
    summary_path = log_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    exp_logger.info(f"Summary saved to {summary_path}")
    exp_logger.close()

    return summary
