"""
Base optimizer wrapper + factory function.

Each optimizer is registered via @register("optimizers", "Name")
and can be built with:

    opt = build_from_config("optimizers", "Adam", model.parameters(), config=cfg)
"""

from typing import Any, Dict, Iterator, Optional

from core.registry import Registry, build_from_config, register as base_register

# Shortcut decorator
register = lambda name: base_register("optimizers", name)


class BaseOptimizerWrapper:
    """Base class for optimizer wrappers. Subclasses define build()."""

    name: str = "base"

    def __init__(
        self,
        params: Iterator,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay
        self.kwargs = kwargs
        self._optimizer = self.build()

    def build(self):
        """Subclasses override to return a torch.optim.Optimizer instance."""
        raise NotImplementedError

    @property
    def optimizer(self):
        return self._optimizer

    def state_dict(self) -> dict:
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self._optimizer.load_state_dict(state_dict)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self._optimizer.step()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(lr={self.lr}, "
            f"weight_decay={self.weight_decay}, {self.kwargs})"
        )


def get_optimizer(name: str, params, config: dict) -> BaseOptimizerWrapper:
    """
    Factory: build an optimizer wrapper by name, reading config.
    """
    opt_cfg = config.get("optimizer", {})
    lr = opt_cfg.get("lr", 0.001)
    wd = opt_cfg.get("weight_decay", 0.0)
    extra = {k: v for k, v in opt_cfg.items() if k in ("lr", "weight_decay")}

    kwargs = {"lr": lr, "weight_decay": wd}
    # merge per-optimizer sub-config
    sub_cfg = opt_cfg.get(name.lower(), {})
    if isinstance(sub_cfg, dict):
        kwargs.update(sub_cfg)

    return build_from_config("optimizers", name, params, config=opt_cfg, **kwargs)
