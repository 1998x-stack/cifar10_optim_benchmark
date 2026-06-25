"""
Model factory: register models, build by name.
"""

from core.registry import Registry, build_from_config, register as base_register

register = lambda name: base_register("models", name)


def get_model(name: str, num_classes: int = 10, **kwargs) -> "nn.Module":
    """Factory: build model by registered name."""
    return build_from_config("models", name, num_classes=num_classes, **kwargs)
