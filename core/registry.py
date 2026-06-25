"""
Generic registry + factory pattern.

Usage:
    @register("optimizers", "Adam")
    class AdamOptimizerWrapper: ...

    opt_cls = Registry.get("optimizers", "Adam")
    opt = opt_cls(model.parameters(), lr=0.001)
"""

from typing import Any, Callable, Dict, Optional, Type


class Registry:
    """Central registry: category_name -> name -> class/fn."""

    _store: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, category: str, name: str, obj: Any) -> None:
        if category not in cls._store:
            cls._store[category] = {}
        if name in cls._store[category]:
            raise KeyError(
                f"Duplicate registration: '{name}' already in category '{category}'"
            )
        cls._store[category][name] = obj

    @classmethod
    def get(cls, category: str, name: str) -> Any:
        cat = cls._store.get(category)
        if cat is None:
            raise KeyError(f"Category '{category}' not found in registry")
        obj = cat.get(name)
        if obj is None:
            raise KeyError(
                f"'{name}' not found in category '{category}'. "
                f"Available: {list(cat.keys())}"
            )
        return obj

    @classmethod
    def list(cls, category: str) -> list:
        return list(cls._store.get(category, {}).keys())

    @classmethod
    def categories(cls) -> list:
        return list(cls._store.keys())


def register(category: str, name: str):
    """Decorator to register a class or function."""

    def decorator(cls_or_fn):
        Registry.register(category, name, cls_or_fn)
        return cls_or_fn

    return decorator


def build_from_config(
    category: str,
    name: str,
    *args,
    config: Optional[dict] = None,
    **kwargs,
) -> Any:
    """Build an object from registry + optional config dict.

    Looks up 'name' in registry['category'], then instantiates it.
    Config dict is merged: entries in `config` for the specific name
    (e.g. config['adam']) are passed as keyword arguments.
    """
    cls = Registry.get(category, name)
    merged_kw = {}
    if config is not None:
        # merge sub-config for this specific name (lowercased)
        specific = config.get(name.lower(), {})
        if isinstance(specific, dict):
            merged_kw.update(specific)
    merged_kw.update(kwargs)
    return cls(*args, **merged_kw)
