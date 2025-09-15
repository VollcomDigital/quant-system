from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path

from ..config import StrategyConfig
from .base import BaseStrategy


def discover_external_strategies(strategies_root: Path) -> dict[str, type[BaseStrategy]]:
    """Discover all BaseStrategy subclasses under the given path.

    Tries two approaches per .py file:
    1) Regular import assuming package structure (root added to sys.path)
    2) Fallback: load module from file via importlib.util.spec_from_file_location
    """
    import sys

    if str(strategies_root) not in sys.path:
        sys.path.insert(0, str(strategies_root))

    found: dict[str, type[BaseStrategy]] = {}

    for py in strategies_root.rglob("*.py"):
        if py.name.startswith("_"):
            continue
        rel = py.relative_to(strategies_root)
        mod_name = ".".join(rel.with_suffix("").parts)
        mod = None
        # Try package import
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            # Fallback: load directly from file path
            try:
                spec = importlib.util.spec_from_file_location(mod_name, str(py))
                if spec and spec.loader:  # type: ignore[attr-defined]
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            except Exception:
                mod = None
        if mod is None:
            continue
        # Native BaseStrategy subclasses
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                name = getattr(obj, "name", obj.__name__)
                found[name] = obj

        # Auto-adapt external classes that define generate_signals(self, data)
        for _, ext_cls in inspect.getmembers(mod, inspect.isclass):
            if ext_cls is BaseStrategy or issubclass(ext_cls, BaseStrategy):
                continue
            if hasattr(ext_cls, "generate_signals") and callable(ext_cls.generate_signals):
                ext_mod_name = mod.__name__
                ext_cls_name = ext_cls.__name__

                def _make_adapter(module_name: str, class_name: str) -> type[BaseStrategy]:
                    from .adapters.ctor_signals_adapter import CtorSignalsAdapter

                    class _AutoAdapter(BaseStrategy):  # type: ignore[misc]
                        name = class_name

                        def param_grid(self) -> dict:
                            return {}

                        def generate_signals(self, df, params):
                            merged = {
                                **params,
                                "external_module": module_name,
                                "external_class": class_name,
                            }
                            return CtorSignalsAdapter().generate_signals(df, merged)

                    _AutoAdapter.__name__ = f"AutoAdapter_{class_name}"
                    return _AutoAdapter

                adapter_cls = _make_adapter(ext_mod_name, ext_cls_name)
                found[ext_cls_name] = adapter_cls
    return found


def load_strategy(
    cfg: StrategyConfig, strategies_root: Path, external_index: dict[str, type[BaseStrategy]]
):
    """Load a strategy class either from module or from external discovery."""
    if cfg.module and cfg.cls:
        mod = importlib.import_module(cfg.module)
        cls = getattr(mod, cfg.cls)
        return cls
    # fallback by name from external index
    if cfg.name in external_index:
        return external_index[cfg.name]
    raise ImportError(f"Strategy not found: {cfg}")
