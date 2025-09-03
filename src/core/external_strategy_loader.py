"""
External Strategy Loader

Loads and manages external trading strategies from separate repositories.
Provides unified interface for strategy testing and execution.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExternalStrategyLoader:
    """
    Loads and manages external trading strategies

    Discovers strategy modules from external repositories and provides
    a unified interface for the quant-system to use them.
    """

    def __init__(self, strategies_path: str | None = None):
        """
        Initialize External Strategy Loader

        Args:
            strategies_path: Path to external strategies directory
                           (defaults to ../quant-strategies relative to project root)
        """
        if strategies_path is None:
            # Default to external_strategies directory (mounted in Docker)
            project_root = Path(__file__).parent.parent.parent
            default_strategies_path = project_root / "external_strategies"
            self.strategies_path = default_strategies_path
        else:
            self.strategies_path = Path(strategies_path)
        self.loaded_strategies: dict[str, type] = {}
        self._discover_strategies()

    def _discover_strategies(self) -> None:
        """Discover available strategy modules.

        Prefer importing strategies from the 'algorithms/python' subdirectory of the
        provided strategies_path. If that subdirectory is missing but the provided
        strategies_path itself contains standalone Python strategy files (common when
        mounting ./quant-strategies/algorithms/python directly to the container root),
        fall back to loading .py files from the root of strategies_path.
        This keeps imports safe while allowing flexible mount layouts.
        """
        try:
            alg_py = Path(self.strategies_path) / "algorithms" / "python"
            search_dir = None

            # Primary: explicit algorithms/python directory
            if alg_py.exists() and alg_py.is_dir():
                search_dir = alg_py
            else:
                # Fallback: if the strategies_path itself directly contains .py files,
                # use that directory (handles mounts like ./quant-strategies/algorithms/python:/app/external_strategies)
                sp = Path(self.strategies_path)
                if sp.exists() and any(sp.glob("*.py")):
                    search_dir = sp

            if search_dir is None:
                logger.warning(
                    "algorithms/python directory not found under strategies_path: %s",
                    alg_py,
                )
                return

            for strategy_file in search_dir.glob("*.py"):
                if strategy_file.name.startswith("_"):
                    continue
                self._load_strategy_file(strategy_file)

        except Exception as e:
            logger.error("Error discovering strategies in algorithms/python: %s", e)

    def _load_strategy_file(self, strategy_file: Path) -> None:
        """
        Load a single strategy from a Python file

        Args:
            strategy_file: Path to strategy Python file
        """
        try:
            # Load the strategy module
            strategy_name = strategy_file.stem
            spec = importlib.util.spec_from_file_location(
                f"external_strategy_{strategy_name}", strategy_file
            )
            if spec is None or spec.loader is None:
                logger.error("Could not load spec for %s", strategy_name)
                return

            module = importlib.util.module_from_spec(spec)
            sys.modules[f"external_strategy_{strategy_name}"] = module
            spec.loader.exec_module(module)

            # Look for strategy class in the module
            strategy_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and attr_name.lower().endswith("strategy")
                    and attr_name != "BaseStrategy"
                ):
                    strategy_class = attr
                    break

            if strategy_class:
                self.loaded_strategies[strategy_name] = strategy_class
                logger.info("Loaded external strategy: %s", strategy_name)
            else:
                logger.warning("No strategy class found in %s", strategy_file.name)

        except Exception as e:
            logger.error("Failed to load strategy %s: %s", strategy_file.name, e)

    def _load_strategy_dir(self, strategy_dir: Path) -> None:
        """
        Load a single strategy from directory

        Args:
            strategy_dir: Path to strategy directory
        """
        try:
            # Look for quant_system adapter
            adapter_path = strategy_dir / "adapters" / "quant_system.py"
            if not adapter_path.exists():
                logger.warning(
                    "No quant_system adapter found for %s", strategy_dir.name
                )
                return

            # Load the adapter module
            spec = importlib.util.spec_from_file_location(
                f"{strategy_dir.name}_adapter", adapter_path
            )
            if spec is None or spec.loader is None:
                logger.error("Could not load spec for %s", strategy_dir.name)
                return

            module = importlib.util.module_from_spec(spec)
            sys.modules[f"{strategy_dir.name}_adapter"] = module
            spec.loader.exec_module(module)

            # Find the adapter class (should end with 'Adapter')
            adapter_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and attr_name.endswith("Adapter")
                    and attr_name != "Adapter"
                ):
                    adapter_class = attr
                    break

            if adapter_class is None:
                logger.error("No adapter class found in %s", strategy_dir.name)
                return

            # Store the strategy
            strategy_name = strategy_dir.name.replace("-", "_")
            self.loaded_strategies[strategy_name] = adapter_class
            logger.info("Loaded strategy: %s", strategy_name)

        except Exception as e:
            logger.error("Failed to load strategy %s: %s", strategy_dir.name, e)

    def get_strategy(self, strategy_name: str, **kwargs: Any) -> Any:
        """
        Get a strategy instance by name

        Args:
            strategy_name: Name of the strategy
            **kwargs: Parameters for strategy initialization

        Returns:
            Strategy adapter instance

        Raises:
            ValueError: If strategy not found
        """
        if strategy_name not in self.loaded_strategies:
            available = list(self.loaded_strategies.keys())
            msg = f"Strategy '{strategy_name}' not found. Available: {available}"
            raise ValueError(msg)

        strategy_class = self.loaded_strategies[strategy_name]
        return strategy_class(**kwargs)

    def list_strategies(self) -> list[str]:
        """Get list of available strategy names"""
        return list(self.loaded_strategies.keys())

    def list_strategy_candidates(self) -> list[str]:
        """
        Non-import-based discovery: list candidate strategy names (file stems and dirs)
        without attempting to import them. This is safe in minimal environments and
        useful for CLI discovery (--strategies=all) when imports would fail due to
        missing optional dependencies.
        """
        candidates: set[str] = set()
        try:
            if not self.strategies_path or not Path(self.strategies_path).exists():
                return []
            sp = Path(self.strategies_path)
            # Python files in root
            for f in sp.glob("*.py"):
                if not f.name.startswith("_") and f.name != "README.py":
                    candidates.add(f.stem)
            # Files under algorithms/python
            alg_py = sp / "algorithms" / "python"
            if alg_py.exists():
                for f in alg_py.glob("*.py"):
                    if not f.name.startswith("_"):
                        candidates.add(f.stem)
            # Files under algorithms/original (some are .py)
            alg_orig = sp / "algorithms" / "original"
            if alg_orig.exists():
                for f in alg_orig.glob("*.py"):
                    if not f.name.startswith("_"):
                        candidates.add(f.stem)
            # Directory-based strategies
            for d in sp.iterdir():
                if d.is_dir() and not d.name.startswith("."):
                    candidates.add(d.name.replace("-", "_"))
        except Exception:
            # Best-effort: return whatever we have collected so far
            pass
        return sorted(candidates)

    def get_strategy_info(self, strategy_name: str) -> dict[str, Any]:
        """
        Get information about a strategy

        Args:
            strategy_name: Name of the strategy

        Returns:
            Dictionary with strategy information
        """
        if strategy_name not in self.loaded_strategies:
            msg = f"Strategy '{strategy_name}' not found"
            raise ValueError(msg)

        # Create a temporary instance to get info
        strategy = self.get_strategy(strategy_name)
        if hasattr(strategy, "get_strategy_info"):
            strategy_info = strategy.get_strategy_info()
            return strategy_info if strategy_info is not None else {}
        return {
            "name": strategy_name,
            "type": "External",
            "parameters": getattr(strategy, "parameters", {}),
            "description": f"External strategy: {strategy_name}",
        }

    def validate_strategy_data(self, strategy_name: str, data: Any) -> bool:
        """
        Validate data for a specific strategy

        Args:
            strategy_name: Name of the strategy
            data: Data to validate

        Returns:
            True if data is valid, False otherwise
        """
        strategy = self.get_strategy(strategy_name)
        if hasattr(strategy, "validate_data"):
            result = strategy.validate_data(data)
            return result if isinstance(result, bool) else True
        return True


# Global strategy loader instance
_strategy_loader: ExternalStrategyLoader | None = None


def get_strategy_loader(strategies_path: str | None = None) -> ExternalStrategyLoader:
    """
    Get global strategy loader instance

    Args:
        strategies_path: Path to strategies directory (only used on first call)

    Returns:
        ExternalStrategyLoader instance

    Behavior:
      - If strategies_path is provided, use it.
      - Otherwise prefer the project 'external_strategies' directory (for Docker mounts).
      - If that doesn't exist, fall back to the bundled 'quant-strategies' directory.
      - If neither exists, initialize loader with None (loader will simply have no strategies).
    """
    global _strategy_loader
    if _strategy_loader is None:
        resolved = strategies_path
        if resolved is None:
            # Resolve sensible defaults relative to project root
            project_root = Path(__file__).parent.parent.parent
            external_dir = project_root / "external_strategies"
            quant_dir = project_root / "quant-strategies"
            if external_dir.exists():
                resolved = str(external_dir)
            elif quant_dir.exists():
                resolved = str(quant_dir)
            else:
                resolved = None
        _strategy_loader = ExternalStrategyLoader(resolved)
    return _strategy_loader


def load_external_strategy(strategy_name: str, **kwargs: Any) -> Any:
    """
    Convenience function to load an external strategy

    Args:
        strategy_name: Name of the strategy
        **kwargs: Strategy parameters

    Returns:
        Strategy adapter instance
    """
    loader = get_strategy_loader()
    return loader.get_strategy(strategy_name, **kwargs)
