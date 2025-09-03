"""
Core module containing the unified components of the quant system.
This module consolidates all the essential functionality without duplication.
"""

from __future__ import annotations

# Import core symbols, but guard optional modules so CLI can run even if some
# components are missing in minimal environments (e.g., CI, trimmed installs).
from .backtest_engine import UnifiedBacktestEngine
from .cache_manager import UnifiedCacheManager
from .data_manager import UnifiedDataManager

# Optional components: try to import, but continue if absent
PortfolioManager = None
UnifiedResultAnalyzer = None

try:
    # Portfolio manager was moved from portfolio_manager.py to collection_manager.py.
    # Keep public API stable by importing the same symbol from the new module.
    from .collection_manager import PortfolioManager  # type: ignore[import-not-found]
except Exception:
    PortfolioManager = None

try:
    from .result_analyzer import UnifiedResultAnalyzer  # type: ignore[import-not-found]
except Exception:
    UnifiedResultAnalyzer = None

__all__ = [
    "PortfolioManager",
    "UnifiedBacktestEngine",
    "UnifiedCacheManager",
    "UnifiedDataManager",
    "UnifiedResultAnalyzer",
]
