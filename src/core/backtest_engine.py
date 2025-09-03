"""
Unified Backtest Engine - Consolidates all backtesting functionality.
Supports single assets, portfolios, parallel processing, and optimization.
"""

from __future__ import annotations

import concurrent.futures
import gc
import logging
import multiprocessing as mp
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from backtesting import Backtest
from backtesting.lib import SignalStrategy

from .cache_manager import UnifiedCacheManager
from .data_manager import UnifiedDataManager
from .result_analyzer import UnifiedResultAnalyzer

# from numba import jit  # Removed for compatibility


warnings.filterwarnings("ignore")

# Defaults
from pathlib import Path

# Default metric used when none specified in manifest
DEFAULT_METRIC = "sortino_ratio"


def _run_backtest_worker(args):
    """
    Module-level worker for ProcessPoolExecutor to avoid pickling bound methods.
    args: (symbol, strategy, cfg_kwargs)
    Returns a serializable dict with result metadata.
    """
    symbol, strategy, cfg_kwargs = args
    try:
        # Import inside worker process
        from .backtest_engine import (
            BacktestConfig,  # type: ignore[import-not-found]
            UnifiedBacktestEngine,  # type: ignore[import-not-found]
        )
    except Exception:
        # Fallback if imports fail in worker - return error
        return {
            "symbol": symbol,
            "strategy": strategy,
            "error": "Worker imports failed",
        }

    try:
        # Construct config inside worker (safe to create per-process)
        try:
            cfg = BacktestConfig(**cfg_kwargs)  # type: ignore[call-arg]
        except Exception:
            # Fallback minimal config object
            class _TmpCfg:
                def __init__(self, **kw):
                    self.__dict__.update(kw)

            cfg = _TmpCfg(**cfg_kwargs)

        # Initialize external strategy loader in the worker process if a path was provided.
        # This ensures StrategyFactory / external loader can discover strategies without
        # relying on the parent process to have initialized the global loader.
        try:
            strategies_path = None
            if isinstance(cfg_kwargs, dict):
                strategies_path = cfg_kwargs.get("strategies_path")
            else:
                strategies_path = getattr(cfg, "strategies_path", None)

            if strategies_path:
                try:
                    from pathlib import Path as _Path  # local import

                    from .external_strategy_loader import (
                        get_strategy_loader,  # type: ignore[import-not-found]
                    )

                    # Try a set of common candidate locations under the provided strategies_path
                    candidates = []
                    try:
                        candidates.append(strategies_path)
                        candidates.append(
                            str(_Path(strategies_path) / "algorithms" / "python")
                        )
                        candidates.append(
                            str(_Path(strategies_path) / "algorithms" / "original")
                        )
                    except Exception:
                        pass

                    loader_initialized = False
                    for cand in candidates:
                        if not cand:
                            continue
                        try:
                            cand_path = _Path(cand)
                            if cand_path.exists():
                                # Initialize the global loader in this worker process using the candidate path
                                get_strategy_loader(str(cand_path))
                                loader_initialized = True
                                break
                        except Exception as exc:
                            # ignore and try next candidate, but log for diagnostics
                            log = logging.getLogger(__name__)
                            log.debug(
                                "Strategy loader init failed for %s: %s", cand, exc
                            )
                            continue

                    # As a final attempt, call get_strategy_loader with the original value
                    if not loader_initialized:
                        try:
                            get_strategy_loader(strategies_path)
                        except Exception as exc:
                            log = logging.getLogger(__name__)
                            log.debug("Final strategy loader init failed: %s", exc)

                except Exception:
                    # Non-fatal: continue without external strategies
                    pass
        except Exception:
            pass

        engine = UnifiedBacktestEngine()
        res = engine.run_backtest(symbol, strategy, cfg)
        # Build serializable payload for parent process
        metrics = res.metrics if getattr(res, "metrics", None) is not None else {}
        trades_raw = None
        equity_raw = None
        try:
            import json as _json

            import pandas as _pd

            trades_obj = getattr(res, "trades", None)
            if trades_obj is not None:
                if isinstance(trades_obj, _pd.DataFrame):
                    trades_raw = trades_obj.to_csv(index=False)
                else:
                    try:
                        trades_raw = _json.dumps(trades_obj)
                    except Exception:
                        trades_raw = str(trades_obj)

            eq = getattr(res, "equity_curve", None)
            if eq is not None and isinstance(eq, _pd.DataFrame):
                equity_raw = eq.to_json(orient="records", date_format="iso")
            elif eq is not None:
                try:
                    equity_raw = _json.dumps(eq)
                except Exception:
                    equity_raw = str(eq)
        except Exception:
            trades_raw = None
            equity_raw = None

        # Provide a compact, JSON-friendly summary of the backtest result for persistence/inspection
        try:
            bt_results_raw = {
                "metrics": metrics,
                "duration_seconds": getattr(res, "duration_seconds", None),
                "data_points": getattr(res, "data_points", None),
                "parameters": getattr(res, "parameters", None),
                # include a lightweight final value if available on the result object
                "final_value": None,
            }
            try:
                if getattr(res, "equity_curve", None) is not None:
                    # If equity_curve is a DataFrame, try to capture the last equity point
                    eq = res.equity_curve
                    if hasattr(eq, "iloc") and len(eq) > 0:
                        last_row = eq.iloc[-1]
                        # try both 'equity' column or the first numeric column
                        if "equity" in last_row:
                            bt_results_raw["final_value"] = float(last_row["equity"])
                        else:
                            # pick first numeric-like column
                            for v in last_row.values:
                                try:
                                    bt_results_raw["final_value"] = float(v)
                                    break
                                except Exception as exc:
                                    logging.getLogger(__name__).debug(
                                        "Failed to extract final_value: %s", exc
                                    )
                                    continue
            except Exception:
                # best-effort only
                pass
        except Exception:
            bt_results_raw = None

        return {
            "symbol": getattr(res, "symbol", symbol),
            "strategy": getattr(res, "strategy", strategy),
            "metrics": metrics,
            "trades_raw": trades_raw,
            "equity_raw": equity_raw,
            "bt_results_raw": bt_results_raw,
            "error": getattr(res, "error", None),
            "duration_seconds": getattr(res, "duration_seconds", None),
            "data_points": getattr(res, "data_points", None),
        }
    except Exception as exc:
        return {"symbol": symbol, "strategy": strategy, "error": str(exc)}


def create_backtesting_strategy_adapter(strategy_instance):
    """Create a backtesting library compatible strategy from our strategy instance."""

    class StrategyAdapter(SignalStrategy):
        """Adapter to make our strategies work with the backtesting library."""

        def init(self):
            """Initialize the strategy with our custom logic."""
            # Get the data in the format our strategies expect (uppercase columns)
            strategy_data = pd.DataFrame(
                {
                    "Open": self.data.Open,
                    "High": self.data.High,
                    "Low": self.data.Low,
                    "Close": self.data.Close,
                    "Volume": self.data.Volume,
                },
                index=self.data.index,
            )

            # Generate signals using our strategy
            try:
                signals = strategy_instance.generate_signals(strategy_data)
                # Ensure signals are aligned with data index
                if isinstance(signals, pd.Series):
                    aligned_signals = signals.reindex(self.data.index, fill_value=0)
                else:
                    aligned_signals = pd.Series(
                        signals, index=self.data.index, dtype=float
                    )

                self.signals = self.I(lambda: aligned_signals.values, name="signals")
            except Exception:
                # If strategy fails, create zero signals
                self.signals = self.I(lambda: [0] * len(self.data), name="signals")

        def next(self):
            """Execute trades based on our strategy signals."""
            if len(self.signals) > 0:
                current_signal = self.signals[-1]

                if current_signal == 1 and not self.position:
                    # Buy signal and no position - go long
                    self.buy()
                elif current_signal == -1 and self.position:
                    # Sell signal and have position - close position
                    self.sell()
                elif current_signal == -1 and not self.position:
                    # Sell signal and no position - go short (if allowed)
                    try:
                        self.sell()
                    except:
                        pass  # Shorting not allowed or failed

    return StrategyAdapter


@dataclass
class BacktestConfig:
    """Configuration for backtest runs."""

    symbols: list[str]
    strategies: list[str]
    start_date: str
    end_date: str
    initial_capital: float = 10000
    interval: str = "1d"
    commission: float = 0.001
    use_cache: bool = True
    save_trades: bool = False
    save_equity_curve: bool = False
    override_old_trades: bool = (
        True  # Whether to clean up old trades for same symbol/strategy
    )
    memory_limit_gb: float = 8.0
    max_workers: int = None
    asset_type: str = None  # 'stocks', 'crypto', 'forex', etc.
    futures_mode: bool = False  # For crypto futures
    leverage: float = 1.0  # For futures trading


@dataclass
class BacktestResult:
    """Standardized backtest result."""

    symbol: str
    strategy: str
    parameters: dict[str, Any]
    metrics: dict[str, float]
    config: BacktestConfig
    equity_curve: pd.DataFrame | None = None
    trades: pd.DataFrame | None = None
    start_date: str = None
    end_date: str = None
    duration_seconds: float = 0
    data_points: int = 0
    error: str | None = None
    source: str | None = None


class UnifiedBacktestEngine:
    """
    Unified backtesting engine that consolidates all backtesting functionality.
    Supports single assets, portfolios, parallel processing, and various asset types.
    """

    def __init__(
        self,
        data_manager: UnifiedDataManager = None,
        cache_manager: UnifiedCacheManager = None,
        max_workers: int | None = None,
        memory_limit_gb: float = 8.0,
    ):
        self.data_manager = data_manager or UnifiedDataManager()
        self.cache_manager = cache_manager or UnifiedCacheManager()
        self.result_analyzer = UnifiedResultAnalyzer()

        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.memory_limit_bytes = int(memory_limit_gb * 1024**3)

        self.logger = logging.getLogger(__name__)
        self.stats = {
            "backtests_run": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "total_time": 0,
        }

    def run_backtest(
        self,
        symbol: str,
        strategy: str,
        config: BacktestConfig,
        custom_parameters: dict[str, Any] | None = None,
    ) -> BacktestResult:
        """
        Run backtest for a single symbol/strategy combination.

        Args:
            symbol: Symbol to backtest
            strategy: Strategy name
            config: Backtest configuration
            custom_parameters: Custom strategy parameters

        Returns:
            BacktestResult object
        """
        start_time = time.time()

        try:
            # Get strategy parameters
            parameters = custom_parameters or self._get_default_parameters(strategy)

            # Check cache first
            if config.use_cache and not custom_parameters:
                cached_result = self.cache_manager.get_backtest_result(
                    symbol, strategy, parameters, config.interval
                )
                if cached_result:
                    self.stats["cache_hits"] += 1
                    self.logger.debug("Cache hit for %s/%s", symbol, strategy)
                    # Convert cached dict to BacktestResult and mark it as coming from cache
                    res = self._dict_to_result(
                        cached_result, symbol, strategy, parameters, config
                    )
                    try:
                        res.from_cache = True
                    except Exception:
                        pass
                    return res

            self.stats["cache_misses"] += 1

            # Get market data
            if config.futures_mode:
                data = self.data_manager.get_crypto_futures_data(
                    symbol,
                    config.start_date,
                    config.end_date,
                    config.interval,
                    config.use_cache,
                )
            else:
                data = self.data_manager.get_data(
                    symbol,
                    config.start_date,
                    config.end_date,
                    config.interval,
                    config.use_cache,
                    config.asset_type,
                )

            if data is None or data.empty:
                return BacktestResult(
                    symbol=symbol,
                    strategy=strategy,
                    parameters=parameters,
                    config=config,
                    metrics={},
                    error="No data available",
                )

            # Run backtest
            result = self._execute_backtest(symbol, strategy, data, parameters, config)

            # Cache result if not using custom parameters
            # NOTE: Backtest output caching is disabled to ensure results are always
            # recomputed and persisted per-run. Data-level caching (market data) is
            # preserved. If desired, re-enable result caching here.
            # if config.use_cache and not custom_parameters and not result.error:
            #     self.cache_manager.cache_backtest_result(
            #         symbol, strategy, parameters, asdict(result), config.interval
            #     )

            result.duration_seconds = time.time() - start_time
            result.data_points = len(data)
            self.stats["backtests_run"] += 1

            return result

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error("Backtest failed for %s/%s: %s", symbol, strategy, e)
            return BacktestResult(
                symbol=symbol,
                strategy=strategy,
                parameters=custom_parameters or {},
                config=config,
                metrics={},
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    def run_batch_backtests(self, config: BacktestConfig) -> list[BacktestResult]:
        """
        Run backtests for multiple symbols and strategies in parallel.

        Args:
            config: Backtest configuration

        Returns:
            List of backtest results
        """
        start_time = time.time()
        self.logger.info(
            "Starting batch backtest: %d symbols, %d strategies",
            len(config.symbols),
            len(config.strategies),
        )

        # Generate all symbol/strategy combinations
        combinations = [
            (symbol, strategy)
            for symbol in config.symbols
            for strategy in config.strategies
        ]

        self.logger.info("Total combinations: %d", len(combinations))

        # Process in batches to manage memory
        batch_size = self._calculate_batch_size(
            len(config.symbols), config.memory_limit_gb
        )
        results = []

        for i in range(0, len(combinations), batch_size):
            batch = combinations[i : i + batch_size]
            self.logger.info(
                "Processing batch %d/%d",
                i // batch_size + 1,
                (len(combinations) - 1) // batch_size + 1,
            )

            batch_results = self._process_batch(batch, config)
            results.extend(batch_results)

            # Force garbage collection between batches
            gc.collect()

        self.stats["total_time"] = time.time() - start_time
        self._log_stats()

        return results

    def run_portfolio_backtest(
        self, config: BacktestConfig, weights: dict[str, float] | None = None
    ) -> BacktestResult:
        """
        Run portfolio backtest with multiple assets.

        Args:
            config: Backtest configuration
            weights: Asset weights (if None, equal weights used)

        Returns:
            Portfolio backtest result
        """
        start_time = time.time()

        if not config.strategies or len(config.strategies) != 1:
            raise ValueError("Portfolio backtest requires exactly one strategy")

        strategy = config.strategies[0]

        try:
            # Get data for all symbols
            all_data = self.data_manager.get_batch_data(
                config.symbols,
                config.start_date,
                config.end_date,
                config.interval,
                config.use_cache,
                config.asset_type,
            )

            if not all_data:
                return BacktestResult(
                    symbol="PORTFOLIO",
                    strategy=strategy,
                    parameters={},
                    config=config,
                    metrics={},
                    error="No data available for any symbol",
                )

            # Calculate equal weights if not provided
            if not weights:
                weights = {symbol: 1.0 / len(all_data) for symbol in all_data}

            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}

            # Run portfolio backtest
            portfolio_result = self._execute_portfolio_backtest(
                all_data, strategy, weights, config
            )

            portfolio_result.duration_seconds = time.time() - start_time
            return portfolio_result

        except Exception as e:
            self.logger.error("Portfolio backtest failed: %s", e)
            return BacktestResult(
                symbol="PORTFOLIO",
                strategy=strategy,
                parameters={},
                config=config,
                metrics={},
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    def run_incremental_backtest(
        self,
        symbol: str,
        strategy: str,
        config: BacktestConfig,
        last_update: datetime | None = None,
    ) -> BacktestResult | None:
        """
        Run incremental backtest - only process new data since last run.

        Args:
            symbol: Symbol to backtest
            strategy: Strategy name
            config: Backtest configuration
            last_update: Last update timestamp

        Returns:
            BacktestResult or None if no new data
        """
        # Check if we have cached results
        parameters = self._get_default_parameters(strategy)
        cached_result = self.cache_manager.get_backtest_result(
            symbol, strategy, parameters, config.interval
        )

        if cached_result and not last_update:
            self.logger.info("Using cached result for %s/%s", symbol, strategy)
            return self._dict_to_result(
                cached_result, symbol, strategy, parameters, config
            )

        # Get data and check if we need to update
        data = self.data_manager.get_data(
            symbol,
            config.start_date,
            config.end_date,
            config.interval,
            config.use_cache,
            config.asset_type,
        )

        if data is None or data.empty:
            return BacktestResult(
                symbol=symbol,
                strategy=strategy,
                parameters=parameters,
                config=config,
                metrics={},
                error="No data available",
            )

        # Check if we have new data since last cached result
        if cached_result and last_update:
            last_data_point = pd.to_datetime(
                cached_result.get("end_date", config.start_date), utc=True
            )

            # Ensure data index is in UTC for comparison
            data_last_point = data.index[-1]
            if data_last_point.tz is None:
                data_last_point = data_last_point.tz_localize("UTC")
            else:
                data_last_point = data_last_point.tz_convert("UTC")

            if data_last_point <= last_data_point:
                self.logger.info("No new data for %s/%s", symbol, strategy)
                return self._dict_to_result(
                    cached_result, symbol, strategy, parameters, config
                )

        # Run backtest
        return self.run_backtest(symbol, strategy, config)

    def _execute_backtest(
        self,
        symbol: str,
        strategy: str,
        data: pd.DataFrame,
        parameters: dict[str, Any],
        config: BacktestConfig,
    ) -> BacktestResult:
        """Execute the actual backtest logic."""
        try:
            # Get strategy class
            strategy_class = self._get_strategy_class(strategy)
            if not strategy_class:
                return BacktestResult(
                    symbol=symbol,
                    strategy=strategy,
                    parameters=parameters,
                    config=config,
                    metrics={},
                    error=f"Strategy {strategy} not found",
                )

            # Initialize strategy
            strategy_instance = strategy_class(**parameters)

            # Prepare data for backtesting library (requires uppercase OHLCV)
            bt_data = self._prepare_data_for_backtesting_lib(data)

            if bt_data is None or bt_data.empty:
                return BacktestResult(
                    symbol=symbol,
                    strategy=strategy,
                    parameters=parameters,
                    config=config,
                    metrics={},
                    error="Data preparation failed",
                )

            # Create strategy adapter for backtesting library
            StrategyAdapter = create_backtesting_strategy_adapter(strategy_instance)

            # Run backtest using the backtesting library
            bt = Backtest(
                bt_data,
                StrategyAdapter,
                cash=config.initial_capital,
                commission=config.commission,
                exclusive_orders=True,
            )

            # Execute backtest
            bt_results = bt.run()

            # Convert backtesting library results to our format
            result = self._convert_backtesting_results(bt_results, bt_data, config)

            # Extract metrics from backtesting library results
            metrics = self._extract_metrics_from_bt_results(bt_results)

            return BacktestResult(
                symbol=symbol,
                strategy=strategy,
                parameters=parameters,
                config=config,
                metrics=metrics,
                equity_curve=(
                    result.get("equity_curve") if config.save_equity_curve else None
                ),
                trades=result.get("trades") if config.save_trades else None,
                start_date=config.start_date,
                end_date=config.end_date,
            )

        except Exception as e:
            return BacktestResult(
                symbol=symbol,
                strategy=strategy,
                parameters=parameters,
                config=config,
                metrics={},
                error=str(e),
            )

    def _execute_portfolio_backtest(
        self,
        data_dict: dict[str, pd.DataFrame],
        strategy: str,
        weights: dict[str, float],
        config: BacktestConfig,
    ) -> BacktestResult:
        """Execute portfolio backtest."""
        try:
            # Align all data to common date range
            aligned_data = self._align_portfolio_data(data_dict)

            if aligned_data.empty:
                return BacktestResult(
                    symbol="PORTFOLIO",
                    strategy=strategy,
                    parameters=weights,
                    config=config,
                    metrics={},
                    error="No aligned data for portfolio",
                )

            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(aligned_data, weights)

            # Create portfolio equity curve
            initial_capital = config.initial_capital
            equity_curve = (1 + portfolio_returns).cumprod() * initial_capital

            # Calculate portfolio metrics
            portfolio_data = {
                "returns": portfolio_returns,
                "equity_curve": equity_curve,
                "weights": weights,
            }

            metrics = self.result_analyzer.calculate_portfolio_metrics(
                portfolio_data, initial_capital
            )

            return BacktestResult(
                symbol="PORTFOLIO",
                strategy=strategy,
                parameters=weights,
                config=config,
                metrics=metrics,
                equity_curve=(
                    equity_curve.to_frame("equity")
                    if config.save_equity_curve
                    else None
                ),
            )

        except Exception as e:
            return BacktestResult(
                symbol="PORTFOLIO",
                strategy=strategy,
                parameters=weights,
                config=config,
                metrics={},
                error=str(e),
            )

    def _process_batch(
        self, batch: list[tuple[str, str]], config: BacktestConfig
    ) -> list[BacktestResult]:
        """Process batch of symbol/strategy combinations.

        Uses a module-level worker to avoid pickling bound methods or objects that
        are not serializable by multiprocessing. Each worker constructs its own
        engine and runs the single backtest there.
        """
        results: list[BacktestResult] = []

        # Build serializable cfg_kwargs for workers (they will construct BacktestConfig)
        for i in range(
            0, len(batch), max(1, len(batch))
        ):  # keep batching but here we pass full batch to executor.map
            # Prepare args for each (symbol, strategy)
            worker_args = []
            for symbol, strategy in batch:
                cfg_kwargs = {
                    "symbols": [symbol],
                    "strategies": [strategy],
                    "start_date": config.start_date,
                    "end_date": config.end_date,
                    "period": getattr(config, "period", None),
                    "initial_capital": getattr(config, "initial_capital", 10000),
                    "interval": getattr(config, "interval", "1d"),
                    "max_workers": getattr(config, "max_workers", None),
                    # propagate strategies_path from parent cfg (may be present on _TmpCfg)
                    "strategies_path": getattr(config, "strategies_path", None),
                    # include commonly expected config attributes so worker-side _TmpCfg has them
                    "use_cache": getattr(config, "use_cache", True),
                    "commission": getattr(config, "commission", 0.001),
                    "save_trades": getattr(config, "save_trades", False),
                    "save_equity_curve": getattr(config, "save_equity_curve", False),
                    # Additional worker-facing attributes to avoid attribute errors in fallback _TmpCfg
                    "override_old_trades": getattr(config, "override_old_trades", True),
                    "memory_limit_gb": getattr(config, "memory_limit_gb", 8.0),
                    "asset_type": getattr(config, "asset_type", None),
                    "futures_mode": getattr(config, "futures_mode", False),
                    "leverage": getattr(config, "leverage", 1.0),
                }
                worker_args.append((symbol, strategy, cfg_kwargs))

            # Use ProcessPoolExecutor with module-level worker to avoid pickling issues
            try:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.max_workers
                ) as executor:
                    for worker_res in executor.map(_run_backtest_worker, worker_args):
                        # worker_res is a serializable dict
                        sym = worker_res.get("symbol")
                        strat = worker_res.get("strategy")
                        err = worker_res.get("error")
                        metrics = worker_res.get("metrics", {}) or {}
                        duration = worker_res.get("duration_seconds", None)
                        data_points = worker_res.get("data_points", None)

                        if err:
                            self.logger.error(
                                "Batch backtest failed for %s/%s: %s", sym, strat, err
                            )
                            self.stats["errors"] += 1
                            results.append(
                                BacktestResult(
                                    symbol=sym or "",
                                    strategy=strat or "",
                                    parameters={},
                                    config=config,
                                    metrics={},
                                    error=err,
                                )
                            )
                        else:
                            # Construct a minimal BacktestResult for downstream processing
                            br = BacktestResult(
                                symbol=sym or "",
                                strategy=strat or "",
                                parameters={},
                                config=config,
                                metrics=metrics,
                                trades=worker_res.get("trades_raw"),
                                start_date=getattr(config, "start_date", None),
                                end_date=getattr(config, "end_date", None),
                                duration_seconds=duration or 0,
                                data_points=int(data_points)
                                if data_points is not None
                                else 0,
                                error=None,
                            )
                            # Attach raw backtest payloads if present so engine.run can persist them
                            try:
                                br.bt_results_raw = worker_res.get(
                                    "bt_results_raw", None
                                )
                            except Exception:
                                pass
                            # Reflect worker-level cache hits in parent engine stats
                            try:
                                if worker_res.get("cache_hit"):
                                    self.stats["cache_hits"] += 1
                            except Exception:
                                pass
                            results.append(br)
            except Exception as e:
                self.logger.error("Failed to execute worker batch: %s", e)
                # Convert all batch items to error BacktestResult
                for symbol, strategy in batch:
                    self.stats["errors"] += 1
                    results.append(
                        BacktestResult(
                            symbol=symbol,
                            strategy=strategy,
                            parameters={},
                            config=config,
                            metrics={},
                            error=str(e),
                        )
                    )

            # we've processed the whole provided batch once; break
            break

        return results

    def _run_single_backtest_task(
        self, symbol: str, strategy: str, config: BacktestConfig
    ) -> BacktestResult:
        """Task function for multiprocessing."""
        # Create new instances for this process
        data_manager = UnifiedDataManager()
        cache_manager = UnifiedCacheManager()

        # Create temporary engine for this process
        temp_engine = UnifiedBacktestEngine(data_manager, cache_manager, max_workers=1)
        return temp_engine.run_backtest(symbol, strategy, config)

    def _prepare_data_with_indicators(
        self, data: pd.DataFrame, strategy_instance
    ) -> pd.DataFrame:
        """Prepare data with technical indicators required by strategy."""
        prepared_data = data.copy()

        # Add basic indicators that most strategies need
        prepared_data = self._add_basic_indicators(prepared_data)

        # Add strategy-specific indicators
        if hasattr(strategy_instance, "add_indicators"):
            prepared_data = strategy_instance.add_indicators(prepared_data)

        return prepared_data

    def _add_basic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators."""
        df = data.copy()

        # Simple moving averages
        for period in [10, 20, 50]:
            df[f"sma_{period}"] = df["close"].rolling(period).mean()

        # RSI
        df["rsi_14"] = self._calculate_rsi(df["close"].values, 14)

        # MACD
        macd_line, signal_line, histogram = self._calculate_macd(df["close"].values)
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_histogram"] = histogram

        # Bollinger Bands
        sma_20 = df["close"].rolling(20).mean()
        std_20 = df["close"].rolling(20).std()
        df["bb_upper"] = sma_20 + (std_20 * 2)
        df["bb_lower"] = sma_20 - (std_20 * 2)
        df["bb_middle"] = sma_20

        return df

    def _simulate_trading(
        self, data: pd.DataFrame, strategy_instance, config: BacktestConfig
    ) -> dict[str, Any]:
        """Simulate trading based on strategy signals."""
        trades = []
        equity_curve = []

        capital = config.initial_capital
        position = 0
        position_size = 0

        # Pre-generate all signals for the entire dataset
        try:
            strategy_data = self._transform_data_for_strategy(data)
            all_signals = strategy_instance.generate_signals(strategy_data)
        except Exception as e:
            self.logger.debug(
                "Strategy %s failed: %s", strategy_instance.__class__.__name__, e
            )
            # If strategy fails, create zero signals
            all_signals = pd.Series(0, index=data.index)

        for i, (timestamp, row) in enumerate(data.iterrows()):
            # Get pre-generated signal for this timestamp
            signal = all_signals.iloc[i] if i < len(all_signals) else 0

            # Execute trades based on signal
            if signal == 1 and position <= 0:  # Buy signal
                if position < 0:  # Close short position
                    pnl = (position_size * row["close"] - position_size * position) * -1
                    capital += pnl
                    trades.append(
                        {
                            "timestamp": timestamp,
                            "action": "cover",
                            "price": row["close"],
                            "size": abs(position_size),
                            "pnl": pnl,
                        }
                    )

                # Open long position - use full capital minus commission for BuyAndHold
                available_capital = capital / (
                    1 + config.commission
                )  # Account for commission in calculation
                position_size = available_capital / row["close"]
                position = row["close"]
                capital -= position_size * row["close"] + (
                    position_size * row["close"] * config.commission
                )

                trades.append(
                    {
                        "timestamp": timestamp,
                        "action": "buy",
                        "price": row["close"],
                        "size": position_size,
                        "pnl": 0,
                    }
                )

            elif signal == -1 and position >= 0:  # Sell signal
                if position > 0:  # Close long position
                    pnl = position_size * (row["close"] - position)
                    capital += pnl + (position_size * row["close"])
                    trades.append(
                        {
                            "timestamp": timestamp,
                            "action": "sell",
                            "price": row["close"],
                            "size": position_size,
                            "pnl": pnl,
                        }
                    )
                    position = 0
                    position_size = 0

            # Calculate current portfolio value
            if position > 0:
                portfolio_value = capital + (position_size * row["close"])
            elif position < 0:
                portfolio_value = capital - (position_size * (row["close"] - position))
            else:
                portfolio_value = capital

            equity_curve.append({"timestamp": timestamp, "equity": portfolio_value})

        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        return {
            "trades": trades_df,
            "equity_curve": pd.DataFrame(equity_curve),
            "final_capital": (
                equity_curve[-1]["equity"] if equity_curve else config.initial_capital
            ),
        }

    def _get_strategy_signal(self, strategy_instance, data: pd.DataFrame) -> int:
        """Get trading signal from strategy."""
        if hasattr(strategy_instance, "generate_signals"):
            try:
                # Transform data to uppercase columns for strategy compatibility
                strategy_data = self._transform_data_for_strategy(data)
                # Use the correct method name (plural)
                signals = strategy_instance.generate_signals(strategy_data)
                if len(signals) > 0:
                    return signals.iloc[-1]  # Return last signal
                return 0
            except Exception as e:
                # Log the actual error for debugging
                self.logger.debug(
                    "Strategy %s failed: %s", strategy_instance.__class__.__name__, e
                )
                # Strategy failed - return 0 (no signal) to generate zero metrics
                return 0

        # No generate_signals method - strategy is invalid, return 0
        return 0

    def _transform_data_for_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data columns to uppercase format expected by external strategies."""
        if data is None or data.empty:
            return data

        # Only select OHLCV columns that strategies expect
        required_columns = ["open", "high", "low", "close", "volume"]

        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Select only OHLCV columns
        df = data[required_columns].copy()

        # Transform lowercase columns to uppercase for strategy compatibility
        column_mapping = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }

        # Rename columns
        df = df.rename(columns=column_mapping)

        return df

    def _align_portfolio_data(self, data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Align multiple asset data to common date range."""
        if not data_dict:
            return pd.DataFrame()

        # Find common date range
        all_dates = None
        for symbol, data in data_dict.items():
            all_dates = (
                set(data.index)
                if all_dates is None
                else all_dates.intersection(set(data.index))
            )

        if not all_dates:
            return pd.DataFrame()

        # Create aligned dataframe
        common_dates = sorted(list(all_dates))
        aligned_data = pd.DataFrame(index=common_dates)

        for symbol, data in data_dict.items():
            aligned_data[f"{symbol}_close"] = data.loc[common_dates, "close"]

        return aligned_data.dropna()

    def _calculate_portfolio_returns(
        self, aligned_data: pd.DataFrame, weights: dict[str, float]
    ) -> pd.Series:
        """Calculate portfolio returns."""
        returns = pd.Series(index=aligned_data.index, dtype=float)

        for i in range(1, len(aligned_data)):
            portfolio_return = 0
            for symbol, weight in weights.items():
                col_name = f"{symbol}_close"
                if col_name in aligned_data.columns:
                    asset_return = (
                        aligned_data[col_name].iloc[i]
                        / aligned_data[col_name].iloc[i - 1]
                    ) - 1
                    portfolio_return += weight * asset_return

            returns.iloc[i] = portfolio_return

        return returns.fillna(0)

    @staticmethod
    # @jit(nopython=True)  # Removed for compatibility
    def _calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Fast RSI calculation using Numba."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gains = np.full_like(prices, np.nan)
        avg_losses = np.full_like(prices, np.nan)
        rsi = np.full_like(prices, np.nan)

        if len(gains) >= period:
            avg_gains[period] = np.mean(gains[:period])
            avg_losses[period] = np.mean(losses[:period])

            for i in range(period + 1, len(prices)):
                avg_gains[i] = (avg_gains[i - 1] * (period - 1) + gains[i - 1]) / period
                avg_losses[i] = (
                    avg_losses[i - 1] * (period - 1) + losses[i - 1]
                ) / period

                if avg_losses[i] == 0:
                    rsi[i] = 100
                else:
                    rs = avg_gains[i] / avg_losses[i]
                    rsi[i] = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    # @jit(nopython=True)  # Removed for compatibility
    def _calculate_macd(
        prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fast MACD calculation using Numba."""
        ema_fast = np.full_like(prices, np.nan)
        ema_slow = np.full_like(prices, np.nan)

        # Calculate EMAs
        alpha_fast = 2.0 / (fast + 1.0)
        alpha_slow = 2.0 / (slow + 1.0)

        ema_fast[0] = prices[0]
        ema_slow[0] = prices[0]

        for i in range(1, len(prices)):
            ema_fast[i] = alpha_fast * prices[i] + (1 - alpha_fast) * ema_fast[i - 1]
            ema_slow[i] = alpha_slow * prices[i] + (1 - alpha_slow) * ema_slow[i - 1]

        macd_line = ema_fast - ema_slow

        # Calculate signal line (EMA of MACD)
        signal_line = np.full_like(prices, np.nan)
        alpha_signal = 2.0 / (signal + 1.0)

        # Start signal line calculation after we have enough MACD data
        signal_start = max(fast, slow)
        if len(macd_line) > signal_start:
            signal_line[signal_start] = macd_line[signal_start]
            for i in range(signal_start + 1, len(prices)):
                signal_line[i] = (
                    alpha_signal * macd_line[i]
                    + (1 - alpha_signal) * signal_line[i - 1]
                )

        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def _calculate_batch_size(self, num_symbols: int, memory_limit_gb: float) -> int:
        """Calculate optimal batch size based on memory constraints."""
        estimated_memory_per_symbol_mb = 50
        available_memory_mb = memory_limit_gb * 1024 * 0.8

        max_batch_size = int(available_memory_mb / estimated_memory_per_symbol_mb)
        return min(max_batch_size, num_symbols, 100)

    def _get_strategy_class(self, strategy_name: str) -> type | None:
        """Get strategy class by name using StrategyFactory."""
        try:
            from .strategy import StrategyFactory

            # Create an instance and get its class
            strategy_instance = StrategyFactory.create_strategy(strategy_name, {})
            return strategy_instance.__class__
        except Exception as e:
            self.logger.error("Failed to load strategy %s: %s", strategy_name, e)
            return None

    def _get_default_parameters(self, strategy_name: str) -> dict[str, Any]:
        """Get default parameters for a strategy."""
        default_params = {
            "rsi": {"period": 14, "overbought": 70, "oversold": 30},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "bollinger_bands": {"period": 20, "deviation": 2},
            "sma_crossover": {"fast_period": 10, "slow_period": 20},
        }
        return default_params.get(strategy_name.lower(), {})

    def _dict_to_result(
        self,
        cached_dict: dict,
        symbol: str,
        strategy: str,
        parameters: dict,
        config: BacktestConfig,
    ) -> BacktestResult:
        """Convert cached dictionary to BacktestResult object."""
        import pandas as pd

        # Handle trades data from cache
        trades = cached_dict.get("trades")
        if trades is not None and isinstance(trades, dict):
            # Convert trades dict back to DataFrame
            trades = pd.DataFrame(trades)
        elif trades is not None and not isinstance(trades, pd.DataFrame):
            trades = None

        # Handle equity_curve data from cache
        equity_curve = cached_dict.get("equity_curve")
        if equity_curve is not None and isinstance(equity_curve, dict):
            # Convert equity_curve dict back to DataFrame
            equity_curve = pd.DataFrame(equity_curve)
        elif equity_curve is not None and not isinstance(equity_curve, pd.DataFrame):
            equity_curve = None

        return BacktestResult(
            symbol=symbol,
            strategy=strategy,
            parameters=parameters,
            config=config,
            metrics=cached_dict.get("metrics", {}),
            trades=trades,
            equity_curve=equity_curve,
            start_date=cached_dict.get("start_date"),
            end_date=cached_dict.get("end_date"),
            duration_seconds=cached_dict.get("duration_seconds", 0),
            data_points=cached_dict.get("data_points", 0),
            error=cached_dict.get("error"),
        )

    def _log_stats(self):
        """Log performance statistics."""
        self.logger.info("Batch backtest completed:")
        self.logger.info("  Total backtests: %s", self.stats["backtests_run"])
        self.logger.info("  Cache hits: %s", self.stats["cache_hits"])
        self.logger.info("  Cache misses: %s", self.stats["cache_misses"])
        self.logger.info("  Errors: %s", self.stats["errors"])
        self.logger.info("  Total time: %.2fs", self.stats["total_time"])
        if self.stats["backtests_run"] > 0:
            avg_time = self.stats["total_time"] / self.stats["backtests_run"]
            self.logger.info("  Avg time per backtest: %.2fs", avg_time)

    def get_performance_stats(self) -> dict[str, Any]:
        """Get engine performance statistics."""
        return self.stats.copy()

    def run(self, manifest: dict[str, Any], outdir: Path | str) -> dict[str, Any]:
        """
        Manifest-driven executor.

        This method expands the provided manifest (as produced by the CLI) into
        BacktestConfig objects and runs batch backtests for each requested
        (interval x strategies x symbols) combination. Results are persisted
        to the DB via src.database.unified_models (best-effort).

        Returns a summary dict with counts and plan_hash.
        """
        import json as _json
        from pathlib import Path as _Path

        outdir = _Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        plan = manifest.get("plan", {})
        symbols = plan.get("symbols", []) or []
        strategies = plan.get("strategies", []) or []
        intervals = plan.get("intervals", []) or ["1d"]
        start = plan.get("start")
        end = plan.get("end")
        period_mode = plan.get("period_mode", "max")
        plan_hash = plan.get("plan_hash")
        target_metric = plan.get("metric", DEFAULT_METRIC)

        # Resolve period_mode -> if 'max' leave start/end None so data manager uses full range
        if period_mode == "max":
            start_date = None
            end_date = None
        else:
            start_date = start
            end_date = end

        # Create run row in DB (best-effort)
        run_obj = None
        run_id = None
        try:
            from src.database import unified_models  # type: ignore[import-not-found]

            try:
                # Prefer robust ensure_run_for_manifest which will attempt fallback creation
                if hasattr(unified_models, "ensure_run_for_manifest"):
                    run_obj = unified_models.ensure_run_for_manifest(manifest)
                else:
                    run_obj = unified_models.create_run_from_manifest(manifest)
                run_id = getattr(run_obj, "run_id", None)
            except Exception:
                run_obj = None
                run_id = None
        except Exception:
            run_obj = None
            run_id = None

        # Prepare a persistence_context passed to lower-level helpers
        # If run_id couldn't be created/resolved, disable persistence to avoid null run_id inserts.
        if run_id is None:
            persistence_context = None
        else:
            persistence_context = {
                "run_id": run_id,
                "target_metric": target_metric,
                "plan_hash": plan_hash,
            }

        total_results = 0
        errors = 0
        persisted = 0
        results_summary = []

        # For each interval, create a BacktestConfig and run batch backtests
        for interval in intervals:
            try:
                # Respect export flags from manifest so workers capture trades/equity when requested.
                exports = plan.get("exports", []) or []
                if isinstance(exports, str):
                    exports = [exports]

                # Capture trades by default when DB persistence is active so we can store
                # detailed executions into unified_models (trades table and trades_raw).
                # Still honor explicit exports flags when provided.
                save_trades = (
                    "all" in exports
                    or "trades" in exports
                    or "trade" in exports
                    or (persistence_context is not None)
                )
                save_equity = (
                    "all" in exports or "equity" in exports or "equity_curve" in exports
                )

                cfg_kwargs = {
                    "symbols": symbols,
                    "strategies": strategies,
                    "start_date": start_date,
                    "end_date": end_date,
                    "period": period_mode,
                    "initial_capital": plan.get("initial_capital", 10000),
                    "interval": interval,
                    "max_workers": plan.get("max_workers", None),
                    # propagate strategies_path from manifest so workers can initialize loaders
                    "strategies_path": plan.get("strategies_path"),
                    "save_trades": save_trades,
                    "save_equity_curve": save_equity,
                }
                # Build BacktestConfig
                try:
                    cfg = BacktestConfig(**cfg_kwargs)
                except Exception:
                    # Fallback: construct minimal config object-like dict
                    class _TmpCfg:
                        def __init__(self, **kw):
                            self.__dict__.update(kw)

                    cfg = _TmpCfg(**cfg_kwargs)

                # Ensure fallback config has expected attributes with sensible defaults
                # so later code can access them regardless of how cfg was constructed.
                _defaults = {
                    "initial_capital": 10000,
                    "interval": getattr(cfg, "interval", "1d"),
                    "max_workers": getattr(cfg, "max_workers", None),
                    "use_cache": getattr(cfg, "use_cache", True),
                    "commission": getattr(cfg, "commission", 0.001),
                    "save_trades": getattr(cfg, "save_trades", False),
                    "save_equity_curve": getattr(cfg, "save_equity_curve", False),
                    "override_old_trades": getattr(cfg, "override_old_trades", True),
                    "memory_limit_gb": getattr(cfg, "memory_limit_gb", 8.0),
                    "asset_type": getattr(cfg, "asset_type", None),
                    "futures_mode": getattr(cfg, "futures_mode", False),
                    "leverage": getattr(cfg, "leverage", 1.0),
                }
                for _k, _v in _defaults.items():
                    if not hasattr(cfg, _k):
                        try:
                            setattr(cfg, _k, _v)
                        except Exception:
                            # be defensive if cfg disallows setattr
                            pass

                # Run batch
                batch_results = self.run_batch_backtests(cfg)
                total_results += len(batch_results)

                # Persist individual results (best-effort) using direct_backtest helper
                try:
                    import src.core.direct_backtest as direct_mod  # type: ignore[import-not-found]

                    # Only attempt persistence when we have a valid persistence_context (run_id resolved)
                    if persistence_context:
                        for r in batch_results:
                            # Map BacktestResult dataclass to expected dict for persistence
                            rd = {
                                "symbol": r.symbol,
                                "strategy": r.strategy,
                                "timeframe": getattr(r.config, "interval", interval),
                                "metrics": r.metrics or {},
                                "trades": r.trades if hasattr(r, "trades") else None,
                                "bt_results": getattr(r, "bt_results_raw", None),
                                "start_date": getattr(r, "start_date", start_date),
                                "end_date": getattr(r, "end_date", end_date),
                                "error": getattr(r, "error", None),
                            }

                            # Force a persistence stub if worker returned no metrics/trades/bt_results
                            # but did not set an explicit error. This ensures full lineage for the run.
                            if (
                                not rd.get("metrics")
                                and not rd.get("trades")
                                and not rd.get("bt_results")
                                and not rd.get("error")
                            ):
                                rd["error"] = "no_result"

                            try:
                                direct_mod._persist_result_to_db(
                                    rd, persistence_context
                                )
                                persisted += 1
                            except Exception:
                                errors += 1
                    else:
                        # Persistence disabled (no run_id); skip storing individual results
                        pass
                except Exception:
                    # If persistence helper unavailable, skip persistence but continue
                    pass

                # Summarize top strategies for this interval
                for r in batch_results[:5]:
                    results_summary.append(
                        {
                            "symbol": r.symbol,
                            "strategy": r.strategy,
                            "interval": getattr(r.config, "interval", interval),
                            "metric": (r.metrics or {}).get(target_metric),
                            "error": getattr(r, "error", None),
                        }
                    )

            except Exception as e:
                errors += 1
                logging.getLogger(__name__).exception(
                    "Failed running interval %s: %s", interval, e
                )
                continue

        summary = {
            "plan_hash": plan_hash,
            "total_results": total_results,
            "persisted": persisted,
            "errors": errors,
            "results_sample": results_summary,
        }

        # Best-effort: finalize ranks/aggregates and upsert BestStrategy rows into unified_models
        try:
            if run_id is not None and target_metric:
                try:
                    from src.database import (
                        unified_models,  # type: ignore[import-not-found]
                    )

                    sess = unified_models.Session()
                    try:
                        # Get distinct symbols for run
                        symbols = (
                            sess.query(unified_models.BacktestResult.symbol)
                            .filter(unified_models.BacktestResult.run_id == run_id)
                            .distinct()
                            .all()
                        )
                        symbols = [s[0] for s in symbols]

                        def _is_higher_better(metric_name: str) -> bool:
                            mn = (metric_name or "").lower()
                            if "drawdown" in mn or "max_drawdown" in mn or "mdd" in mn:
                                return False
                            return True

                        for symbol in symbols:
                            rows = (
                                sess.query(unified_models.BacktestResult)
                                .filter(
                                    unified_models.BacktestResult.run_id == run_id,
                                    unified_models.BacktestResult.symbol == symbol,
                                )
                                .all()
                            )

                            entries = []
                            higher_better = _is_higher_better(target_metric)
                            for r in rows:
                                mval = None
                                try:
                                    if r.metrics and isinstance(r.metrics, dict):
                                        raw = r.metrics.get(target_metric)
                                        mval = None if raw is None else float(raw)
                                except Exception as exc:
                                    logging.getLogger(__name__).debug(
                                        "Failed to parse metric %s: %s",
                                        target_metric,
                                        exc,
                                    )
                                # Treat None as worst
                                sort_key = (
                                    float("-inf")
                                    if higher_better
                                    else float("inf")
                                    if mval is None
                                    else mval
                                )
                                if mval is None:
                                    sort_key = (
                                        float("-inf") if higher_better else float("inf")
                                    )
                                entries.append((sort_key, mval is None, r))

                            # Sort and assign ranks
                            entries.sort(key=lambda x: x[0], reverse=higher_better)
                            for idx, (_sort_key, _is_null, row) in enumerate(entries):
                                try:
                                    row.rank_in_symbol = idx + 1
                                    sess.add(row)
                                except Exception:
                                    pass

                            # Persist SymbolAggregate and BestStrategy for top entry
                            if entries:
                                best_row = entries[0][2]
                                topn = []
                                for e in entries[:3]:
                                    r = e[2]
                                    topn.append(
                                        {
                                            "strategy": r.strategy,
                                            "interval": r.interval,
                                            "rank": r.rank_in_symbol,
                                            "metric": None
                                            if r.metrics is None
                                            else r.metrics.get(target_metric),
                                        }
                                    )
                                existing_agg = (
                                    sess.query(unified_models.SymbolAggregate)
                                    .filter(
                                        unified_models.SymbolAggregate.run_id == run_id,
                                        unified_models.SymbolAggregate.symbol == symbol,
                                        unified_models.SymbolAggregate.best_by
                                        == target_metric,
                                    )
                                    .one_or_none()
                                )
                                summary_json = {"top": topn}
                                if existing_agg:
                                    existing_agg.best_result = best_row.result_id
                                    existing_agg.summary = summary_json
                                    sess.add(existing_agg)
                                else:
                                    agg = unified_models.SymbolAggregate(
                                        run_id=run_id,
                                        symbol=symbol,
                                        best_by=target_metric,
                                        best_result=best_row.result_id,
                                        summary=summary_json,
                                    )
                                    sess.add(agg)

                                # Upsert BestStrategy
                                try:
                                    bs_existing = (
                                        sess.query(unified_models.BestStrategy)
                                        .filter(
                                            unified_models.BestStrategy.symbol
                                            == symbol,
                                            unified_models.BestStrategy.timeframe
                                            == best_row.interval,
                                        )
                                        .one_or_none()
                                    )

                                    def _num(mdict, key):
                                        try:
                                            if mdict and isinstance(mdict, dict):
                                                v = mdict.get(key)
                                                return (
                                                    float(v) if v is not None else None
                                                )
                                        except Exception:
                                            return None
                                        return None

                                    sortino_val = _num(
                                        best_row.metrics, "sortino_ratio"
                                    ) or _num(best_row.metrics, "Sortino_Ratio")
                                    calmar_val = _num(
                                        best_row.metrics, "calmar_ratio"
                                    ) or _num(best_row.metrics, "Calmar_Ratio")
                                    sharpe_val = _num(
                                        best_row.metrics, "sharpe_ratio"
                                    ) or _num(best_row.metrics, "Sharpe_Ratio")
                                    total_return_val = _num(
                                        best_row.metrics, "total_return"
                                    ) or _num(best_row.metrics, "Total_Return")
                                    max_dd_val = _num(
                                        best_row.metrics, "max_drawdown"
                                    ) or _num(best_row.metrics, "Max_Drawdown")

                                    if bs_existing:
                                        bs_existing.strategy = best_row.strategy
                                        bs_existing.sortino_ratio = sortino_val
                                        bs_existing.calmar_ratio = calmar_val
                                        bs_existing.sharpe_ratio = sharpe_val
                                        bs_existing.total_return = total_return_val
                                        bs_existing.max_drawdown = max_dd_val
                                        bs_existing.backtest_result_id = getattr(
                                            best_row, "result_id", None
                                        )
                                        bs_existing.updated_at = datetime.utcnow()
                                        sess.add(bs_existing)
                                    else:
                                        bs = unified_models.BestStrategy(
                                            symbol=symbol,
                                            timeframe=best_row.interval,
                                            strategy=best_row.strategy,
                                            sortino_ratio=sortino_val,
                                            calmar_ratio=calmar_val,
                                            sharpe_ratio=sharpe_val,
                                            total_return=total_return_val,
                                            max_drawdown=max_dd_val,
                                            backtest_result_id=getattr(
                                                best_row, "result_id", None
                                            ),
                                            updated_at=datetime.utcnow(),
                                        )
                                        sess.add(bs)
                                except Exception:
                                    logging.getLogger(__name__).exception(
                                        "Failed to upsert BestStrategy for %s", symbol
                                    )

                        sess.commit()
                    finally:
                        try:
                            sess.close()
                        except Exception:
                            pass
                except Exception:
                    logging.getLogger(__name__).exception(
                        "Failed to finalize BestStrategy for run %s", run_id
                    )
        except Exception:
            # Non-fatal: continue even if finalization fails
            pass

        # Write summary file
        try:
            summary_path = outdir / "engine_run_summary.json"
            with summary_path.open("w", encoding="utf-8") as fh:
                _json.dump(summary, fh, indent=2, sort_keys=True, ensure_ascii=False)
        except Exception:
            pass

        return summary

    def clear_cache(self, symbol: str | None = None, strategy: str | None = None):
        """Clear cached results."""
        self.cache_manager.clear_cache(cache_type="backtest", symbol=symbol)

    def _prepare_data_for_backtesting_lib(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for the backtesting library (requires uppercase OHLCV columns)."""
        try:
            # Check if we have lowercase columns and convert them
            if all(
                col in data.columns
                for col in ["open", "high", "low", "close", "volume"]
            ):
                bt_data = data.rename(
                    columns={
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                    }
                )[["Open", "High", "Low", "Close", "Volume"]].copy()
            # Check if we already have uppercase columns
            elif all(
                col in data.columns
                for col in ["Open", "High", "Low", "Close", "Volume"]
            ):
                bt_data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
            else:
                self.logger.error("Missing required OHLCV columns in data")
                return None

            # Ensure no NaN values
            bt_data = bt_data.dropna()

            return bt_data

        except Exception as e:
            self.logger.error("Error preparing data for backtesting library: %s", e)
            return None

    def _extract_metrics_from_bt_results(self, bt_results) -> dict[str, Any]:
        """Extract metrics from backtesting library results.

        This function is defensive: backtesting library results may contain pandas
        Timestamps/Timedeltas or other non-scalar types. Coerce values to floats
        where sensible and fall back to None/0 when conversion fails.
        """
        import math

        try:

            def _as_float(v):
                """Safely coerce a value to float or return None."""
                if v is None:
                    return None
                # Already a float/int
                if isinstance(v, (int, float)):
                    if isinstance(v, bool):
                        return float(v)
                    if math.isfinite(v):
                        return float(v)
                    return None
                # Numpy numeric types
                try:
                    import numpy as _np

                    if isinstance(v, _np.generic):
                        return float(v.item())
                except Exception:
                    pass
                # Pandas Timestamp/Timedelta -> convert to numeric where appropriate
                try:
                    import pandas as _pd

                    if isinstance(v, _pd.Timedelta):
                        # convert to total days as a numeric proxy (timedeltas appear for volatility sometimes)
                        try:
                            return float(v.total_seconds())
                        except Exception:
                            return None
                    if isinstance(v, _pd.Timestamp):
                        # Timestamp is not numeric; return None
                        return None
                except Exception:
                    pass
                # Strings that may include percent signs or commas
                if isinstance(v, str):
                    try:
                        s = v.strip().replace("%", "").replace(",", "")
                        return float(s)
                    except Exception:
                        return None
                # Fallback: try numeric conversion
                try:
                    return float(v)
                except Exception:
                    return None

            def _get_first(keys, default=None):
                for k in keys:
                    try:
                        if isinstance(bt_results, dict) and k in bt_results:
                            return bt_results.get(k)
                    except Exception:
                        pass
                return default

            # Map keys with fallbacks
            total_return = (
                _as_float(
                    _get_first(["Return [%]", "Total_Return", "total_return"], 0.0)
                )
                or 0.0
            )
            sharpe = (
                _as_float(
                    _get_first(["Sharpe Ratio", "Sharpe_Ratio", "sharpe_ratio"], 0.0)
                )
                or 0.0
            )
            sortino = (
                _as_float(
                    _get_first(["Sortino Ratio", "Sortino_Ratio", "sortino_ratio"], 0.0)
                )
                or 0.0
            )
            calmar = (
                _as_float(
                    _get_first(["Calmar Ratio", "Calmar_Ratio", "calmar_ratio"], 0.0)
                )
                or 0.0
            )
            max_dd = _as_float(
                _get_first(["Max. Drawdown [%]", "Max_Drawdown", "max_drawdown"], 0.0)
            )
            max_dd = 0.0 if max_dd is None else abs(max_dd)
            volatility = (
                _as_float(_get_first(["Volatility [%]", "volatility"], 0.0)) or 0.0
            )
            num_trades = _get_first(["# Trades", "num_trades", "Trades"], 0) or 0
            try:
                num_trades = int(num_trades)
            except Exception:
                num_trades = 0
            win_rate = _as_float(_get_first(["Win Rate [%]", "win_rate"], 0.0)) or 0.0
            profit_factor = (
                _as_float(_get_first(["Profit Factor", "profit_factor"], 1.0)) or 1.0
            )
            best_trade = (
                _as_float(_get_first(["Best Trade [%]", "best_trade"], 0.0)) or 0.0
            )
            worst_trade = (
                _as_float(_get_first(["Worst Trade [%]", "worst_trade"], 0.0)) or 0.0
            )
            avg_trade = (
                _as_float(_get_first(["Avg. Trade [%]", "avg_trade"], 0.0)) or 0.0
            )
            avg_trade_duration = (
                _as_float(
                    _get_first(["Avg. Trade Duration", "avg_trade_duration"], 0.0)
                )
                or 0.0
            )
            start_value = _as_float(_get_first(["Start", "start_value"], 0.0)) or 0.0
            end_value = (
                _as_float(
                    _get_first(["End", "end_value", "Equity Final [$]"], start_value)
                )
                or start_value
            )
            buy_hold = (
                _as_float(_get_first(["Buy & Hold Return [%]", "buy_hold_return"], 0.0))
                or 0.0
            )
            exposure = (
                _as_float(_get_first(["Exposure Time [%]", "exposure_time"], 0.0))
                or 0.0
            )

            metrics = {
                "total_return": total_return,
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "calmar_ratio": calmar,
                "max_drawdown": max_dd,
                "volatility": volatility,
                "num_trades": num_trades,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "best_trade": best_trade,
                "worst_trade": worst_trade,
                "avg_trade": avg_trade,
                "avg_trade_duration": avg_trade_duration,
                "start_value": start_value,
                "end_value": end_value,
                "buy_hold_return": buy_hold,
                "exposure_time": exposure,
            }

            return metrics
        except Exception as e:
            self.logger.error(
                "Error extracting metrics from backtesting results: %s", e
            )
            return {}

    def _convert_backtesting_results(
        self, bt_results, bt_data: pd.DataFrame, config: BacktestConfig
    ) -> dict[str, Any]:
        """Convert backtesting library results to our internal format."""
        try:
            # Get trades from backtesting library
            trades_df = None
            equity_curve_df = None

            # Try to get trades if available
            try:
                if hasattr(bt_results, "_trades") and bt_results._trades is not None:
                    trades_df = bt_results._trades.copy()

                # Get equity curve from backtesting library
                if (
                    hasattr(bt_results, "_equity_curve")
                    and bt_results._equity_curve is not None
                ):
                    equity_curve_df = bt_results._equity_curve.copy()

            except Exception as e:
                self.logger.debug("Could not extract detailed trade data: %s", e)

            result = {
                "trades": trades_df,
                "equity_curve": equity_curve_df,
                "final_value": float(bt_results.get("End", config.initial_capital)),
                "total_trades": int(bt_results.get("# Trades", 0)),
            }

            return result

        except Exception as e:
            self.logger.error("Error converting backtesting results: %s", e)
            return {
                "trades": None,
                "equity_curve": None,
                "final_value": config.initial_capital,
                "total_trades": 0,
            }
