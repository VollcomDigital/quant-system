from __future__ import annotations

import importlib
import inspect
import itertools
import inspect
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import CollectionConfig, Config
from ..data.alpaca_source import AlpacaSource
from ..data.alphavantage_source import AlphaVantageSource
from ..data.base import DataSource
from ..data.ccxt_source import CCXTSource
from ..data.finnhub_source import FinnhubSource
from ..data.polygon_source import PolygonSource
from ..data.tiingo_source import TiingoSource
from ..data.twelvedata_source import TwelveDataSource
from ..data.yfinance_source import YFinanceSource
from ..strategies.base import BaseStrategy
from ..strategies.registry import discover_external_strategies
from ..utils.telemetry import get_logger, time_block
from .metrics import (
    omega_ratio,
    pain_index,
    sharpe_ratio,
    sortino_ratio,
    tail_ratio,
    total_return,
)
from .results_cache import ResultsCache


@dataclass
class BestResult:
    collection: str
    symbol: str
    timeframe: str
    strategy: str
    params: dict[str, Any]
    metric_name: str
    metric_value: float
    stats: dict[str, Any]


class BacktestRunner:
    def __init__(self, cfg: Config, strategies_root: Path, run_id: str | None = None):
        self.cfg = cfg
        self.strategies_root = strategies_root
        self.external_index = discover_external_strategies(strategies_root)
        self.results_cache = ResultsCache(Path(self.cfg.cache_dir).parent / "results")
        self.run_id = run_id
        self.logger = get_logger()
        self._pybroker_components: tuple[Any, ...] | None = None

    def _ensure_pybroker(self) -> tuple[Any, ...]:
        if self._pybroker_components is None:
            try:
                pybroker = importlib.import_module("pybroker")
                common = importlib.import_module("pybroker.common")
                strategy_cls = pybroker.Strategy
                config_cls = pybroker.StrategyConfig
                fee_mode_cls = common.FeeMode
                price_type_cls = common.PriceType
                data_col_enum = common.DataCol
                try:
                    pybroker.disable_logging()
                except Exception:
                    pass
                try:
                    pybroker.disable_progress_bar()
                except Exception:
                    pass
                self._pybroker_components = (
                    strategy_cls,
                    config_cls,
                    fee_mode_cls,
                    price_type_cls,
                    data_col_enum,
                )
            except Exception as exc:  # pragma: no cover - sanity guard
                raise RuntimeError("PyBroker must be installed to run backtests.") from exc
        return self._pybroker_components

    def _make_source(self, col: CollectionConfig) -> DataSource:
        cache_dir = Path(self.cfg.cache_dir)
        src = col.source.lower()
        if src == "yfinance":
            return YFinanceSource(cache_dir)
        if src in ("ccxt", "binance", "bybit"):
            if not col.exchange:
                exchange = src if src != "ccxt" else None
                if not exchange:
                    raise ValueError("exchange is required for ccxt collection")
                return CCXTSource(exchange, cache_dir)
            return CCXTSource(col.exchange, cache_dir)
        if src == "polygon":
            return PolygonSource(cache_dir)
        if src == "tiingo":
            return TiingoSource(cache_dir)
        if src == "alpaca":
            return AlpacaSource(cache_dir)
        if src == "finnhub":
            return FinnhubSource(cache_dir)
        if src == "twelvedata":
            return TwelveDataSource(cache_dir)
        if src == "alphavantage":
            return AlphaVantageSource(cache_dir)
        raise ValueError(f"Unsupported data source: {col.source}")

    def _fees_slippage_for(self, col: CollectionConfig) -> tuple[float, float]:
        if col.fees is not None or col.slippage is not None:
            return (
                col.fees if col.fees is not None else self.cfg.fees,
                col.slippage if col.slippage is not None else self.cfg.slippage,
            )
        src = col.source.lower()
        if src in ("binance", "bybit", "ccxt"):
            return (0.0006, 0.0005)
        return (0.0005, 0.0005)

    @staticmethod
    def _bars_per_year(timeframe: str) -> int:
        tf = timeframe.strip().lower()
        if not tf:
            return 252
        digits = "".join(ch for ch in tf if ch.isdigit())
        unit = tf[len(digits) :].strip() or "d"
        value = int(digits) if digits else 1
        value = max(1, value)
        if unit in {"d", "day", "days"}:
            return max(1, int(round(252 / value)))
        if unit in {"w", "week", "weeks"}:
            return max(1, int(round(52 / value)))
        if unit in {"h", "hour", "hours"}:
            return max(1, int(round((24 * 365) / value)))
        if unit in {"m", "min", "minute", "minutes"}:
            return max(1, int(round((60 * 24 * 365) / value)))
        if unit in {"s", "sec", "second", "seconds"}:
            return max(1, int(round((60 * 60 * 24 * 365) / value)))
        return 252

    @staticmethod
    def _sample_series(series: pd.Series, max_points: int = 500) -> list[dict[str, float]]:
        if series.empty:
            return []
        cleaned = series.dropna()
        if cleaned.empty:
            return []
        step = max(1, len(cleaned) // max_points)
        sampled: list[dict[str, float]] = []
        for idx in range(0, len(cleaned), step):
            ts = cleaned.index[idx]
            ts_val = ts.isoformat() if isinstance(ts, datetime) else str(ts)
            sampled.append({"ts": ts_val, "value": float(cleaned.iloc[idx])})
        last_ts = cleaned.index[-1]
        last_val = cleaned.iloc[-1]
        last_key = last_ts.isoformat() if isinstance(last_ts, datetime) else str(last_ts)
        if not sampled or sampled[-1]["ts"] != last_key:
            sampled.append({"ts": last_key, "value": float(last_val)})
        return sampled

    @staticmethod
    def _convert_decimal(value: Any) -> Any:
        if isinstance(value, Decimal):
            return float(value)
        return value

    @staticmethod
    def _fractional_enabled(col: CollectionConfig, symbol: str) -> bool:
        src = col.source.lower()
        if src in {"binance", "bybit", "ccxt"}:
            return True
        return "/" in symbol or symbol.endswith("USDT")

    @staticmethod
    def _prepare_pybroker_frame(
        df: pd.DataFrame, symbol: str, data_col_enum: Any
    ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
        if df.empty:
            raise ValueError("No price data available for backtest.")
        working = df.copy()
        if not working.index.is_monotonic_increasing:
            working = working.sort_index()
        rename_map = {}
        for column in working.columns:
            lowered = column.lower()
            if lowered in {"open", "high", "low", "close", "volume"}:
                rename_map[column] = lowered
        working = working.rename(columns=rename_map)
        required = {"open", "high", "low", "close"}
        missing = sorted(required - set(map(str.lower, working.columns)))
        if missing:
            raise ValueError(f"Missing price columns: {', '.join(missing)}")
        if "volume" not in map(str.lower, working.columns):
            working["volume"] = 0.0
        dates = pd.to_datetime(working.index)
        working[data_col_enum.DATE.value] = dates
        working[data_col_enum.SYMBOL.value] = symbol
        ordered_cols = [
            data_col_enum.SYMBOL.value,
            data_col_enum.DATE.value,
            data_col_enum.OPEN.value,
            data_col_enum.HIGH.value,
            data_col_enum.LOW.value,
            data_col_enum.CLOSE.value,
            data_col_enum.VOLUME.value,
        ]
        return working[ordered_cols].reset_index(drop=True), dates

    def _compute_cagr(
        self,
        equity_curve: pd.Series,
        dates: pd.DatetimeIndex,
        periods_per_year: int,
    ) -> float:
        if equity_curve.empty or len(equity_curve) < 2:
            return float("nan")
        start = equity_curve.iloc[0]
        end = equity_curve.iloc[-1]
        if start <= 0 or end <= 0:
            return float("nan")
        delta_days = (dates[-1] - dates[0]).days
        years = delta_days / 365.25 if delta_days > 0 else 0.0
        if years <= 0 and periods_per_year > 0:
            years = len(equity_curve) / periods_per_year
        if years <= 0:
            return float("nan")
        try:
            return float(end ** (1 / years) - 1.0)
        except Exception:  # pragma: no cover - guard against numerical issues
            return float("nan")

    def _grid(self, grid: dict[str, Iterable[Any]]):
        if not grid:
            yield {}
            return
        keys = list(grid.keys())
        for values in itertools.product(*(grid[k] for k in keys)):
            yield dict(zip(keys, values, strict=False))

    def _evaluate_metric(
        self,
        metric: str,
        returns: pd.Series,
        equity: pd.Series,
        periods_per_year: int,
    ) -> float:
        metric = metric.lower()
        if metric == "sharpe":
            return sharpe_ratio(
                returns,
                risk_free_rate=self.cfg.risk_free_rate,
                periods_per_year=periods_per_year,
            )
        if metric == "sortino":
            return sortino_ratio(
                returns,
                risk_free_rate=self.cfg.risk_free_rate,
                periods_per_year=periods_per_year,
            )
        if metric == "profit":
            return total_return(equity)
        raise ValueError(f"Unknown metric: {metric}")

    def _run_pybroker_simulation(
        self,
        data: pd.DataFrame,
        dates: pd.DatetimeIndex,
        symbol: str,
        entries: pd.Series,
        exits: pd.Series,
        fee_percent: float,
        slippage_percent: float,
        timeframe: str,
        fractional: bool,
        periods_per_year: int,
    ) -> tuple[pd.Series, pd.Series, dict[str, Any]] | None:
        strategy_cls, config_cls, fee_mode_cls, price_type_cls, data_col_enum = (
            self._ensure_pybroker()
        )
        fee_total = max(0.0, fee_percent + slippage_percent)
        config_kwargs: dict[str, Any] = {
            "initial_cash": 10_000.0,
            "subtract_fees": True,
            "enable_fractional_shares": fractional,
            "exit_on_last_bar": True,
            "exit_cover_fill_price": price_type_cls.CLOSE,
            "exit_sell_fill_price": price_type_cls.CLOSE,
            "bars_per_year": periods_per_year,
        }
        if fee_total > 0:
            config_kwargs["fee_mode"] = fee_mode_cls.ORDER_PERCENT
            config_kwargs["fee_amount"] = fee_total

            # check if all set params are __init__ params and delete false params
            allowed = set(inspect.signature(config_cls.__init__).parameters)
            allowed.discard("self")
            config_kwargs = {k: v for k, v in config_kwargs.items() if k in allowed}

        strategy = strategy_cls(
            data,
            start_date=data[data_col_enum.DATE.value].iloc[0],
            end_date=data[data_col_enum.DATE.value].iloc[-1],
            config=config_cls(**config_kwargs),
        )

        aligned_entries = entries.reindex(dates, fill_value=False).astype(bool).to_numpy()
        aligned_exits = exits.reindex(dates, fill_value=False).astype(bool).to_numpy()

        def exec_fn(ctx):
            idx = ctx.bars - 1
            if idx < 0 or idx >= len(aligned_entries):
                return
            if aligned_entries[idx] and ctx.long_pos() is None:
                ctx.buy_fill_price = price_type_cls.CLOSE
                ctx.buy_shares = ctx.calc_target_shares(target_size=1.0)
            if aligned_exits[idx] and ctx.long_pos() is not None:
                ctx.sell_fill_price = price_type_cls.CLOSE
                ctx.sell_all_shares()

        strategy.add_execution(exec_fn, symbol)

        try:
            result = strategy.backtest(calc_bootstrap=False)
        except Exception:
            return None

        portfolio_df = result.portfolio
        if not isinstance(portfolio_df, pd.DataFrame):
            portfolio_df = pd.DataFrame(portfolio_df)
        if portfolio_df.empty:
            return None
        portfolio_df = portfolio_df.copy()
        if "date" not in portfolio_df.columns:
            if isinstance(portfolio_df.index, pd.DatetimeIndex):
                portfolio_df = portfolio_df.reset_index()
                if "index" in portfolio_df.columns and "date" not in portfolio_df.columns:
                    portfolio_df = portfolio_df.rename(columns={"index": "date"})
            else:
                portfolio_df["date"] = pd.to_datetime(dates[: len(portfolio_df)])
        portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])
        equity_series = (
            portfolio_df.set_index("date")["equity"].astype(float).groupby(level=0).last()
        )
        equity_series = equity_series.reindex(pd.to_datetime(dates))
        equity_series = equity_series.ffill().bfill()
        if equity_series.isna().all():
            return None
        equity_series.index = dates
        returns = equity_series.pct_change().fillna(0.0)
        equity_curve = (1.0 + returns).cumprod()
        drawdown_series = equity_curve / equity_curve.cummax() - 1.0

        trades_df = result.trades
        if isinstance(trades_df, pd.DataFrame):
            trades_frame = trades_df.copy()
        elif isinstance(trades_df, list):
            trades_frame = pd.DataFrame(trades_df)
        else:
            trades_frame = pd.DataFrame()

        trades_records: list[dict[str, Any]] = []
        if not trades_frame.empty:
            for column in trades_frame.columns:
                trades_frame[column] = trades_frame[column].map(self._convert_decimal)
                if pd.api.types.is_datetime64_any_dtype(trades_frame[column]):
                    trades_frame[column] = trades_frame[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
            trades_records = trades_frame.head(50).to_dict("records")

        trade_count = int(getattr(result.metrics, "trade_count", len(trades_frame)))
        omega = float(omega_ratio(returns))
        tail = float(tail_ratio(returns))
        sharpe_val = float(
            sharpe_ratio(
                returns,
                risk_free_rate=self.cfg.risk_free_rate,
                periods_per_year=periods_per_year,
            )
        )
        sortino_val = float(
            sortino_ratio(
                returns,
                risk_free_rate=self.cfg.risk_free_rate,
                periods_per_year=periods_per_year,
            )
        )
        profit = float(total_return(equity_curve))
        pain = float(pain_index(equity_curve))
        max_dd = float(drawdown_series.min()) if not drawdown_series.empty else float("nan")
        cagr = self._compute_cagr(equity_curve, dates, periods_per_year)
        calmar = float(cagr / abs(max_dd)) if max_dd < 0 and np.isfinite(cagr) else float("nan")

        stats = {
            "sharpe": sharpe_val,
            "sortino": sortino_val,
            "omega": omega,
            "tail_ratio": tail,
            "profit": profit,
            "pain_index": pain,
            "trades": trade_count,
            "max_drawdown": max_dd,
            "cagr": cagr,
            "calmar": calmar,
            "equity_curve": self._sample_series(equity_curve),
            "drawdown_curve": self._sample_series(drawdown_series),
            "trades_log": trades_records,
        }
        return returns, equity_curve, stats

    def run_all(self, only_cached: bool = False) -> list[BestResult]:
        best_results: list[BestResult] = []
        self.metrics = {
            "result_cache_hits": 0,
            "result_cache_misses": 0,
            "param_evals": 0,
            "symbols_tested": 0,
            "strategies_used": set(),
        }
        self.failures: list[dict[str, Any]] = []

        overrides = {s.name: s.params for s in self.cfg.strategies} if self.cfg.strategies else {}

        jobs: list[tuple[CollectionConfig, str, str, str]] = []
        for col in self.cfg.collections:
            for symbol in col.symbols:
                for timeframe in self.cfg.timeframes:
                    for name in self.external_index.keys():
                        jobs.append((col, symbol, timeframe, name))

        for job in jobs:
            col, symbol, timeframe, strat_name = job
            StrategyClass = self.external_index[strat_name]
            strat: BaseStrategy = StrategyClass()
            base_params = overrides.get(strat_name, {}) if overrides else {}
            grid_override = base_params.get("grid") if isinstance(base_params, dict) else None
            if isinstance(grid_override, dict):
                grid = grid_override
                static_params = {k: v for k, v in base_params.items() if k != "grid"}
            else:
                grid = strat.param_grid() | base_params
                static_params = {}

            source = self._make_source(col)
            with time_block(
                self.logger,
                "data_fetch",
                collection=col.name,
                symbol=symbol,
                timeframe=timeframe,
                source=col.source,
            ):
                try:
                    df = source.fetch(symbol, timeframe, only_cached=only_cached)
                except Exception as exc:
                    self.failures.append(
                        {
                            "collection": col.name,
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "source": col.source,
                            "error": str(exc),
                        }
                    )
                    continue

            if df.empty:
                continue

            _, _, _, _, data_col_enum = self._ensure_pybroker()
            data_frame, dates = self._prepare_pybroker_frame(df, symbol, data_col_enum)
            fractional = self._fractional_enabled(col, symbol)
            periods_per_year = self._bars_per_year(timeframe)
            price = df["Close" if "Close" in df.columns else "close"].astype(float)
            data_fingerprint = f"{len(df)}:{df.index[-1].isoformat()}:{float(price.iloc[-1])}"
            fees_use, slippage_use = self._fees_slippage_for(col)

            search_method = getattr(self.cfg, "param_search", "grid") or "grid"
            trials_target = max(1, int(getattr(self.cfg, "param_trials", 25)))

            fixed_params = dict(static_params)
            search_space: dict[str, list[Any]] = {}
            for name, values in grid.items():
                if isinstance(values, set | tuple | list):
                    options = list(values)
                else:
                    options = [values]
                if len(options) <= 1:
                    if options:
                        fixed_params[name] = options[0]
                else:
                    search_space[name] = options

            best_val = -np.inf
            best_params: dict[str, Any] | None = None
            best_stats: dict[str, Any] | None = None
            self.metrics["symbols_tested"] += 1
            self.metrics["strategies_used"].add(strat.name)

            collection_name = col.name
            strategy_name = strat.name
            timeframe_name = timeframe
            symbol_name = symbol
            fingerprint = data_fingerprint
            fee_value = fees_use
            slippage_value = slippage_use
            frame_df = df
            fetch_data_frame = data_frame
            fetch_dates = dates
            fractional_flag = fractional
            bars_per_year = periods_per_year

            def evaluate(
                var_params: dict[str, Any],
                *,
                fixed=fixed_params,
                strat_obj=strat,
                df_local=frame_df,
                collection_key=collection_name,
                symbol_key=symbol_name,
                timeframe_key=timeframe_name,
                strategy_key=strategy_name,
                fingerprint_key=fingerprint,
                fee_key=fee_value,
                slippage_key=slippage_value,
                data_frame_local=fetch_data_frame,
                dates_local=fetch_dates,
                fractional_local=fractional_flag,
                bars_per_year_local=bars_per_year,
            ) -> float:
                nonlocal best_val, best_params, best_stats
                full_params = {**fixed, **var_params}
                call_params = full_params.copy()
                try:
                    entries, exits = strat_obj.generate_signals(df_local, call_params)
                except Exception as exc:
                    self.failures.append(
                        {
                            "collection": collection_key,
                            "symbol": symbol_key,
                            "timeframe": timeframe_key,
                            "strategy": strategy_key,
                            "params": full_params,
                            "stage": "generate_signals",
                            "error": str(exc),
                        }
                    )
                    return float("nan")
                entries = entries.reindex(df_local.index, fill_value=False)
                exits = exits.reindex(df_local.index, fill_value=False)

                cached = self.results_cache.get(
                    collection=collection_key,
                    symbol=symbol_key,
                    timeframe=timeframe_key,
                    strategy=strategy_key,
                    params=full_params,
                    metric_name=self.cfg.metric,
                    data_fingerprint=fingerprint_key,
                    fees=fee_key,
                    slippage=slippage_key,
                )
                if cached is not None:
                    self.metrics["result_cache_hits"] += 1
                    val_cached = float(cached["metric_value"])
                    try:
                        self.results_cache.set(
                            collection=collection_key,
                            symbol=symbol_key,
                            timeframe=timeframe_key,
                            strategy=strategy_key,
                            params=full_params,
                            metric_name=self.cfg.metric,
                            metric_value=val_cached,
                            stats=cached["stats"],
                            data_fingerprint=fingerprint_key,
                            fees=fee_key,
                            slippage=slippage_key,
                            run_id=self.run_id,
                        )
                    except Exception:
                        pass
                    if val_cached > best_val:
                        best_val = val_cached
                        best_params = full_params.copy()
                        best_stats = cached["stats"]
                    return val_cached

                self.metrics["result_cache_misses"] += 1

                sim_result = self._run_pybroker_simulation(
                    data_frame_local,
                    dates_local,
                    symbol_key,
                    entries,
                    exits,
                    fee_key,
                    slippage_key,
                    timeframe_key,
                    fractional_local,
                    bars_per_year_local,
                )
                if sim_result is None:
                    return float("-inf")
                returns, equity_curve, stats = sim_result
                self.metrics["param_evals"] += 1
                metric_val = self._evaluate_metric(
                    self.cfg.metric, returns, equity_curve, bars_per_year_local
                )
                if not np.isfinite(metric_val):
                    return float("-inf")
                if metric_val > best_val:
                    best_val = metric_val
                    best_params = full_params.copy()
                    best_stats = stats
                    self.results_cache.set(
                        collection=collection_key,
                        symbol=symbol_key,
                        timeframe=timeframe_key,
                        strategy=strategy_key,
                        params=full_params,
                        metric_name=self.cfg.metric,
                        metric_value=float(best_val),
                        stats=stats,
                        data_fingerprint=fingerprint_key,
                        fees=fee_key,
                        slippage=slippage_key,
                        run_id=self.run_id,
                    )
                return float(metric_val)

            space_items = list(search_space.items())

            if search_space:
                if search_method == "optuna":
                    try:
                        import optuna
                    except Exception:
                        search_method = "grid"
                if search_method == "optuna":

                    def objective(trial, space=space_items):
                        var_params = {
                            name: trial.suggest_categorical(name, options)
                            for name, options in space
                        }
                        result = evaluate(var_params)
                        return result if np.isfinite(result) else float("-inf")

                    total_combos = 1
                    for options in search_space.values():
                        total_combos *= max(1, len(options))
                    n_trials = min(trials_target, max(1, total_combos))
                    study = optuna.create_study(direction="maximize")
                    study.optimize(objective, n_trials=n_trials)
                else:
                    for params in self._grid(search_space):
                        evaluate(params)
            else:
                evaluate({})

            if best_params is not None and best_stats is not None:
                best_results.append(
                    BestResult(
                        collection=col.name,
                        symbol=symbol,
                        timeframe=timeframe,
                        strategy=strat.name,
                        params=best_params,
                        metric_name=self.cfg.metric,
                        metric_value=float(best_val),
                        stats=best_stats,
                    )
                )

        if isinstance(self.metrics.get("strategies_used"), set):
            self.metrics["strategies_count"] = len(self.metrics["strategies_used"])  # type: ignore
            self.metrics.pop("strategies_used", None)
        return best_results
