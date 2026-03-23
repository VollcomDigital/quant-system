from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .contracts import EvaluationOutcome, EvaluationRequest


class Evaluator(Protocol):
    def evaluate(
        self,
        request: EvaluationRequest,
        data_frame: pd.DataFrame,
        dates: pd.DatetimeIndex,
        entries: pd.Series,
        exits: pd.Series,
        fractional: bool,
    ) -> EvaluationOutcome:
        raise NotImplementedError


class BacktestEvaluator:
    _TRADES_LOG_LIMIT = 50

    def __init__(
        self,
        simulation_fn: Callable[
            ..., tuple[pd.Series, pd.Series, dict[str, Any], pd.DataFrame] | None
        ],
        metric_fn: Callable[[str, pd.Series, pd.Series, int], float],
    ):
        self._simulation_fn = simulation_fn
        self._metric_fn = metric_fn

    @staticmethod
    def _extract_trade_pnl_series(trades_frame: pd.DataFrame) -> pd.Series:
        for key in ("pnl", "profit", "net_profit", "pl", "p/l"):
            if key not in trades_frame.columns:
                continue
            values = pd.to_numeric(trades_frame[key], errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan)
            return values
        return pd.Series(np.nan, index=trades_frame.index, dtype=float)

    @staticmethod
    def _extract_trade_timestamp_series(trades_frame: pd.DataFrame) -> pd.Series:
        for key in ("exit_date", "exit_time", "close_date", "date", "entry_date", "entry_time"):
            if key not in trades_frame.columns:
                continue
            return pd.to_datetime(trades_frame[key], errors="coerce")
        return pd.Series(pd.NaT, index=trades_frame.index)

    @classmethod
    def _build_report_trades_log(cls, trades_frame: pd.DataFrame) -> list[dict[str, Any]]:
        if not trades_frame.empty:
            serialized = trades_frame.copy()
            for column in serialized.columns:
                if pd.api.types.is_datetime64_any_dtype(serialized[column]):
                    serialized[column] = serialized[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
            return serialized.head(cls._TRADES_LOG_LIMIT).to_dict("records")
        return []

    @staticmethod
    def _build_incomplete_trade_meta(
        analyzed_trades_count: int,
        total_trades: int,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "is_complete": False,
            "analyzed_trades_count": analyzed_trades_count,
            "total_trades": total_trades,
            "reason": reason,
        }

    @staticmethod
    def _build_execution_variance_incomplete_meta(
        checked_fills: int,
        violations: int,
        reason: str,
        tolerance_bps: float | None,
    ) -> dict[str, Any]:
        return {
            "is_complete": False,
            "checked_fills": checked_fills,
            "violations": violations,
            "violation_ratio": None,
            "reason": reason,
            "price_tolerance_bps_used": tolerance_bps,
        }

    def _validate_trade_counts(
        self, trades_frame: pd.DataFrame, stats: dict[str, Any]
    ) -> tuple[bool, dict[str, Any] | None]:
        total_trades = int(len(trades_frame))
        raw_trade_count = stats.get("trades")
        if raw_trade_count is None:
            return False, self._build_incomplete_trade_meta(
                analyzed_trades_count=total_trades,
                total_trades=total_trades,
                reason="missing_trade_count_metric",
            )
        try:
            expected_trades = int(raw_trade_count)
        except (TypeError, ValueError):
            return False, self._build_incomplete_trade_meta(
                analyzed_trades_count=total_trades,
                total_trades=total_trades,
                reason="invalid_trade_count_metric",
            )
        if expected_trades != total_trades:
            return False, self._build_incomplete_trade_meta(
                analyzed_trades_count=total_trades,
                total_trades=expected_trades,
                reason="truncated_trades_frame",
            )
        return True, None

    @staticmethod
    def _compute_dominant_trade_share(
        pnls: np.ndarray,
        threshold: float | None,
    ) -> tuple[int | None, float | None]:
        if threshold is None or pnls.size == 0:
            return None, None
        winners_desc = np.sort(pnls[pnls > 0])[::-1]
        if winners_desc.size == 0:
            return None, None
        cumulative = np.cumsum(winners_desc)
        required_profit = float(threshold) * float(cumulative[-1])
        dominant_trade_count = int(np.searchsorted(cumulative, required_profit, side="left") + 1)
        dominant_trade_share = dominant_trade_count / float(pnls.size)
        return dominant_trade_count, dominant_trade_share

    @staticmethod
    def _compute_max_slice_profit_share(
        paired_pnls: np.ndarray,
        exit_times: pd.DatetimeIndex,
        slices: int | None,
    ) -> float | None:
        if slices is None or slices < 2 or len(paired_pnls) < slices:
            return None
        start = exit_times.min()
        end = exit_times.max()
        if start >= end:
            return None
        positive_pnls = np.where(paired_pnls > 0, paired_pnls, 0.0)
        grouped = (
            pd.Series(positive_pnls, index=exit_times)
            .groupby(pd.cut(exit_times, bins=slices), observed=False)
            .sum()
        )
        grouped_sum = float(grouped.sum())
        if grouped_sum <= 0:
            return None
        return float(grouped.max() / grouped_sum)

    def _build_trade_meta(
        self,
        trades_frame: pd.DataFrame,
        data_frame: pd.DataFrame,
        dates: pd.DatetimeIndex,
        stats: dict[str, Any],
        request: EvaluationRequest,
    ) -> dict[str, Any]:
        outlier_meta = self._build_outlier_dependency_meta(trades_frame, stats, request)

        execution_variance_meta = self._build_execution_price_variance_meta(
            trades_frame,
            data_frame,
            dates,
            tolerance_bps=request.result_consistency_execution_price_tolerance_bps,
        )

        return {
            "outlier_dependency": outlier_meta,
            "execution_price_variance": execution_variance_meta,
        }

    def _build_outlier_dependency_meta(
        self,
        trades_frame: pd.DataFrame,
        stats: dict[str, Any],
        request: EvaluationRequest,
    ) -> dict[str, Any]:
        is_complete, incomplete_meta = self._validate_trade_counts(trades_frame, stats)
        if not is_complete and isinstance(incomplete_meta, dict):
            return dict(incomplete_meta)

        total_trades = int(len(trades_frame))
        pnl_series = self._extract_trade_pnl_series(trades_frame)
        valid_pnl = pnl_series.notna()
        pnls = pnl_series[valid_pnl].to_numpy(dtype=float)
        exit_series = self._extract_trade_timestamp_series(trades_frame)
        # Slice concentration requires both valid pnl and valid timestamp for each trade.
        valid_exit = exit_series.notna()
        paired = valid_pnl & valid_exit
        paired_pnls = pnl_series[paired].to_numpy(dtype=float)
        exit_times = pd.DatetimeIndex(exit_series[paired])

        total_positive_profit = float(np.where(pnls > 0, pnls, 0.0).sum())
        threshold = request.result_consistency_outlier_dependency_profit_share_threshold
        dominant_trade_count, dominant_trade_share = self._compute_dominant_trade_share(
            pnls,
            threshold,
        )
        slices = request.result_consistency_outlier_dependency_slices
        max_slice_profit_share = self._compute_max_slice_profit_share(
            paired_pnls,
            exit_times,
            slices,
        )
        return {
            "is_complete": True,
            "analyzed_trades_count": total_trades,
            "total_trades": total_trades,
            "trade_count_with_pnl": int(len(pnls)),
            "total_positive_profit": total_positive_profit,
            "profit_share_threshold_used": threshold,
            "dominant_trade_count_for_profit_share": dominant_trade_count,
            "dominant_trade_share_for_profit_share": dominant_trade_share,
            "slices_used": slices,
            "max_slice_profit_share": max_slice_profit_share,
        }

    @classmethod
    def _build_execution_price_variance_meta(
        cls,
        trades_frame: pd.DataFrame,
        data_frame: pd.DataFrame,
        dates: pd.DatetimeIndex,
        tolerance_bps: float | None,
    ) -> dict[str, Any]:
        if tolerance_bps is None:
            return cls._build_execution_variance_incomplete_meta(
                checked_fills=0,
                violations=0,
                reason="policy_disabled",
                tolerance_bps=None,
            )
        if trades_frame.empty:
            return cls._build_execution_variance_incomplete_meta(
                checked_fills=0,
                violations=0,
                reason="missing_trades_frame",
                tolerance_bps=tolerance_bps,
            )
        high_col = (
            "high"
            if "high" in data_frame.columns
            else "High" if "High" in data_frame.columns else None
        )
        low_col = (
            "low"
            if "low" in data_frame.columns
            else "Low" if "Low" in data_frame.columns else None
        )
        if high_col is None or low_col is None:
            return cls._build_execution_variance_incomplete_meta(
                checked_fills=0,
                violations=0,
                reason="missing_ohlc_columns",
                tolerance_bps=tolerance_bps,
            )

        bars = pd.DataFrame(
            {
                "high": pd.to_numeric(data_frame[high_col], errors="coerce").to_numpy(dtype=float),
                "low": pd.to_numeric(data_frame[low_col], errors="coerce").to_numpy(dtype=float),
            },
            index=pd.to_datetime(dates, errors="coerce", utc=True),
        )
        bars = bars.dropna(subset=["high", "low"])
        if bars.empty:
            return cls._build_execution_variance_incomplete_meta(
                checked_fills=0,
                violations=0,
                reason="missing_bar_timestamps",
                tolerance_bps=tolerance_bps,
            )

        fill_specs = (
            ("entry_price", ("entry_date", "entry_time", "entry_datetime")),
            ("exit_price", ("exit_date", "exit_time", "close_date", "close_time")),
        )
        has_fill_columns = False
        checked_fills = 0
        violations = 0
        unmatched_timestamps = 0

        for price_col, time_candidates in fill_specs:
            if price_col not in trades_frame.columns:
                continue
            time_col = next((c for c in time_candidates if c in trades_frame.columns), None)
            if time_col is None:
                continue
            has_fill_columns = True
            price_series = pd.to_numeric(trades_frame[price_col], errors="coerce")
            ts_series = pd.to_datetime(trades_frame[time_col], errors="coerce", utc=True)
            valid = price_series.notna() & ts_series.notna()
            if not valid.any():
                continue

            for price, ts in zip(price_series[valid].to_numpy(dtype=float), ts_series[valid]):
                bar = bars.loc[ts] if ts in bars.index else None
                if bar is None:
                    unmatched_timestamps += 1
                    continue
                if isinstance(bar, pd.DataFrame):
                    high = float(pd.to_numeric(bar["high"], errors="coerce").iloc[-1])
                    low = float(pd.to_numeric(bar["low"], errors="coerce").iloc[-1])
                else:
                    high = float(bar["high"])
                    low = float(bar["low"])
                tol = abs(price) * (float(tolerance_bps) / 10_000.0)
                checked_fills += 1
                if price < (low - tol) or price > (high + tol):
                    violations += 1

        if not has_fill_columns:
            return cls._build_execution_variance_incomplete_meta(
                checked_fills=0,
                violations=0,
                reason="missing_fill_columns",
                tolerance_bps=tolerance_bps,
            )
        if checked_fills == 0:
            return cls._build_execution_variance_incomplete_meta(
                checked_fills=0,
                violations=0,
                reason="missing_fill_timestamps",
                tolerance_bps=tolerance_bps,
            )
        if unmatched_timestamps > 0:
            return cls._build_execution_variance_incomplete_meta(
                checked_fills=checked_fills,
                violations=violations,
                reason="unmatched_fill_timestamps",
                tolerance_bps=tolerance_bps,
            )
        return {
            "is_complete": True,
            "checked_fills": checked_fills,
            "violations": violations,
            "violation_ratio": float(violations / checked_fills) if checked_fills > 0 else None,
            "reason": None,
            "price_tolerance_bps_used": tolerance_bps,
        }

    def evaluate(
        self,
        request: EvaluationRequest,
        data_frame: pd.DataFrame,
        dates: pd.DatetimeIndex,
        entries: pd.Series,
        exits: pd.Series,
        fractional: bool,
    ) -> EvaluationOutcome:
        sim_result = self._simulation_fn(
            data_frame,
            dates,
            request.symbol,
            entries,
            exits,
            request.fees,
            request.slippage,
            request.timeframe,
            fractional,
            request.bars_per_year,
        )
        if sim_result is None:
            return EvaluationOutcome(
                metric_value=float("-inf"),
                stats={},
                valid=False,
                attempted=True,
                simulation_executed=False,
                metric_computed=False,
                reject_reason="simulation_failed",
            )

        returns, equity_curve, stats, trades_frame = sim_result
        if not isinstance(trades_frame, pd.DataFrame):
            raise ValueError("simulation_fn must return a pandas DataFrame for trades_frame")
        stats = dict(stats)
        # Derive reporting log and consistency metadata from full trades when available.
        stats["trades_log"] = self._build_report_trades_log(trades_frame)
        stats["trade_meta"] = self._build_trade_meta(
            trades_frame,
            data_frame,
            dates,
            stats,
            request,
        )
        metric_val = self._metric_fn(
            request.metric_name,
            returns,
            equity_curve,
            request.bars_per_year,
        )
        if not np.isfinite(metric_val):
            return EvaluationOutcome(
                metric_value=float("-inf"),
                stats=dict(stats),
                valid=False,
                attempted=True,
                simulation_executed=True,
                metric_computed=True,
                reject_reason="metric_not_finite",
            )

        return EvaluationOutcome(
            metric_value=float(metric_val),
            stats=dict(stats),
            valid=True,
            attempted=True,
            simulation_executed=True,
            metric_computed=True,
        )
