from __future__ import annotations

from collections.abc import Callable, Iterator
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
            return pd.to_datetime(trades_frame[key], errors="coerce", utc=True)
        return pd.Series(pd.NaT, index=trades_frame.index, dtype="datetime64[ns, UTC]")

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
        expected_trades: int | None = None,
    ) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "is_complete": False,
            "analyzed_trades_count": analyzed_trades_count,
            "total_trades": total_trades,
            "reason": reason,
        }
        if expected_trades is not None:
            meta["expected_trades"] = expected_trades
        return meta

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
        self,
        trades_frame: pd.DataFrame,
        expected_trade_count_raw: Any,
    ) -> tuple[bool, dict[str, Any] | None]:
        total_trades = int(len(trades_frame))
        if expected_trade_count_raw is None:
            return False, self._build_incomplete_trade_meta(
                analyzed_trades_count=total_trades,
                total_trades=total_trades,
                reason="missing_trade_count_metric",
            )
        try:
            expected_trades = int(expected_trade_count_raw)
        except (TypeError, ValueError):
            return False, self._build_incomplete_trade_meta(
                analyzed_trades_count=total_trades,
                total_trades=total_trades,
                reason="invalid_trade_count_metric",
            )
        if expected_trades != total_trades:
            return False, self._build_incomplete_trade_meta(
                analyzed_trades_count=total_trades,
                total_trades=total_trades,
                reason="truncated_trades_frame",
                expected_trades=expected_trades,
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
        # Share is measured against all trades with valid pnl, not only winners.
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
        expected_trade_count_raw: Any,
        request: EvaluationRequest,
    ) -> dict[str, Any]:
        outlier_meta = self._build_outlier_dependency_meta(
            trades_frame,
            expected_trade_count_raw,
            request,
        )

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
        expected_trade_count_raw: Any,
        request: EvaluationRequest,
    ) -> dict[str, Any]:
        is_complete, incomplete_meta = self._validate_trade_counts(
            trades_frame,
            expected_trade_count_raw,
        )
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
        ohlc_columns = cls._resolve_execution_range_columns(data_frame)
        if ohlc_columns is None:
            return cls._build_execution_variance_incomplete_meta(
                checked_fills=0,
                violations=0,
                reason="missing_ohlc_columns",
                tolerance_bps=tolerance_bps,
            )
        high_col, low_col = ohlc_columns
        bars = cls._build_execution_bars_lookup(data_frame, dates, high_col, low_col)
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
        has_fill_columns, checked_fills, violations, unmatched_timestamps = (
            cls._compute_execution_fill_counters(
                trades_frame=trades_frame,
                bars=bars,
                fill_specs=fill_specs,
                tolerance_bps=float(tolerance_bps),
            )
        )

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

    @staticmethod
    def _resolve_execution_range_columns(data_frame: pd.DataFrame) -> tuple[str, str] | None:
        candidates = (
            ("high", "low"),
            ("High", "Low"),
        )
        for high_col, low_col in candidates:
            if high_col in data_frame.columns and low_col in data_frame.columns:
                return high_col, low_col
        return None

    @staticmethod
    def _build_execution_bars_lookup(
        data_frame: pd.DataFrame,
        dates: pd.DatetimeIndex,
        high_col: str,
        low_col: str,
    ) -> pd.DataFrame:
        bars = pd.DataFrame(
            {
                "high": pd.to_numeric(data_frame[high_col], errors="coerce").to_numpy(dtype=float),
                "low": pd.to_numeric(data_frame[low_col], errors="coerce").to_numpy(dtype=float),
            },
            index=pd.to_datetime(dates, errors="coerce", utc=True),
        )
        if bars.index.has_duplicates:
            # Keep last bar for duplicate timestamps to match previous behavior.
            bars = bars[~bars.index.duplicated(keep="last")]
        return bars

    @staticmethod
    def _iter_execution_fill_series(
        trades_frame: pd.DataFrame,
        fill_specs: tuple[tuple[str, tuple[str, ...]], ...],
    ) -> Iterator[tuple[pd.Series, pd.Series]]:
        for price_col, time_candidates in fill_specs:
            if price_col not in trades_frame.columns:
                continue
            time_col = next((c for c in time_candidates if c in trades_frame.columns), None)
            if time_col is None:
                continue
            price_series = pd.to_numeric(trades_frame[price_col], errors="coerce")
            ts_series = pd.to_datetime(trades_frame[time_col], errors="coerce", utc=True)
            yield price_series, ts_series

    @classmethod
    def _compute_execution_fill_counters(
        cls,
        *,
        trades_frame: pd.DataFrame,
        bars: pd.DataFrame,
        fill_specs: tuple[tuple[str, tuple[str, ...]], ...],
        tolerance_bps: float,
    ) -> tuple[bool, int, int, int]:
        has_fill_columns = False
        checked_fills = 0
        violations = 0
        unmatched_timestamps = 0
        for price_series, ts_series in cls._iter_execution_fill_series(trades_frame, fill_specs):
            has_fill_columns = True
            spec_checked, spec_violations, spec_unmatched = cls._count_execution_fill_violations(
                bars=bars,
                price_series=price_series,
                ts_series=ts_series,
                tolerance_bps=tolerance_bps,
            )
            checked_fills += spec_checked
            violations += spec_violations
            unmatched_timestamps += spec_unmatched
        return has_fill_columns, checked_fills, violations, unmatched_timestamps

    @staticmethod
    def _count_execution_fill_violations(
        *,
        bars: pd.DataFrame,
        price_series: pd.Series,
        ts_series: pd.Series,
        tolerance_bps: float,
    ) -> tuple[int, int, int]:
        valid = price_series.notna() & ts_series.notna()
        if not valid.any():
            return 0, 0, 0
        fills = pd.DataFrame(
            {"price": price_series[valid].to_numpy(dtype=float)},
            index=pd.DatetimeIndex(ts_series[valid]),
        )
        bar_match = bars.reindex(fills.index)
        matched = bar_match["high"].notna() & bar_match["low"].notna()
        unmatched_timestamps = int((~matched).sum())
        if not matched.any():
            return 0, 0, unmatched_timestamps

        matched_prices = fills.loc[matched, "price"].to_numpy(dtype=float)
        matched_high = bar_match.loc[matched, "high"].to_numpy(dtype=float)
        matched_low = bar_match.loc[matched, "low"].to_numpy(dtype=float)
        tolerance = np.abs(matched_prices) * (tolerance_bps / 10_000.0)
        out_of_range = (matched_prices < (matched_low - tolerance)) | (
            matched_prices > (matched_high + tolerance)
        )
        checked_fills = int(matched.sum())
        violations = int(np.sum(out_of_range))
        return checked_fills, violations, unmatched_timestamps

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
        expected_trade_count_raw = stats.get("trades")
        # Derive reporting log and consistency metadata from full trades when available.
        stats["trades_log"] = self._build_report_trades_log(trades_frame)
        stats["trade_meta"] = self._build_trade_meta(
            trades_frame,
            data_frame,
            dates,
            expected_trade_count_raw,
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
