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
    def _extract_trade_pnl(trade: dict[str, Any]) -> float | None:
        for key in ("pnl", "profit", "net_profit", "pl", "p/l"):
            if key not in trade:
                continue
            try:
                value = float(trade[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                return value
        return None

    @staticmethod
    def _extract_trade_timestamp(trade: dict[str, Any]) -> pd.Timestamp | None:
        for key in ("exit_date", "exit_time", "close_date", "date", "entry_date", "entry_time"):
            if key not in trade:
                continue
            parsed = pd.to_datetime(trade[key], errors="coerce")
            if pd.isna(parsed):
                continue
            return pd.Timestamp(parsed)
        return None

    @staticmethod
    def _normalized_trades_for_analysis(
        trades_frame: pd.DataFrame,
        stats: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], int, bool, str | None]:
        if not trades_frame.empty:
            records = trades_frame.to_dict("records")
            return records, len(records), True, None

        trades_log = stats.get("trades_log")
        if not isinstance(trades_log, list):
            return [], int(stats.get("trades", 0)), False, "missing_trades_log"
        total_trades = int(stats.get("trades", len(trades_log)))
        if total_trades != len(trades_log):
            return trades_log, total_trades, False, "truncated_trades_log"
        return trades_log, total_trades, True, None

    @classmethod
    def _build_report_trades_log(
        cls, trades_frame: pd.DataFrame, stats: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if not trades_frame.empty:
            serialized = trades_frame.copy()
            for column in serialized.columns:
                if pd.api.types.is_datetime64_any_dtype(serialized[column]):
                    serialized[column] = serialized[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
            return serialized.head(cls._TRADES_LOG_LIMIT).to_dict("records")
        trades_log = stats.get("trades_log")
        if isinstance(trades_log, list):
            return [trade for trade in trades_log if isinstance(trade, dict)]
        return []

    def _build_trade_meta(
        self,
        trades_frame: pd.DataFrame,
        stats: dict[str, Any],
        request: EvaluationRequest,
    ) -> dict[str, Any]:
        trades_rows, total_trades, is_complete, reason = self._normalized_trades_for_analysis(
            trades_frame, stats
        )
        if not is_complete:
            return {
                "is_complete": bool(is_complete),
                "trades_log_count": len(trades_rows),
                "total_trades": total_trades,
                "reason": str(reason),
            }

        pnls: list[float] = []
        exit_times: list[pd.Timestamp] = []
        for trade in trades_rows:
            pnl = self._extract_trade_pnl(trade)
            if pnl is None:
                continue
            pnls.append(pnl)
            ts = self._extract_trade_timestamp(trade)
            if ts is not None:
                exit_times.append(ts)

        total_positive_profit = float(sum(v for v in pnls if v > 0))
        dominant_trade_count: int | None = None
        dominant_trade_share: float | None = None
        threshold = request.result_consistency_profit_share_threshold
        if threshold is not None and total_positive_profit > 0:
            winners_desc = np.sort(np.asarray([v for v in pnls if v > 0], dtype=float))[::-1]
            if winners_desc.size > 0:
                required_profit = float(threshold) * total_positive_profit
                cumulative = np.cumsum(winners_desc)
                dominant_trade_count = int(np.searchsorted(cumulative, required_profit, side="left") + 1)
                if len(pnls) > 0:
                    dominant_trade_share = dominant_trade_count / float(len(pnls))

        max_slice_profit_share: float | None = None
        slices = request.result_consistency_slices
        if (
            slices is not None
            and slices >= 2
            and len(exit_times) == len(pnls)
            and len(pnls) >= slices
        ):
            timeline = pd.DatetimeIndex(exit_times)
            start = timeline.min()
            end = timeline.max()
            if start < end:
                bins = pd.interval_range(start=start, end=end, periods=slices)
                positive_pnls = np.asarray([v if v > 0 else 0.0 for v in pnls], dtype=float)
                grouped = (
                    pd.Series(positive_pnls, index=timeline)
                    .groupby(pd.cut(timeline, bins), observed=False)
                    .sum()
                )
                grouped_sum = float(grouped.sum())
                if grouped_sum > 0:
                    max_slice_profit_share = float(grouped.max() / grouped_sum)

        return {
            "is_complete": True,
            "trades_log_count": len(trades_rows),
            "total_trades": total_trades,
            "trade_count_with_pnl": len(pnls),
            "total_positive_profit": total_positive_profit,
            "profit_share_threshold_used": threshold,
            "dominant_trade_count_for_profit_share": dominant_trade_count,
            "dominant_trade_share_for_profit_share": dominant_trade_share,
            "slices_used": slices,
            "max_slice_profit_share": max_slice_profit_share,
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

        if len(sim_result) == 4:
            returns, equity_curve, stats, trades_frame = sim_result
            if not isinstance(trades_frame, pd.DataFrame):
                trades_frame = pd.DataFrame(trades_frame)
        else:
            # Backward-compat path for test doubles still returning 3-tuple.
            returns, equity_curve, stats = sim_result
            trades_frame = pd.DataFrame()
        stats = dict(stats)
        # Derive reporting log and consistency metadata from full trades when available.
        stats["trades_log"] = self._build_report_trades_log(trades_frame, stats)
        stats["trade_meta"] = self._build_trade_meta(trades_frame, stats, request)
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
