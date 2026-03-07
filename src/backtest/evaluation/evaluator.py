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
    def __init__(
        self,
        simulation_fn: Callable[..., tuple[pd.Series, pd.Series, dict[str, Any]] | None],
        metric_fn: Callable[[str, pd.Series, pd.Series, int], float],
    ):
        self._simulation_fn = simulation_fn
        self._metric_fn = metric_fn

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

        returns, equity_curve, stats = sim_result
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
