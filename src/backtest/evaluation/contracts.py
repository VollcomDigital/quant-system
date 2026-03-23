from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

EvaluationMode = Literal["backtest", "walk_forward"]


@dataclass(frozen=True)
class EvaluationModeConfig:
    mode: EvaluationMode = "backtest"
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationSharedFields:
    collection: str
    symbol: str
    timeframe: str
    strategy: str
    params: dict[str, Any]
    metric_name: str
    data_fingerprint: str
    fees: float
    slippage: float


@dataclass(frozen=True)
class EvaluationRequest(EvaluationSharedFields):
    source: str
    bars_per_year: int
    mode_config: EvaluationModeConfig
    result_consistency_outlier_dependency_slices: int | None = None
    result_consistency_outlier_dependency_profit_share_threshold: float | None = None
    result_consistency_execution_price_tolerance_bps: float | None = None


@dataclass(frozen=True)
class EvaluationOutcome:
    metric_value: float
    stats: dict[str, Any]
    valid: bool
    attempted: bool
    simulation_executed: bool
    metric_computed: bool
    reject_reason: str | None = None
    artifacts_meta: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class EvaluationCacheRecord(EvaluationSharedFields):
    metric_value: float
    stats: dict[str, Any]
    evaluation_mode: EvaluationMode
    mode_config_hash: str
    validation_config_hash: str

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "EvaluationCacheRecord":
        evaluation_mode = str(payload["evaluation_mode"]).strip().lower()
        if evaluation_mode not in {"backtest", "walk_forward"}:
            raise ValueError(
                "Invalid evaluation_mode in EvaluationCacheRecord payload: "
                f"{payload['evaluation_mode']}"
            )
        return cls(
            collection=str(payload["collection"]),
            symbol=str(payload["symbol"]),
            timeframe=str(payload["timeframe"]),
            strategy=str(payload["strategy"]),
            params=dict(payload["params"]),
            metric_name=str(payload["metric_name"]),
            metric_value=float(payload["metric_value"]),
            stats=dict(payload["stats"]),
            data_fingerprint=str(payload["data_fingerprint"]),
            fees=float(payload["fees"]),
            slippage=float(payload["slippage"]),
            evaluation_mode=cast(EvaluationMode, evaluation_mode),
            mode_config_hash=str(payload["mode_config_hash"]),
            validation_config_hash=str(payload["validation_config_hash"]),
        )
    
@dataclass(frozen=True)
class ResultRecord(EvaluationSharedFields):
    run_id: str | None
    evaluation_mode: EvaluationMode
    source: str
    metric_value: float
    stats: dict[str, Any]
    mode_config_hash: str
