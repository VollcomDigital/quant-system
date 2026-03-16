from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EvaluationMode = Literal["backtest", "walk_forward"]


@dataclass(frozen=True)
class EvaluationModeConfig:
    mode: EvaluationMode = "backtest"
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRequest:
    collection: str
    symbol: str
    timeframe: str
    source: str
    strategy: str
    params: dict[str, Any]
    metric_name: str
    data_fingerprint: str
    fees: float
    slippage: float
    bars_per_year: int
    mode_config: EvaluationModeConfig


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
class EvaluationCacheRecord:
    collection: str
    symbol: str
    timeframe: str
    strategy: str
    params: dict[str, Any]
    metric_name: str
    metric_value: float
    stats: dict[str, Any]
    data_fingerprint: str
    fees: float
    slippage: float
    evaluation_mode: str
    mode_config_hash: str
    validation_config_hash: str

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "EvaluationCacheRecord":
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
            evaluation_mode=str(payload["evaluation_mode"]),
            mode_config_hash=str(payload["mode_config_hash"]),
            validation_config_hash=str(payload["validation_config_hash"]),
        )
    
@dataclass(frozen=True)
class ResultRecord:
    run_id: str | None
    evaluation_mode: EvaluationMode
    collection: str
    symbol: str
    timeframe: str
    source: str
    strategy: str
    params: dict[str, Any]
    metric_name: str
    metric_value: float
    stats: dict[str, Any]
    data_fingerprint: str
    fees: float
    slippage: float
    mode_config_hash: str
