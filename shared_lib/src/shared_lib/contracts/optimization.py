"""Portfolio optimizer request/response schemas."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import Field, field_validator, model_validator

from shared_lib.contracts._base import Schema

Objective = Literal["mean_variance", "mean_cvar", "risk_parity", "min_variance"]


class OptimizerRequest(Schema):
    request_id: str = Field(min_length=1)
    universe: tuple[str, ...] = Field(min_length=1)
    objective: Objective
    gross_leverage: Decimal = Field(gt=0)
    bounds: dict[str, tuple[Decimal, Decimal]] = Field(default_factory=dict)
    risk_aversion: Decimal = Field(ge=0)

    @field_validator("bounds")
    @classmethod
    def _bounds_sane(
        cls, v: dict[str, tuple[Decimal, Decimal]]
    ) -> dict[str, tuple[Decimal, Decimal]]:
        for sym, (lo, hi) in v.items():
            if lo > hi:
                raise ValueError(f"bounds[{sym}]: lower > upper")
            if hi > Decimal("1"):
                raise ValueError(f"bounds[{sym}]: upper must not exceed 1.0")
        return v


class OptimizerResponse(Schema):
    request_id: str = Field(min_length=1)
    weights: dict[str, Decimal] = Field(min_length=1)
    objective_value: Decimal

    @model_validator(mode="after")
    def _weights_sum_to_1(self) -> OptimizerResponse:
        total = sum(self.weights.values(), start=Decimal("0"))
        if abs(total - Decimal("1")) > Decimal("1e-6"):
            raise ValueError(f"weights must sum to 1.0; got {total}")
        return self
