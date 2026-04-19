"""RL environment metadata."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import Field, model_validator

from shared_lib.contracts._base import Schema

ActionSpaceKind = Literal["discrete", "continuous"]


class RLEnvironmentMetadata(Schema):
    env_id: str = Field(min_length=1)
    observation_space_shape: tuple[int, ...]
    action_space_kind: ActionSpaceKind
    action_space_bounds: tuple[Decimal, Decimal] | None = None
    reward_scale: Decimal = Field(gt=0)

    @model_validator(mode="after")
    def _shape_is_positive(self) -> RLEnvironmentMetadata:
        if not self.observation_space_shape:
            raise ValueError("observation_space_shape must not be empty")
        if any(d <= 0 for d in self.observation_space_shape):
            raise ValueError("observation_space_shape dims must be > 0")
        if self.action_space_kind == "continuous" and self.action_space_bounds is None:
            raise ValueError("continuous action space requires bounds")
        return self
