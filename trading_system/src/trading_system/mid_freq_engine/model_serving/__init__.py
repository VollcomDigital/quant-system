"""Model-serving client contract.

Phase 6 ships an in-memory reference client. A Triton/gRPC adapter
plugs in behind `ModelServingClient` in Phase 9. The contract is
designed so that the simulator, the paper-trading OMS, and the live
OMS all consume it the same way.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Protocol, runtime_checkable

from pydantic import Field
from shared_lib.contracts._base import Schema, aware_datetime_validator

__all__ = [
    "InferenceRequest",
    "InferenceResponse",
    "InMemoryModelServingClient",
    "ModelServingClient",
]


class InferenceRequest(Schema):
    model_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    inputs: dict[str, Decimal] = Field(min_length=1)
    deadline: datetime
    idempotency_key: str = Field(min_length=1)

    _ts = aware_datetime_validator("deadline")


class InferenceResponse(Schema):
    model_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    predictions: dict[str, Decimal]
    confidence: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))


@runtime_checkable
class ModelServingClient(Protocol):
    def predict(self, request: InferenceRequest) -> InferenceResponse:
        ...


@dataclass
class InMemoryModelServingClient:
    _handlers: dict[tuple[str, str], Callable[[InferenceRequest], InferenceResponse]] = field(
        default_factory=dict, init=False, repr=False
    )

    def register(
        self,
        *,
        model_id: str,
        version: str,
        responder: Callable[[InferenceRequest], InferenceResponse],
    ) -> None:
        self._handlers[(model_id, version)] = responder

    def predict(self, request: InferenceRequest) -> InferenceResponse:
        if request.deadline <= datetime.now(tz=UTC):
            raise TimeoutError(
                f"deadline {request.deadline.isoformat()} already elapsed"
            )
        key = (request.model_id, request.version)
        try:
            responder = self._handlers[key]
        except KeyError as exc:
            raise LookupError(f"no handler for {key!r}") from exc
        response = responder(request)
        if response.model_id != request.model_id or response.version != request.version:
            raise ValueError(
                f"response model_id/version mismatch: "
                f"request={key}, response=({response.model_id}, {response.version})"
            )
        return response
