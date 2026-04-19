"""Phase 6 Task 5 - mid_freq_engine.model_serving gRPC client contract.

`ModelServingClient` is a Protocol: `.predict(InferenceRequest) ->
InferenceResponse`. Phase 6 ships the contract plus an in-memory
reference implementation. A Triton/gRPC transport lands in Phase 9.

Invariants:
- InferenceRequest carries `model_id`, `version`, `inputs` (dict of
  feature-name -> value), `deadline` (timezone-aware), and an
  `idempotency_key`.
- InferenceResponse carries `model_id`, `version`, `predictions`, and
  the server-side confidence (must be in [0,1]).
- Missing required features raise `ValueError` at the client layer
  before any RPC is made.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest


def _request(**kwargs):
    from trading_system.mid_freq_engine.model_serving import InferenceRequest

    defaults = dict(
        model_id="kronos-v0",
        version="v1",
        inputs={"price": Decimal("100"), "volume": Decimal("1000")},
        deadline=datetime.now(tz=UTC) + timedelta(seconds=10),
        idempotency_key="req-1",
    )
    defaults.update(kwargs)
    return InferenceRequest(**defaults)


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def test_inference_request_rejects_naive_deadline() -> None:
    from trading_system.mid_freq_engine.model_serving import InferenceRequest

    with pytest.raises(ValueError):
        InferenceRequest(
            model_id="m",
            version="v1",
            inputs={"x": Decimal("1")},
            deadline=datetime(2026, 4, 19),
            idempotency_key="k",
        )


def test_inference_request_rejects_empty_inputs() -> None:
    from trading_system.mid_freq_engine.model_serving import InferenceRequest

    with pytest.raises(ValueError, match="inputs"):
        InferenceRequest(
            model_id="m",
            version="v1",
            inputs={},
            deadline=datetime.now(tz=UTC) + timedelta(seconds=5),
            idempotency_key="k",
        )


def test_inference_response_confidence_must_be_in_unit_interval() -> None:
    from trading_system.mid_freq_engine.model_serving import InferenceResponse

    with pytest.raises(ValueError):
        InferenceResponse(
            model_id="m",
            version="v1",
            predictions={"signal": Decimal("0.01")},
            confidence=Decimal("1.5"),
        )


# ---------------------------------------------------------------------------
# In-memory reference client
# ---------------------------------------------------------------------------


def test_in_memory_client_returns_registered_response() -> None:
    from trading_system.mid_freq_engine.model_serving import (
        InferenceResponse,
        InMemoryModelServingClient,
    )

    client = InMemoryModelServingClient()
    client.register(
        model_id="m",
        version="v1",
        responder=lambda req: InferenceResponse(
            model_id=req.model_id,
            version=req.version,
            predictions={"signal": Decimal("0.02")},
            confidence=Decimal("0.9"),
        ),
    )
    resp = client.predict(_request(model_id="m", version="v1"))
    assert resp.predictions["signal"] == Decimal("0.02")


def test_in_memory_client_raises_on_unknown_model() -> None:
    from trading_system.mid_freq_engine.model_serving import InMemoryModelServingClient

    client = InMemoryModelServingClient()
    with pytest.raises(LookupError):
        client.predict(_request(model_id="unknown"))


def test_in_memory_client_refuses_expired_deadline() -> None:
    from trading_system.mid_freq_engine.model_serving import (
        InferenceResponse,
        InMemoryModelServingClient,
    )

    client = InMemoryModelServingClient()
    client.register(
        model_id="m",
        version="v1",
        responder=lambda req: InferenceResponse(
            model_id=req.model_id,
            version=req.version,
            predictions={"signal": Decimal("0.02")},
            confidence=Decimal("0.9"),
        ),
    )
    past = datetime.now(tz=UTC) - timedelta(seconds=1)
    with pytest.raises(TimeoutError):
        # Build the request with a small positive deadline then sleep
        # is awkward; instead, directly craft an expired request.
        from trading_system.mid_freq_engine.model_serving import InferenceRequest
        client.predict(
            InferenceRequest(
                model_id="m",
                version="v1",
                inputs={"x": Decimal("1")},
                deadline=past,
                idempotency_key="expired",
            )
        )


def test_in_memory_client_validates_response_shape() -> None:
    from trading_system.mid_freq_engine.model_serving import (
        InferenceResponse,
        InMemoryModelServingClient,
    )

    client = InMemoryModelServingClient()
    client.register(
        model_id="m",
        version="v1",
        responder=lambda req: InferenceResponse(
            model_id="WRONG",  # model_id mismatch -> client raises
            version=req.version,
            predictions={"signal": Decimal("0.02")},
            confidence=Decimal("0.9"),
        ),
    )
    with pytest.raises(ValueError, match="mismatch"):
        client.predict(_request(model_id="m", version="v1"))
