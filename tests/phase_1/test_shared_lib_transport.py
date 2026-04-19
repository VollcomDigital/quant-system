"""Phase 1 Task 5 - shared_lib.transport envelopes.

Every cross-service message (gRPC, Kafka, ZeroMQ, HTTP) uses one of two
envelopes:

- `RpcEnvelope` - synchronous request/response with idempotency_key,
  trace_id, span_id, deadline, and payload_schema.
- `EventEnvelope` - asynchronous fact with event_id, trace_id, span_id,
  occurred_at, topic, schema_version, and payload.

These envelopes are the stable contract between Python research services
and Rust/C++ execution services. The tests enforce the minimum fields,
validation, and idempotent decode/encode round-trip.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

# ---------------------------------------------------------------------------
# RpcEnvelope
# ---------------------------------------------------------------------------


def test_rpc_envelope_round_trip() -> None:
    from shared_lib.transport import RpcEnvelope

    env = RpcEnvelope(
        idempotency_key="idem-1",
        trace_id="t1",
        span_id="s1",
        deadline=datetime(2026, 4, 19, 14, 0, 0, tzinfo=UTC),
        payload_schema="trading_system.orders.submit.v1",
        payload={"order_id": "o-1", "qty": "100"},
    )
    as_json = env.model_dump_json()
    assert RpcEnvelope.model_validate_json(as_json) == env


def test_rpc_envelope_requires_idempotency_key() -> None:
    from shared_lib.transport import RpcEnvelope

    with pytest.raises(ValueError):
        RpcEnvelope(
            idempotency_key="",
            trace_id="t",
            span_id="s",
            deadline=datetime(2026, 4, 19, tzinfo=UTC),
            payload_schema="x",
            payload={},
        )


def test_rpc_envelope_deadline_must_be_aware() -> None:
    from shared_lib.transport import RpcEnvelope

    with pytest.raises(ValueError):
        RpcEnvelope(
            idempotency_key="i",
            trace_id="t",
            span_id="s",
            deadline=datetime(2026, 4, 19),
            payload_schema="x",
            payload={},
        )


def test_rpc_envelope_deadline_must_be_future_when_created() -> None:
    # Not enforced by pydantic directly, but the helper constructor should
    # reject past deadlines.
    from shared_lib.transport import new_rpc_envelope

    with pytest.raises(ValueError, match="past"):
        new_rpc_envelope(
            payload_schema="x",
            payload={},
            trace_id="t",
            span_id="s",
            deadline=datetime.now(tz=UTC) - timedelta(seconds=1),
        )


# ---------------------------------------------------------------------------
# EventEnvelope
# ---------------------------------------------------------------------------


def test_event_envelope_requires_event_id() -> None:
    from shared_lib.transport import EventEnvelope

    with pytest.raises(ValueError):
        EventEnvelope(
            event_id="",
            trace_id="t",
            span_id="s",
            occurred_at=datetime(2026, 4, 19, tzinfo=UTC),
            topic="trading_system.fills.v1",
            schema_version=1,
            payload={},
        )


def test_event_envelope_schema_version_must_be_positive() -> None:
    from shared_lib.transport import EventEnvelope

    with pytest.raises(ValueError):
        EventEnvelope(
            event_id="e1",
            trace_id="t",
            span_id="s",
            occurred_at=datetime(2026, 4, 19, tzinfo=UTC),
            topic="t",
            schema_version=0,
            payload={},
        )


def test_event_envelope_topic_follows_naming_convention() -> None:
    """Topic names must be lowercase, dot-separated, and carry the owning
    domain as the first segment."""
    from shared_lib.transport import EventEnvelope

    with pytest.raises(ValueError, match="topic"):
        EventEnvelope(
            event_id="e1",
            trace_id="t",
            span_id="s",
            occurred_at=datetime(2026, 4, 19, tzinfo=UTC),
            topic="TradingSystem.Fills.V1",  # camelCase rejected
            schema_version=1,
            payload={},
        )


# ---------------------------------------------------------------------------
# Constructors auto-generate ids and propagate trace context.
# ---------------------------------------------------------------------------


def test_new_event_envelope_auto_generates_ids() -> None:
    from shared_lib.transport import new_event_envelope

    e = new_event_envelope(
        topic="data_platform.bars.v1",
        schema_version=1,
        payload={"symbol": "AAPL"},
        trace_id="t",
        span_id="s",
    )
    assert e.event_id
    assert e.trace_id == "t"


def test_new_rpc_envelope_picks_up_current_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    from shared_lib.logging import bind_trace
    from shared_lib.transport import new_rpc_envelope

    with bind_trace(trace_id="outer", span_id="sp"):
        e = new_rpc_envelope(
            payload_schema="x",
            payload={},
            deadline=datetime.now(tz=UTC) + timedelta(seconds=30),
        )
    assert e.trace_id == "outer"
    assert e.span_id == "sp"


# ---------------------------------------------------------------------------
# DLQ convention
# ---------------------------------------------------------------------------


def test_dlq_topic_name_is_derived() -> None:
    from shared_lib.transport import dlq_topic

    assert dlq_topic("trading_system.fills.v1") == "trading_system.fills.v1.dlq.v1"


def test_dlq_topic_rejects_invalid_source_topic() -> None:
    from shared_lib.transport import dlq_topic

    with pytest.raises(ValueError):
        dlq_topic("INVALID-TOPIC")


# ---------------------------------------------------------------------------
# Envelope redacts sensitive payload keys when serialising for logs.
# ---------------------------------------------------------------------------


def test_envelope_payload_redaction_for_logs() -> None:
    from shared_lib.transport import EventEnvelope, redact_payload_for_logging

    env = EventEnvelope(
        event_id="e1",
        trace_id="t",
        span_id="s",
        occurred_at=datetime(2026, 4, 19, tzinfo=UTC),
        topic="data_platform.connectors.v1",
        schema_version=1,
        payload={"symbol": "AAPL", "api_key": "super-sensitive"},
    )
    safe = redact_payload_for_logging(env)
    assert safe["payload"]["api_key"] == "***REDACTED***"
    assert safe["payload"]["symbol"] == "AAPL"
