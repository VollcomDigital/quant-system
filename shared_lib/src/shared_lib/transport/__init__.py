"""Envelopes for RPC and event traffic.

- `RpcEnvelope`   - sync request/response. Idempotency key + deadline.
- `EventEnvelope` - async fact. Event id + occurred_at + topic + schema_version.

These envelopes are the stable contract between Python research services
and Rust/C++ execution services. Topic names follow the convention
defined in `docs/architecture/service-communication-standards.md`
(lowercase, dot-separated, owning-domain prefix, trailing version).
"""

from __future__ import annotations

import re
import secrets
from datetime import UTC, datetime
from typing import Any

from pydantic import Field

from shared_lib.contracts._base import Schema, aware_datetime_validator

__all__ = [
    "EventEnvelope",
    "RpcEnvelope",
    "dlq_topic",
    "new_event_envelope",
    "new_rpc_envelope",
    "redact_payload_for_logging",
]


TOPIC_PATTERN = re.compile(
    r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+\.v[0-9]+$"
)


_REDACT_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "secret",
        "password",
        "passwd",
        "private_key",
        "token",
        "access_token",
        "refresh_token",
    }
)
_REDACTED = "***REDACTED***"


class RpcEnvelope(Schema):
    idempotency_key: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    span_id: str = Field(min_length=1)
    deadline: datetime
    payload_schema: str = Field(min_length=1)
    payload: dict[str, Any]

    _ts = aware_datetime_validator("deadline")


class EventEnvelope(Schema):
    event_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    span_id: str = Field(min_length=1)
    occurred_at: datetime
    topic: str
    schema_version: int = Field(gt=0)
    payload: dict[str, Any]

    _ts = aware_datetime_validator("occurred_at")

    @staticmethod
    def _validate_topic(topic: str) -> str:
        if not TOPIC_PATTERN.match(topic):
            raise ValueError(
                f"topic {topic!r} must match "
                "lowercase.dotted.with_version (e.g. 'trading_system.fills.v1')"
            )
        return topic

    def __init__(self, **data: Any) -> None:
        if "topic" in data:
            data["topic"] = self._validate_topic(data["topic"])
        super().__init__(**data)


def new_event_envelope(
    *,
    topic: str,
    schema_version: int,
    payload: dict[str, Any],
    trace_id: str | None = None,
    span_id: str | None = None,
) -> EventEnvelope:
    """Construct an `EventEnvelope`, auto-picking ids from the log context."""
    from shared_lib.logging import _span_id as current_span
    from shared_lib.logging import _trace_id as current_trace

    return EventEnvelope(
        event_id=secrets.token_hex(16),
        trace_id=trace_id or current_trace.get() or secrets.token_hex(16),
        span_id=span_id or current_span.get() or secrets.token_hex(8),
        occurred_at=datetime.now(tz=UTC),
        topic=topic,
        schema_version=schema_version,
        payload=payload,
    )


def new_rpc_envelope(
    *,
    payload_schema: str,
    payload: dict[str, Any],
    deadline: datetime,
    idempotency_key: str | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
) -> RpcEnvelope:
    """Construct an `RpcEnvelope`.

    Rejects past deadlines. Picks up trace/span from the current logging
    context when not given explicitly.
    """
    if deadline.tzinfo is None:
        raise ValueError("deadline must be timezone-aware")
    if deadline <= datetime.now(tz=UTC):
        raise ValueError("deadline is in the past; refusing to create envelope")

    from shared_lib.logging import _span_id as current_span
    from shared_lib.logging import _trace_id as current_trace

    return RpcEnvelope(
        idempotency_key=idempotency_key or secrets.token_hex(16),
        trace_id=trace_id or current_trace.get() or secrets.token_hex(16),
        span_id=span_id or current_span.get() or secrets.token_hex(8),
        deadline=deadline,
        payload_schema=payload_schema,
        payload=payload,
    )


def dlq_topic(source_topic: str) -> str:
    """Return the canonical dead-letter topic for `source_topic`."""
    if not TOPIC_PATTERN.match(source_topic):
        raise ValueError(f"source topic {source_topic!r} is not a valid topic name")
    return f"{source_topic}.dlq.v1"


def redact_payload_for_logging(env: EventEnvelope | RpcEnvelope) -> dict[str, Any]:
    """Return a dict suitable for logging, with secrets redacted."""
    base = env.model_dump()
    safe_payload: dict[str, Any] = {}
    for k, v in base.get("payload", {}).items():
        safe_payload[k] = _REDACTED if k.lower() in _REDACT_KEYS else v
    base["payload"] = safe_payload
    return base
