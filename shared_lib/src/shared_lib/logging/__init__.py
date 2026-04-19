"""Structured JSON logging primitives for every quant-system domain.

The contract is deliberately small:

- `configure_logging(service, stream=None, level="INFO")` - idempotent,
  returns a `logging.Logger` that emits one JSON object per line carrying
  at least: `event`, `level`, `service`, `module`, `stage`, `trace_id`,
  `span_id`, `timestamp` (ISO 8601 UTC).
- `log_event(logger, event, stage, level="INFO", **fields)` - writes one
  structured record.
- `bind_trace(trace_id, span_id)` - context manager that pushes a
  trace/span pair onto contextvars so nested calls inherit the trace.
- `time_block(logger, event, stage, **fields)` - context manager that
  emits the event with a `duration_sec` field on exit.

Secrets are redacted by key name. Decimals serialise as strings; datetimes
serialise as ISO-8601.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from decimal import Decimal
from typing import IO, Any

__all__ = [
    "JsonLineFormatter",
    "bind_trace",
    "configure_logging",
    "log_event",
    "time_block",
]


_REDACT_KEYS: frozenset[str] = frozenset(
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

_trace_id: ContextVar[str | None] = ContextVar("_trace_id", default=None)
_span_id: ContextVar[str | None] = ContextVar("_span_id", default=None)


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat()
    return str(value)


def _redact(fields: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in fields.items():
        out[k] = _REDACTED if k.lower() in _REDACT_KEYS else v
    return out


class JsonLineFormatter(logging.Formatter):
    """Formats LogRecords as single-line JSON with the Phase 1 contract."""

    def __init__(self, service: str) -> None:
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        extra: dict[str, Any] = getattr(record, "_structured", {}) or {}
        payload: dict[str, Any] = {
            "event": extra.get("event", record.getMessage()),
            "level": extra.get("level", record.levelname),
            "service": self._service,
            "module": extra.get("module", record.name),
            "stage": extra.get("stage", "unspecified"),
            "trace_id": _trace_id.get(),
            "span_id": _span_id.get(),
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
        }
        for k, v in extra.items():
            if k in {"event", "level", "module", "stage"}:
                continue
            payload[k] = v
        return json.dumps(payload, default=_json_default)


def _get_cached_logger(service: str) -> logging.Logger | None:
    logger = logging.getLogger(f"shared_lib.{service}")
    return logger if logger.handlers else None


def configure_logging(
    service: str,
    *,
    stream: IO[str] | None = None,
    level: str = "INFO",
) -> logging.Logger:
    """Return the canonical domain logger.

    - `service` must be non-empty (enforces per-service attribution).
    - `stream=None` installs a `NullHandler` so import-only usage is safe.
    - Repeat calls with the same `service` return the same logger and do
      not add additional handlers.
    """
    if not service:
        raise ValueError("configure_logging requires a non-empty `service` name")

    cached = _get_cached_logger(service)
    if cached is not None:
        return cached

    logger = logging.getLogger(f"shared_lib.{service}")
    logger.propagate = False

    if stream is None:
        handler: logging.Handler = logging.NullHandler()
    else:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonLineFormatter(service=service))

    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    stage: str,
    level: str = "INFO",
    **fields: Any,
) -> None:
    """Emit a single structured record."""
    safe = _redact(fields)
    extra = {
        "event": event,
        "level": level.upper(),
        "module": logger.name,
        "stage": stage,
        **safe,
    }
    numeric = getattr(logging, level.upper(), logging.INFO)
    logger.log(numeric, event, extra={"_structured": extra})


@contextmanager
def bind_trace(*, trace_id: str, span_id: str) -> Iterator[None]:
    """Push a trace_id/span_id pair onto contextvars for the block's lifetime."""
    t_token = _trace_id.set(trace_id)
    s_token = _span_id.set(span_id)
    try:
        yield
    finally:
        _trace_id.reset(t_token)
        _span_id.reset(s_token)


@contextmanager
def time_block(
    logger: logging.Logger,
    event: str,
    *,
    stage: str,
    **fields: Any,
) -> Iterator[None]:
    """Emit `event` with a `duration_sec` field on exit."""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = round(time.perf_counter() - start, 6)
        log_event(logger, event, stage=stage, duration_sec=duration, **fields)
