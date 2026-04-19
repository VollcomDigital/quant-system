"""Phase 1 Task 1 - shared_lib.logging structured JSON invariants.

The new logger is the single source of log output for every domain package.
It must:

- always emit parseable JSON
- always include the required contract fields
  (trace_id, span_id, service, module, stage, event, level, timestamp)
- propagate trace/span context via contextvars so downstream modules and
  agents can bind to an existing trace
- never duplicate handlers across repeat calls (idempotent config)
- redact known-sensitive keys before serialisation
- encode Decimal and datetime values deterministically
- fall back to a NullHandler when no stream is configured
"""

from __future__ import annotations

import io
import json
import logging
from datetime import UTC, datetime
from decimal import Decimal

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stream() -> io.StringIO:
    return io.StringIO()


@pytest.fixture(autouse=True)
def _clean_logger_state():
    # Reset root + shared_lib loggers between tests so idempotency checks are
    # meaningful.
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith("shared_lib") or name == "quant":
            logging.getLogger(name).handlers.clear()
    yield


# ---------------------------------------------------------------------------
# Edge case 1: every emitted line must be valid JSON and carry the contract
# fields.
# ---------------------------------------------------------------------------


REQUIRED_FIELDS = {
    "event",
    "level",
    "service",
    "module",
    "stage",
    "trace_id",
    "span_id",
    "timestamp",
}


def test_log_event_emits_json_with_required_fields(stream: io.StringIO) -> None:
    from shared_lib.logging import configure_logging, log_event

    logger = configure_logging(service="backtest_engine", stream=stream)
    log_event(logger, "run_started", stage="bootstrap", run_id="r-1")

    line = stream.getvalue().strip()
    payload = json.loads(line)
    assert REQUIRED_FIELDS.issubset(payload.keys()), (
        f"Missing required fields: {REQUIRED_FIELDS - set(payload.keys())}"
    )
    assert payload["event"] == "run_started"
    assert payload["stage"] == "bootstrap"
    assert payload["service"] == "backtest_engine"
    assert payload["run_id"] == "r-1"
    assert payload["level"] == "INFO"


# ---------------------------------------------------------------------------
# Edge case 2: `configure_logging` must be idempotent - calling it multiple
# times must not duplicate handlers.
# ---------------------------------------------------------------------------


def test_configure_logging_is_idempotent(stream: io.StringIO) -> None:
    from shared_lib.logging import configure_logging, log_event

    logger_a = configure_logging(service="shared_lib", stream=stream)
    logger_b = configure_logging(service="shared_lib", stream=stream)

    assert logger_a is logger_b
    assert len(logger_a.handlers) == 1

    log_event(logger_a, "hello", stage="ping")
    # A duplicated handler would yield two JSON lines.
    lines = [ln for ln in stream.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 1


# ---------------------------------------------------------------------------
# Edge case 3: trace_id/span_id must propagate through contextvars so nested
# calls inherit the current trace.
# ---------------------------------------------------------------------------


def test_trace_context_propagates_via_contextvars(stream: io.StringIO) -> None:
    from shared_lib.logging import bind_trace, configure_logging, log_event

    logger = configure_logging(service="ai_agents", stream=stream)
    with bind_trace(trace_id="trace-abc", span_id="span-123"):
        log_event(logger, "agent_started", stage="plan")

    payload = json.loads(stream.getvalue().strip())
    assert payload["trace_id"] == "trace-abc"
    assert payload["span_id"] == "span-123"


def test_trace_context_is_cleared_after_exit(stream: io.StringIO) -> None:
    from shared_lib.logging import bind_trace, configure_logging, log_event

    logger = configure_logging(service="ai_agents", stream=stream)
    with bind_trace(trace_id="t1", span_id="s1"):
        pass
    log_event(logger, "no_trace", stage="idle")
    payload = json.loads(stream.getvalue().strip())
    # After the context exits, trace/span must either be absent or explicit
    # null - never the previous ids.
    assert payload.get("trace_id") in (None, "", "none")
    assert payload.get("span_id") in (None, "", "none")


# ---------------------------------------------------------------------------
# Edge case 4: redaction - keys named in the secret set must never appear
# as plaintext in emitted JSON.
# ---------------------------------------------------------------------------


SECRET_KEYS = ("api_key", "secret", "password", "private_key", "token")


@pytest.mark.parametrize("key", SECRET_KEYS)
def test_secret_keys_are_redacted(stream: io.StringIO, key: str) -> None:
    from shared_lib.logging import configure_logging, log_event

    logger = configure_logging(service="shared_lib", stream=stream)
    log_event(logger, "boot", stage="init", **{key: "super-sensitive"})

    raw = stream.getvalue()
    assert "super-sensitive" not in raw, f"Secret value leaked via key {key}"
    payload = json.loads(raw.strip())
    assert payload[key] == "***REDACTED***"


# ---------------------------------------------------------------------------
# Edge case 5: deterministic serialisation of Decimal and datetime values -
# no float-style precision loss, no unserialisable errors.
# ---------------------------------------------------------------------------


def test_decimal_is_serialised_as_string(stream: io.StringIO) -> None:
    from shared_lib.logging import configure_logging, log_event

    logger = configure_logging(service="shared_lib", stream=stream)
    log_event(logger, "price", stage="tick", notional=Decimal("1.2345678901234567890"))

    payload = json.loads(stream.getvalue().strip())
    assert payload["notional"] == "1.2345678901234567890"


def test_datetime_is_serialised_as_iso8601(stream: io.StringIO) -> None:
    from shared_lib.logging import configure_logging, log_event

    logger = configure_logging(service="shared_lib", stream=stream)
    when = datetime(2026, 4, 19, 13, 37, 42, tzinfo=UTC)
    log_event(logger, "event", stage="test", when=when)
    payload = json.loads(stream.getvalue().strip())
    assert payload["when"] == "2026-04-19T13:37:42+00:00"


# ---------------------------------------------------------------------------
# Edge case 6: level escalation - log_event exposes level-aware emitters.
# ---------------------------------------------------------------------------


def test_log_event_respects_level(stream: io.StringIO) -> None:
    from shared_lib.logging import configure_logging, log_event

    logger = configure_logging(service="shared_lib", stream=stream, level="INFO")
    log_event(logger, "critical_event", stage="rms", level="ERROR", reason="breach")
    payload = json.loads(stream.getvalue().strip())
    assert payload["level"] == "ERROR"


# ---------------------------------------------------------------------------
# Edge case 7: missing service is rejected - no global default service name
# is allowed to leak through.
# ---------------------------------------------------------------------------


def test_configure_logging_requires_service_name(stream: io.StringIO) -> None:
    from shared_lib.logging import configure_logging

    with pytest.raises(ValueError, match="service"):
        configure_logging(service="", stream=stream)


# ---------------------------------------------------------------------------
# Edge case 8: concurrent trace contexts - nested bind_trace blocks must
# stack, and unwinding restores the outer id.
# ---------------------------------------------------------------------------


def test_nested_trace_contexts_stack(stream: io.StringIO) -> None:
    from shared_lib.logging import bind_trace, configure_logging, log_event

    logger = configure_logging(service="shared_lib", stream=stream)
    with bind_trace(trace_id="outer", span_id="o1"):
        log_event(logger, "outer", stage="test")
        with bind_trace(trace_id="inner", span_id="i1"):
            log_event(logger, "inner", stage="test")
        log_event(logger, "after_inner", stage="test")
    lines = [json.loads(ln) for ln in stream.getvalue().splitlines() if ln.strip()]
    assert lines[0]["trace_id"] == "outer"
    assert lines[1]["trace_id"] == "inner"
    assert lines[2]["trace_id"] == "outer"


# ---------------------------------------------------------------------------
# Edge case 9: time_block must log start/end durations with the same trace.
# ---------------------------------------------------------------------------


def test_time_block_emits_duration(stream: io.StringIO) -> None:
    from shared_lib.logging import configure_logging, time_block

    logger = configure_logging(service="shared_lib", stream=stream)
    with time_block(logger, "load_bars", stage="ingest", symbol="AAPL"):
        pass
    lines = [json.loads(ln) for ln in stream.getvalue().splitlines() if ln.strip()]
    assert any(ln["event"] == "load_bars" and "duration_sec" in ln for ln in lines)


# ---------------------------------------------------------------------------
# Edge case 10: when no stream is given, logger falls back to NullHandler
# so import-only usage never raises.
# ---------------------------------------------------------------------------


def test_no_stream_falls_back_to_null_handler() -> None:
    from shared_lib.logging import configure_logging

    logger = configure_logging(service="shared_lib")
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)
