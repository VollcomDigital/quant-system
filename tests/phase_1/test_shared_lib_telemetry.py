"""Phase 1 Task 2 - shared_lib.telemetry OpenTelemetry bootstrap.

The telemetry module must:

- provide a single `bootstrap` entry point that is called once per
  process/service
- support four distinct workload profiles: batch, api, agent, live
- integrate with `shared_lib.logging`'s trace/span contextvars so JSON logs
  carry the same trace_id that spans use
- work when the `opentelemetry` package is not installed (no-op fallback)
- be idempotent and thread-safe
- expose a `start_span` context manager that binds trace_id/span_id onto
  `shared_lib.logging` automatically
- expose a `record_exception` hook that both tags the active span and
  logs a structured error event
"""

from __future__ import annotations

import io
import json

import pytest

# ---------------------------------------------------------------------------
# Edge case 1: bootstrap accepts only the four documented profiles.
# ---------------------------------------------------------------------------


def test_bootstrap_rejects_unknown_profile() -> None:
    from shared_lib.telemetry import bootstrap

    with pytest.raises(ValueError, match="profile"):
        bootstrap(service="x", profile="bogus")


@pytest.mark.parametrize("profile", ["batch", "api", "agent", "live"])
def test_bootstrap_accepts_each_documented_profile(profile: str) -> None:
    from shared_lib.telemetry import bootstrap

    # None of these should raise even when OTel is not installed.
    tracer = bootstrap(service=f"svc-{profile}", profile=profile)
    assert tracer is not None


# ---------------------------------------------------------------------------
# Edge case 2: bootstrap is idempotent.
# ---------------------------------------------------------------------------


def test_bootstrap_is_idempotent() -> None:
    from shared_lib.telemetry import bootstrap

    a = bootstrap(service="svc-idem", profile="batch")
    b = bootstrap(service="svc-idem", profile="batch")
    assert a is b


# ---------------------------------------------------------------------------
# Edge case 3: when OTel is missing, the fallback tracer still returns a
# span context manager that binds trace/span onto shared_lib.logging so log
# correlation keeps working end-to-end.
# ---------------------------------------------------------------------------


def test_start_span_binds_trace_onto_logger() -> None:
    from shared_lib.logging import configure_logging, log_event
    from shared_lib.telemetry import bootstrap, start_span

    stream = io.StringIO()
    configure_logging(service="svc-span", stream=stream)
    bootstrap(service="svc-span", profile="agent")

    with start_span(service="svc-span", name="plan"):
        log_event(
            configure_logging(service="svc-span", stream=stream),
            "planning",
            stage="agent",
        )
    payload = json.loads(stream.getvalue().strip())
    assert payload["trace_id"]
    assert payload["span_id"]
    # Trace ids should be hex-shaped 16+ hex chars (OTel format) or the
    # shared_lib fallback form (uuid-like). Accept both.
    assert len(payload["trace_id"]) >= 16


# ---------------------------------------------------------------------------
# Edge case 4: nested spans produce unique span_ids and inherit trace_id.
# ---------------------------------------------------------------------------


def test_nested_spans_share_trace_but_differ_by_span() -> None:
    from shared_lib.logging import configure_logging, log_event
    from shared_lib.telemetry import bootstrap, start_span

    stream = io.StringIO()
    configure_logging(service="svc-nest", stream=stream)
    bootstrap(service="svc-nest", profile="live")

    with start_span(service="svc-nest", name="outer"):
        log_event(
            configure_logging(service="svc-nest", stream=stream),
            "outer",
            stage="exec",
        )
        with start_span(service="svc-nest", name="inner"):
            log_event(
                configure_logging(service="svc-nest", stream=stream),
                "inner",
                stage="exec",
            )

    lines = [
        json.loads(ln)
        for ln in stream.getvalue().splitlines()
        if ln.strip()
    ]
    assert len(lines) == 2
    assert lines[0]["trace_id"] == lines[1]["trace_id"]
    assert lines[0]["span_id"] != lines[1]["span_id"]


# ---------------------------------------------------------------------------
# Edge case 5: record_exception emits an ERROR-level log and keeps the
# active trace_id.
# ---------------------------------------------------------------------------


def test_record_exception_logs_error_with_trace() -> None:
    from shared_lib.logging import configure_logging
    from shared_lib.telemetry import bootstrap, record_exception, start_span

    stream = io.StringIO()
    configure_logging(service="svc-err", stream=stream)
    bootstrap(service="svc-err", profile="api")

    with start_span(service="svc-err", name="submit_order"):
        try:
            raise RuntimeError("broker timeout")
        except RuntimeError as exc:
            record_exception(service="svc-err", exc=exc, stage="broker")

    lines = [
        json.loads(ln)
        for ln in stream.getvalue().splitlines()
        if ln.strip()
    ]
    assert any(
        ln["event"] == "exception" and ln["level"] == "ERROR" for ln in lines
    )
    error_line = next(ln for ln in lines if ln["event"] == "exception")
    assert error_line["exception_type"] == "RuntimeError"
    assert "broker timeout" in error_line["exception_message"]
    assert error_line["trace_id"]


# ---------------------------------------------------------------------------
# Edge case 6: per-profile attributes - batch carries `job_type=batch` as
# an attribute on the tracer so downstream exporters can filter.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "profile,expected_attr",
    [
        ("batch", "batch"),
        ("api", "api"),
        ("agent", "agent"),
        ("live", "live"),
    ],
)
def test_profile_attribute_is_set_on_tracer(profile: str, expected_attr: str) -> None:
    from shared_lib.telemetry import bootstrap

    tracer = bootstrap(service=f"svc-{profile}-attr", profile=profile)
    assert tracer.profile == expected_attr


# ---------------------------------------------------------------------------
# Edge case 7: start_span outside a bootstrap still works (auto-bootstrap).
# ---------------------------------------------------------------------------


def test_start_span_auto_bootstraps() -> None:
    from shared_lib.logging import configure_logging
    from shared_lib.telemetry import start_span

    stream = io.StringIO()
    configure_logging(service="svc-auto", stream=stream)
    # No explicit bootstrap call.
    with start_span(service="svc-auto", name="auto"):
        pass  # no error is success
