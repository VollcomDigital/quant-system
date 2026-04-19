"""OpenTelemetry bootstrap for quant-system domain packages.

One entry point per process: `bootstrap(service, profile)`. The four
supported profiles match the Phase 1 roadmap:

- `batch`  - Airflow tasks, backfills, DAGs.
- `api`    - FastAPI services (web_control_plane, data_platform APIs).
- `agent`  - AI agent runs.
- `live`   - OMS/EMS/mid-frequency live trading paths.

The module works with or without the `opentelemetry` package. When OTel is
not installed, a deterministic in-process tracer is used so `trace_id`/`span_id`
still flow through `shared_lib.logging` and tests remain runnable on a
minimal toolchain.
"""

from __future__ import annotations

import secrets
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from shared_lib.logging import bind_trace, configure_logging, log_event

if TYPE_CHECKING:  # pragma: no cover
    pass

__all__ = [
    "SUPPORTED_PROFILES",
    "Tracer",
    "bootstrap",
    "record_exception",
    "start_span",
]


SUPPORTED_PROFILES: frozenset[str] = frozenset({"batch", "api", "agent", "live"})


@dataclass
class Tracer:
    service: str
    profile: str


_REGISTRY: dict[str, Tracer] = {}
_LOCK = threading.Lock()


def bootstrap(*, service: str, profile: str) -> Tracer:
    """Initialise (or return the existing) tracer for `service`.

    Idempotent: repeat calls for the same service return the same Tracer.
    Unknown profiles raise `ValueError`.
    """
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(
            f"Unknown telemetry profile {profile!r}; "
            f"must be one of {sorted(SUPPORTED_PROFILES)}"
        )

    with _LOCK:
        cached = _REGISTRY.get(service)
        if cached is not None:
            return cached
        tracer = Tracer(service=service, profile=profile)
        _REGISTRY[service] = tracer
        return tracer


def _ensure_tracer(service: str) -> Tracer:
    cached = _REGISTRY.get(service)
    if cached is None:
        # Auto-bootstrap with a conservative default profile.
        cached = bootstrap(service=service, profile="batch")
    return cached


def _new_trace_id() -> str:
    return secrets.token_hex(16)


def _new_span_id() -> str:
    return secrets.token_hex(8)


@contextmanager
def start_span(*, service: str, name: str) -> Iterator[str]:
    """Begin a span bound to `shared_lib.logging` trace/span contextvars.

    The span id is always fresh; the trace id is inherited from the current
    logging context if one is active, otherwise a new one is allocated.
    """
    _ensure_tracer(service)
    # Importing here to avoid a circular import and to read the current
    # context value lazily.
    from shared_lib.logging import _trace_id as _current_trace

    current_trace = _current_trace.get()
    trace_id = current_trace or _new_trace_id()
    span_id = _new_span_id()

    # The fallback tracer deliberately does not emit its own log lines;
    # domain callers own log content. A future OTel-backed implementation
    # will emit real span records via the OTel SDK.
    configure_logging(service=service)
    with bind_trace(trace_id=trace_id, span_id=span_id):
        yield span_id


def record_exception(*, service: str, exc: BaseException, stage: str) -> None:
    """Log a structured error event that carries the active trace context."""
    logger = configure_logging(service=service)
    log_event(
        logger,
        "exception",
        stage=stage,
        level="ERROR",
        exception_type=type(exc).__name__,
        exception_message=str(exc),
    )
