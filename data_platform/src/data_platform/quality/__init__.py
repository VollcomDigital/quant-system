"""Data-quality primitives.

Every check returns a `shared_lib.contracts.ValidationResult` so the
same plumbing (factor-promotion gate, control-plane alerts, agent risk
monitor) can consume them uniformly.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

from shared_lib.contracts import Bar, ValidationResult

__all__ = [
    "check_continuity",
    "check_freshness",
    "check_no_duplicates",
    "check_schema",
    "check_survivorship",
]


def _result(check_id: str, target: str, passed: bool, reason: str | None) -> ValidationResult:
    now = datetime.now(tz=_UTC)
    return ValidationResult(
        check_id=check_id,
        target=target,
        passed=passed,
        reason=reason,
        evaluated_at=now,
    )


from datetime import UTC as _UTC  # noqa: E402


def check_schema(bars: Sequence[Bar]) -> ValidationResult:
    target = "bars"
    if not bars:
        return _result("schema", target, True, None)
    symbols = {b.symbol for b in bars}
    if len(symbols) > 1:
        return _result(
            "schema",
            target,
            False,
            f"mixed symbol set: {sorted(symbols)!r}",
        )
    intervals = {b.interval for b in bars}
    if len(intervals) > 1:
        return _result(
            "schema",
            target,
            False,
            f"mixed interval set: {sorted(intervals)!r}",
        )
    return _result("schema", target, True, None)


def check_continuity(
    bars: Sequence[Bar], *, expected_step: timedelta
) -> ValidationResult:
    target = "bars"
    if len(bars) < 2:
        return _result("continuity", target, True, None)
    ordered = sorted(bars, key=lambda b: b.timestamp)
    gaps: list[str] = []
    for a, b in zip(ordered, ordered[1:], strict=False):
        delta = b.timestamp - a.timestamp
        if delta != expected_step:
            gaps.append(f"{a.timestamp.isoformat()}->{b.timestamp.isoformat()} ({delta})")
    if gaps:
        return _result("continuity", target, False, f"gaps: {gaps}")
    return _result("continuity", target, True, None)


def check_no_duplicates(bars: Sequence[Bar]) -> ValidationResult:
    target = "bars"
    seen: set[datetime] = set()
    dupes: list[datetime] = []
    for b in bars:
        if b.timestamp in seen:
            dupes.append(b.timestamp)
        else:
            seen.add(b.timestamp)
    if dupes:
        return _result(
            "no_duplicates",
            target,
            False,
            f"duplicate timestamps: {sorted(dupes)}",
        )
    return _result("no_duplicates", target, True, None)


def check_freshness(
    bars: Sequence[Bar], *, now: datetime, max_lag: timedelta
) -> ValidationResult:
    if not bars:
        raise ValueError("check_freshness requires at least one bar")
    target = "bars"
    latest = max(b.timestamp for b in bars)
    lag = now - latest
    if lag > max_lag:
        return _result(
            "freshness",
            target,
            False,
            f"latest bar is {lag} old; max allowed {max_lag}",
        )
    return _result("freshness", target, True, None)


def check_survivorship(*, got: set[str], expected: set[str]) -> ValidationResult:
    target = "universe"
    missing = sorted(expected - got)
    if missing:
        return _result(
            "survivorship",
            target,
            False,
            f"missing symbols: {missing}",
        )
    return _result("survivorship", target, True, None)
