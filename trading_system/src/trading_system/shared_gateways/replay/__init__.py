"""Gateway replay + gap-detection + heartbeat helpers.

Phase 7 ships pure functions + a small `HeartbeatTracker`. Real venue
sessions consume these primitives so paper-trading parity stays
deterministic and gateways can be exercised without live broker
connections.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TypeVar

__all__ = ["HeartbeatTracker", "detect_gaps", "replay_sequenced"]


T = TypeVar("T")


def replay_sequenced(  # noqa: UP047 - TypeVar keeps Py 3.11 compat
    messages: Iterable[tuple[int, T]],
) -> Iterator[T]:
    """Yield message payloads in strict ascending-sequence order.

    Refuses out-of-order or duplicate sequence numbers so a corrupted
    replay is fail-closed.
    """
    last: int | None = None
    for seq, payload in messages:
        if last is not None and seq < last:
            raise ValueError(f"out of order: {seq} after {last}")
        if last is not None and seq == last:
            raise ValueError(f"duplicate sequence number: {seq}")
        last = seq
        yield payload


def detect_gaps(seqs: Sequence[int]) -> list[int]:
    """Return the missing sequence numbers in `seqs` (which must be sorted)."""
    if list(seqs) != sorted(seqs):
        raise ValueError("input must be sorted")
    if not seqs:
        return []
    expected = set(range(seqs[0], seqs[-1] + 1))
    return sorted(expected - set(seqs))


@dataclass
class HeartbeatTracker:
    timeout: timedelta
    last_heartbeat: datetime

    def __post_init__(self) -> None:
        if self.last_heartbeat.tzinfo is None:
            raise ValueError("last_heartbeat must be timezone-aware")

    def record(self, *, at: datetime) -> None:
        if at.tzinfo is None:
            raise ValueError("`at` must be timezone-aware")
        self.last_heartbeat = at

    def is_timed_out(self, *, at: datetime) -> bool:
        if at.tzinfo is None:
            raise ValueError("`at` must be timezone-aware")
        return (at - self.last_heartbeat) > self.timeout
