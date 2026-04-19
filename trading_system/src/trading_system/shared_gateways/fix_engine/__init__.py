"""Minimal FIX 4.4-compatible parser + session state.

Phase 7 ships parsing only - real FIX session implementations
(QuickFIX, native engines) plug in behind the same `FixSession` shape.
The protocol parser is deliberately separated from the order-state
logic so the tests can exercise either independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["FixMessage", "FixSession", "parse_fix_message"]


_SOH = b"\x01"


@dataclass(frozen=True, slots=True)
class FixMessage:
    tags: dict[str, str]


def parse_fix_message(raw: bytes) -> FixMessage:
    if not raw:
        raise ValueError("empty FIX payload")
    if not raw.startswith(b"8="):
        raise ValueError("FIX message must begin with 8=...")
    fields = [f for f in raw.split(_SOH) if f]
    tags: dict[str, str] = {}
    for f in fields:
        if b"=" not in f:
            raise ValueError(f"malformed FIX field: {f!r}")
        tag, value = f.split(b"=", 1)
        tags[tag.decode("ascii")] = value.decode("ascii", errors="replace")
    return FixMessage(tags=tags)


@dataclass
class FixSession:
    sender_comp_id: str
    target_comp_id: str
    logged_in: bool = False
    _outbound_seq: int = field(default=0, init=False, repr=False)

    def next_outbound_seq(self) -> int:
        self._outbound_seq += 1
        return self._outbound_seq

    def reset(self) -> None:
        self._outbound_seq = 0
        self.logged_in = False

    def mark_logged_in(self) -> None:
        self.logged_in = True
