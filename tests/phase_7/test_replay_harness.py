"""Phase 7 Task 6 - replay + gap + sequence-recovery harnesses.

These verify the gateway-side recovery semantics that real venue
sessions need:

- Sequenced messages can be replayed deterministically.
- Gaps trigger a structured `GapDetected` event.
- Heartbeat tracking surfaces a `HeartbeatTimeout` after the configured
  silence window.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

# ---------------------------------------------------------------------------
# Sequenced replay
# ---------------------------------------------------------------------------


def test_replay_in_order_messages_yields_each_once() -> None:
    from trading_system.shared_gateways.replay import replay_sequenced

    messages = [(1, "a"), (2, "b"), (3, "c")]
    out = list(replay_sequenced(messages))
    assert out == ["a", "b", "c"]


def test_replay_rejects_out_of_order_input() -> None:
    from trading_system.shared_gateways.replay import replay_sequenced

    messages = [(1, "a"), (3, "c"), (2, "b")]
    with pytest.raises(ValueError, match="order"):
        list(replay_sequenced(messages))


def test_replay_rejects_duplicate_sequence_number() -> None:
    from trading_system.shared_gateways.replay import replay_sequenced

    with pytest.raises(ValueError, match="duplicate"):
        list(replay_sequenced([(1, "a"), (1, "b")]))


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------


def test_gap_detector_returns_missing_sequence_numbers() -> None:
    from trading_system.shared_gateways.replay import detect_gaps

    seqs = [1, 2, 4, 5, 8]
    assert detect_gaps(seqs) == [3, 6, 7]


def test_gap_detector_returns_empty_when_contiguous() -> None:
    from trading_system.shared_gateways.replay import detect_gaps

    assert detect_gaps([10, 11, 12, 13]) == []


def test_gap_detector_rejects_unsorted_input() -> None:
    from trading_system.shared_gateways.replay import detect_gaps

    with pytest.raises(ValueError, match="sorted"):
        detect_gaps([3, 1, 2])


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


def test_heartbeat_timeout_fires_after_window() -> None:
    from trading_system.shared_gateways.replay import HeartbeatTracker

    now = datetime(2026, 4, 19, 14, tzinfo=UTC)
    hb = HeartbeatTracker(timeout=timedelta(seconds=30), last_heartbeat=now)
    assert hb.is_timed_out(at=now + timedelta(seconds=29)) is False
    assert hb.is_timed_out(at=now + timedelta(seconds=31)) is True


def test_heartbeat_record_resets_clock() -> None:
    from trading_system.shared_gateways.replay import HeartbeatTracker

    now = datetime(2026, 4, 19, 14, tzinfo=UTC)
    hb = HeartbeatTracker(timeout=timedelta(seconds=10), last_heartbeat=now)
    hb.record(at=now + timedelta(seconds=20))  # explicit re-anchor
    assert hb.is_timed_out(at=now + timedelta(seconds=29)) is False
    assert hb.is_timed_out(at=now + timedelta(seconds=31)) is True


def test_heartbeat_rejects_naive_timestamp() -> None:
    from trading_system.shared_gateways.replay import HeartbeatTracker

    with pytest.raises(ValueError):
        HeartbeatTracker(
            timeout=timedelta(seconds=10),
            last_heartbeat=datetime(2026, 4, 19, 14),
        )
