"""Phase 7 Task 5 - shared_gateways.fix_engine + binary_protocols.

Phase 7 ships a minimal pure-Python FIX 4.4-compatible parser + a
binary frame parser, both with the protocol parsing strictly separated
from session/order state logic. Real FIX session implementations
(QuickFIX, native engines) plug in behind the same `FixSession` and
`BinaryFrameParser` Protocols.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# FIX message parsing
# ---------------------------------------------------------------------------


def test_fix_parser_decodes_simple_message() -> None:
    from trading_system.shared_gateways.fix_engine import parse_fix_message

    raw = b"8=FIX.4.4\x019=12\x0135=D\x0149=SENDER\x0156=TARGET\x0110=000\x01"
    msg = parse_fix_message(raw)
    assert msg.tags["8"] == "FIX.4.4"
    assert msg.tags["35"] == "D"
    assert msg.tags["49"] == "SENDER"


def test_fix_parser_rejects_malformed_message() -> None:
    from trading_system.shared_gateways.fix_engine import parse_fix_message

    with pytest.raises(ValueError):
        parse_fix_message(b"not_fix_at_all")


def test_fix_parser_handles_empty_payload() -> None:
    from trading_system.shared_gateways.fix_engine import parse_fix_message

    with pytest.raises(ValueError):
        parse_fix_message(b"")


# ---------------------------------------------------------------------------
# FIX session: parsing is separate from state.
# ---------------------------------------------------------------------------


def test_fix_session_tracks_sequence_numbers() -> None:
    from trading_system.shared_gateways.fix_engine import FixSession

    s = FixSession(sender_comp_id="ME", target_comp_id="THEM")
    assert s.next_outbound_seq() == 1
    assert s.next_outbound_seq() == 2


def test_fix_session_resets_on_logon_response() -> None:
    from trading_system.shared_gateways.fix_engine import FixSession

    s = FixSession(sender_comp_id="ME", target_comp_id="THEM")
    s.next_outbound_seq()
    s.next_outbound_seq()
    s.reset()
    assert s.next_outbound_seq() == 1


def test_fix_session_marks_logged_in() -> None:
    from trading_system.shared_gateways.fix_engine import FixSession

    s = FixSession(sender_comp_id="ME", target_comp_id="THEM")
    assert s.logged_in is False
    s.mark_logged_in()
    assert s.logged_in is True


# ---------------------------------------------------------------------------
# Binary frame parser: length-prefixed, parsing is separate from state.
# ---------------------------------------------------------------------------


def test_binary_frame_parser_returns_frame_payload() -> None:
    from trading_system.shared_gateways.binary_protocols import parse_binary_frame

    # 4-byte big-endian length + payload bytes.
    payload = b"hello"
    frame = (len(payload)).to_bytes(4, "big") + payload
    decoded, remaining = parse_binary_frame(frame)
    assert decoded == payload
    assert remaining == b""


def test_binary_frame_parser_returns_remaining_bytes() -> None:
    from trading_system.shared_gateways.binary_protocols import parse_binary_frame

    payload = b"abc"
    frame = (len(payload)).to_bytes(4, "big") + payload + b"trailing"
    decoded, remaining = parse_binary_frame(frame)
    assert decoded == payload
    assert remaining == b"trailing"


def test_binary_frame_parser_rejects_truncated_header() -> None:
    from trading_system.shared_gateways.binary_protocols import parse_binary_frame

    with pytest.raises(ValueError):
        parse_binary_frame(b"\x00\x01")


def test_binary_frame_parser_rejects_truncated_payload() -> None:
    from trading_system.shared_gateways.binary_protocols import parse_binary_frame

    # Length says 10 but only 3 bytes follow.
    with pytest.raises(ValueError):
        parse_binary_frame((10).to_bytes(4, "big") + b"abc")


def test_binary_frame_parser_rejects_zero_length() -> None:
    from trading_system.shared_gateways.binary_protocols import parse_binary_frame

    with pytest.raises(ValueError):
        parse_binary_frame((0).to_bytes(4, "big"))
