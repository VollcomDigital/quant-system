"""Binary frame parser scaffolds.

Phase 7 ships a length-prefixed frame decoder. Native venue parsers
(ITCH/OUCH, exchange-specific binary) live in
`trading_system/native/hft_engine/network/` and consume the same
"frame -> payload" boundary.
"""

from __future__ import annotations

__all__ = ["parse_binary_frame"]


_HEADER_BYTES = 4


def parse_binary_frame(buffer: bytes) -> tuple[bytes, bytes]:
    """Decode one length-prefixed frame.

    Returns `(payload, remaining)` where `remaining` is whatever bytes
    follow the framed payload. Raises `ValueError` on truncated header,
    truncated payload, or zero-length frame.
    """
    if len(buffer) < _HEADER_BYTES:
        raise ValueError(
            f"truncated frame header: {len(buffer)} < {_HEADER_BYTES}"
        )
    length = int.from_bytes(buffer[:_HEADER_BYTES], "big")
    if length <= 0:
        raise ValueError(f"frame length must be > 0; got {length}")
    end = _HEADER_BYTES + length
    if len(buffer) < end:
        raise ValueError(
            f"truncated frame payload: have {len(buffer)} bytes, need {end}"
        )
    return buffer[_HEADER_BYTES:end], buffer[end:]
