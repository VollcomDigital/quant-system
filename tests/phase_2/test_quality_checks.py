"""Phase 2 Task 6 - data_platform.quality checks.

Five data-quality primitives. Each must return a
`shared_lib.contracts.ValidationResult` so downstream consumers (web
control plane, risk monitor agent, factor-promotion gate) integrate
uniformly.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest


def _bar(ts: datetime, symbol: str = "AAPL", *,
         o: str = "1", h: str = "2", lo: str = "0.5", c: str = "1.5", v: str = "100"):
    from shared_lib.contracts import Bar

    return Bar(
        symbol=symbol,
        interval="1d",
        timestamp=ts,
        open=Decimal(o),
        high=Decimal(h),
        low=Decimal(lo),
        close=Decimal(c),
        volume=Decimal(v),
    )


# ---------------------------------------------------------------------------
# Schema check
# ---------------------------------------------------------------------------


def test_schema_check_pass_on_canonical_bars() -> None:
    from data_platform.quality import check_schema

    bars = [_bar(datetime(2026, 4, 1, tzinfo=UTC) + timedelta(days=i)) for i in range(3)]
    result = check_schema(bars)
    assert result.passed is True


def test_schema_check_fails_on_mixed_symbols() -> None:
    from data_platform.quality import check_schema

    bars = [
        _bar(datetime(2026, 4, 1, tzinfo=UTC), symbol="AAPL"),
        _bar(datetime(2026, 4, 2, tzinfo=UTC), symbol="MSFT"),
    ]
    result = check_schema(bars)
    assert result.passed is False
    assert "symbol" in (result.reason or "")


# ---------------------------------------------------------------------------
# Continuity check
# ---------------------------------------------------------------------------


def test_continuity_check_detects_gap() -> None:
    from data_platform.quality import check_continuity

    bars = [
        _bar(datetime(2026, 4, 1, tzinfo=UTC)),
        _bar(datetime(2026, 4, 2, tzinfo=UTC)),
        _bar(datetime(2026, 4, 5, tzinfo=UTC)),  # 2-day gap
    ]
    result = check_continuity(bars, expected_step=timedelta(days=1))
    assert result.passed is False


def test_continuity_check_accepts_exact_cadence() -> None:
    from data_platform.quality import check_continuity

    bars = [_bar(datetime(2026, 4, 1, tzinfo=UTC) + timedelta(days=i)) for i in range(5)]
    result = check_continuity(bars, expected_step=timedelta(days=1))
    assert result.passed is True


# ---------------------------------------------------------------------------
# Duplicate / missing-bar check
# ---------------------------------------------------------------------------


def test_duplicate_check_detects_repeated_timestamp() -> None:
    from data_platform.quality import check_no_duplicates

    ts = datetime(2026, 4, 1, tzinfo=UTC)
    result = check_no_duplicates([_bar(ts), _bar(ts)])
    assert result.passed is False


def test_duplicate_check_passes_on_unique_timestamps() -> None:
    from data_platform.quality import check_no_duplicates

    bars = [_bar(datetime(2026, 4, 1, tzinfo=UTC) + timedelta(days=i)) for i in range(3)]
    assert check_no_duplicates(bars).passed is True


# ---------------------------------------------------------------------------
# Freshness
# ---------------------------------------------------------------------------


def test_freshness_check_fails_when_stale() -> None:
    from data_platform.quality import check_freshness

    bars = [_bar(datetime(2026, 4, 1, tzinfo=UTC))]
    result = check_freshness(
        bars,
        now=datetime(2026, 4, 19, tzinfo=UTC),
        max_lag=timedelta(days=5),
    )
    assert result.passed is False


def test_freshness_check_passes_when_recent() -> None:
    from data_platform.quality import check_freshness

    bars = [_bar(datetime(2026, 4, 18, tzinfo=UTC))]
    result = check_freshness(
        bars,
        now=datetime(2026, 4, 19, tzinfo=UTC),
        max_lag=timedelta(days=5),
    )
    assert result.passed is True


def test_freshness_check_rejects_empty_input() -> None:
    from data_platform.quality import check_freshness

    with pytest.raises(ValueError):
        check_freshness(
            [],
            now=datetime(2026, 4, 19, tzinfo=UTC),
            max_lag=timedelta(days=5),
        )


# ---------------------------------------------------------------------------
# Survivorship / symbol-mapping
# ---------------------------------------------------------------------------


def test_survivorship_audit_flags_missing_symbols() -> None:
    from data_platform.quality import check_survivorship

    got = {"AAPL", "MSFT"}
    expected = {"AAPL", "MSFT", "DELISTED"}
    result = check_survivorship(got=got, expected=expected)
    assert result.passed is False
    assert "DELISTED" in (result.reason or "")


def test_survivorship_audit_passes_when_complete() -> None:
    from data_platform.quality import check_survivorship

    universe = {"AAPL", "MSFT"}
    assert check_survivorship(got=universe, expected=universe).passed is True
