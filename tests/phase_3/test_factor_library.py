"""Phase 3 Task 1 - alpha_research.factor_library invariants.

A `Factor` is the production-promoted successor to a notebook draft. It
must carry the full metadata contract:

- description
- source_dependencies
- stationarity_assumption
- universe_coverage
- leakage_review
- validation_status (reused from Phase 2 FactorDefinition)

Factors are computed over a bar history and produce FactorRecords. The
library wraps a registry + a compute API; callers cannot publish bars
that lack a matching definition.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest


def _bar(ts: datetime, symbol: str = "AAPL", close: str = "100"):
    from shared_lib.contracts import Bar

    return Bar(
        symbol=symbol,
        interval="1d",
        timestamp=ts,
        open=Decimal(close),
        high=Decimal(close),
        low=Decimal(close),
        close=Decimal(close),
        volume=Decimal("0"),
    )


# ---------------------------------------------------------------------------
# FactorMetadata
# ---------------------------------------------------------------------------


def test_factor_metadata_basic() -> None:
    from alpha_research.factor_library import FactorMetadata

    m = FactorMetadata(
        factor_id="mom_12_1",
        version="v1",
        description="12-1 momentum",
        source_dependencies=("bars",),
        stationarity_assumption="mean-reverting monthly",
        universe_coverage="us_equities",
        leakage_review="reviewed: uses only t-1 bar",
        validation_status="candidate",
    )
    assert m.factor_id == "mom_12_1"


def test_factor_metadata_rejects_empty_leakage_review() -> None:
    from alpha_research.factor_library import FactorMetadata

    with pytest.raises(ValueError, match="leakage"):
        FactorMetadata(
            factor_id="x",
            version="v1",
            description="d",
            source_dependencies=(),
            stationarity_assumption="s",
            universe_coverage="u",
            leakage_review="",  # empty string -> rejected
            validation_status="candidate",
        )


def test_factor_metadata_status_must_be_valid() -> None:
    from alpha_research.factor_library import FactorMetadata

    with pytest.raises(ValueError):
        FactorMetadata(
            factor_id="x",
            version="v1",
            description="d",
            source_dependencies=(),
            stationarity_assumption="s",
            universe_coverage="u",
            leakage_review="ok",
            validation_status="lgtm",  # not in enum
        )


# ---------------------------------------------------------------------------
# Factor compute protocol
# ---------------------------------------------------------------------------


def test_factor_compute_yields_factor_records() -> None:
    from alpha_research.factor_library import Factor, FactorMetadata
    from shared_lib.contracts import FactorRecord

    class _MeanCloseFactor(Factor):
        metadata = FactorMetadata(
            factor_id="mean_close",
            version="v1",
            description="average close so far",
            source_dependencies=("bars",),
            stationarity_assumption="non-stationary",
            universe_coverage="us_equities",
            leakage_review="uses only <= t closes",
            validation_status="candidate",
        )

        def compute(self, bars):
            total = Decimal("0")
            count = 0
            for b in bars:
                total += b.close
                count += 1
                yield FactorRecord(
                    factor_id=self.metadata.factor_id,
                    version=self.metadata.version,
                    symbol=b.symbol,
                    as_of=b.timestamp,
                    value=total / count,
                )

    start = datetime(2026, 4, 1, tzinfo=UTC)
    bars = [_bar(start + timedelta(days=i), close=str(100 + i)) for i in range(3)]
    rows = list(_MeanCloseFactor().compute(bars))
    assert len(rows) == 3
    assert rows[-1].value == Decimal("101")


# ---------------------------------------------------------------------------
# FactorLibrary registry integration
# ---------------------------------------------------------------------------


def test_factor_library_registers_and_compiles_to_feature_store_defn() -> None:
    from alpha_research.factor_library import Factor, FactorLibrary, FactorMetadata
    from data_platform.feature_store import FactorRegistry

    class _F(Factor):
        metadata = FactorMetadata(
            factor_id="f1",
            version="v1",
            description="d",
            source_dependencies=("bars",),
            stationarity_assumption="s",
            universe_coverage="u",
            leakage_review="ok",
            validation_status="candidate",
        )

        def compute(self, bars):  # pragma: no cover - not exercised here
            yield from ()

    lib = FactorLibrary()
    lib.register(_F())
    registry = FactorRegistry()
    lib.export_to(registry)
    assert registry.has("f1", "v1")


def test_factor_library_rejects_duplicate_registration() -> None:
    from alpha_research.factor_library import Factor, FactorLibrary, FactorMetadata

    class _F(Factor):
        metadata = FactorMetadata(
            factor_id="f1",
            version="v1",
            description="d",
            source_dependencies=(),
            stationarity_assumption="s",
            universe_coverage="u",
            leakage_review="ok",
            validation_status="candidate",
        )

        def compute(self, bars):  # pragma: no cover
            yield from ()

    lib = FactorLibrary()
    lib.register(_F())
    with pytest.raises(ValueError, match="already"):
        lib.register(_F())


def test_factor_library_lookup_by_id_and_version() -> None:
    from alpha_research.factor_library import Factor, FactorLibrary, FactorMetadata

    class _F(Factor):
        metadata = FactorMetadata(
            factor_id="f1",
            version="v2",
            description="d",
            source_dependencies=(),
            stationarity_assumption="s",
            universe_coverage="u",
            leakage_review="ok",
            validation_status="candidate",
        )

        def compute(self, bars):  # pragma: no cover
            yield from ()

    lib = FactorLibrary()
    f = _F()
    lib.register(f)
    assert lib.get("f1", "v2") is f
    with pytest.raises(LookupError):
        lib.get("f1", "vX")


# ---------------------------------------------------------------------------
# Promotion status
# ---------------------------------------------------------------------------


def test_factor_metadata_allows_full_promotion_chain() -> None:
    from alpha_research.factor_library import FactorMetadata

    for status in ("candidate", "validated", "promoted", "retired"):
        FactorMetadata(
            factor_id="x",
            version="v1",
            description="d",
            source_dependencies=(),
            stationarity_assumption="s",
            universe_coverage="u",
            leakage_review="ok",
            validation_status=status,  # type: ignore[arg-type]
        )
