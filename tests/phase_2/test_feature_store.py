"""Phase 2 Task 4 - data_platform.feature_store invariants.

Feature store as source of truth:

- Factor definitions are versioned (factor_id + version) and immutable
  at the (factor_id, version) pair.
- Every written `FactorRecord` must be tagged with a definition that
  already exists in the registry.
- Reads return records filtered by factor_id + optional version + window.
- Research, backtest, and live all go through the same read surface.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

# ---------------------------------------------------------------------------
# Factor definition registry
# ---------------------------------------------------------------------------


def test_factor_registry_registers_and_retrieves() -> None:
    from data_platform.feature_store import FactorDefinition, FactorRegistry

    reg = FactorRegistry()
    defn = FactorDefinition(
        factor_id="mom_12_1",
        version="v1",
        description="12-1 momentum",
        source_dependencies=("bars",),
        universe="us_equities",
        validation_status="candidate",
    )
    reg.register(defn)
    assert reg.get("mom_12_1", "v1") == defn


def test_factor_registry_rejects_duplicate_version() -> None:
    from data_platform.feature_store import FactorDefinition, FactorRegistry

    reg = FactorRegistry()
    defn = FactorDefinition(
        factor_id="x",
        version="v1",
        description="d",
        source_dependencies=(),
        universe="u",
        validation_status="candidate",
    )
    reg.register(defn)
    with pytest.raises(ValueError, match="already registered"):
        reg.register(defn)


def test_factor_registry_versions_are_independent() -> None:
    from data_platform.feature_store import FactorDefinition, FactorRegistry

    reg = FactorRegistry()
    for v in ("v1", "v2"):
        reg.register(
            FactorDefinition(
                factor_id="x",
                version=v,
                description="d",
                source_dependencies=(),
                universe="u",
                validation_status="candidate",
            )
        )
    assert {d.version for d in reg.list("x")} == {"v1", "v2"}


def test_factor_definition_rejects_invalid_status() -> None:
    from data_platform.feature_store import FactorDefinition

    with pytest.raises(ValueError):
        FactorDefinition(
            factor_id="x",
            version="v1",
            description="d",
            source_dependencies=(),
            universe="u",
            validation_status="weird",
        )


# ---------------------------------------------------------------------------
# FeatureStore write/read
# ---------------------------------------------------------------------------


def _registry_with(factor_id: str = "mom_12_1", version: str = "v1"):
    from data_platform.feature_store import FactorDefinition, FactorRegistry

    reg = FactorRegistry()
    reg.register(
        FactorDefinition(
            factor_id=factor_id,
            version=version,
            description="d",
            source_dependencies=(),
            universe="u",
            validation_status="candidate",
        )
    )
    return reg


def _sample_record(factor_id: str = "mom_12_1", version: str = "v1",
                   offset_days: int = 0, symbol: str = "AAPL"):
    from shared_lib.contracts import FactorRecord

    return FactorRecord(
        factor_id=factor_id,
        as_of=datetime(2026, 4, 1, tzinfo=UTC) + timedelta(days=offset_days),
        symbol=symbol,
        value=Decimal("0.01"),
        version=version,
    )


def test_feature_store_write_then_read() -> None:
    from data_platform.feature_store import FeatureStore

    store = FeatureStore(registry=_registry_with())
    store.write([_sample_record(offset_days=i) for i in range(5)])
    rows = list(store.read(factor_id="mom_12_1", version="v1"))
    assert len(rows) == 5


def test_feature_store_refuses_write_against_unknown_definition() -> None:
    from data_platform.feature_store import FeatureStore

    store = FeatureStore(registry=_registry_with(factor_id="known"))
    with pytest.raises(LookupError, match="unknown factor"):
        store.write([_sample_record(factor_id="unknown")])


def test_feature_store_refuses_write_against_unknown_version() -> None:
    from data_platform.feature_store import FeatureStore

    store = FeatureStore(registry=_registry_with(version="v1"))
    with pytest.raises(LookupError, match="version"):
        store.write([_sample_record(version="v2")])


def test_feature_store_read_filters_by_window() -> None:
    from data_platform.feature_store import FeatureStore

    store = FeatureStore(registry=_registry_with())
    store.write([_sample_record(offset_days=i) for i in range(10)])
    rows = list(
        store.read(
            factor_id="mom_12_1",
            version="v1",
            start=datetime(2026, 4, 3, tzinfo=UTC),
            end=datetime(2026, 4, 7, tzinfo=UTC),
        )
    )
    assert len(rows) == 4  # days 3,4,5,6 (end-exclusive)


def test_feature_store_read_filters_by_symbol() -> None:
    from data_platform.feature_store import FeatureStore

    store = FeatureStore(registry=_registry_with())
    store.write(
        [
            _sample_record(symbol="AAPL"),
            _sample_record(symbol="MSFT"),
        ]
    )
    rows = list(store.read(factor_id="mom_12_1", version="v1", symbol="AAPL"))
    assert len(rows) == 1 and rows[0].symbol == "AAPL"


def test_feature_store_read_requires_version_when_registry_has_multiple() -> None:
    from data_platform.feature_store import FeatureStore

    reg = _registry_with(version="v1")
    reg.register(
        __import__(
            "data_platform.feature_store", fromlist=["FactorDefinition"]
        ).FactorDefinition(
            factor_id="mom_12_1",
            version="v2",
            description="",
            source_dependencies=(),
            universe="u",
            validation_status="candidate",
        )
    )
    store = FeatureStore(registry=reg)
    with pytest.raises(ValueError, match="version"):
        list(store.read(factor_id="mom_12_1"))
