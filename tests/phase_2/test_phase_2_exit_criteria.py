"""Phase 2 Exit Criteria - aggregate gate.

Exit criteria (from `tasks/todo.md`):

- Connector interfaces are separated from cache/storage concerns.
- Airflow DAG structure exists for backfill and refresh workflows.
- Feature definitions are versioned and reusable across research,
  backtesting, and live systems.
- Vendor historical data path is documented separately from live broker
  connectivity.
"""

from __future__ import annotations


def test_exit_connectors_separated_from_storage() -> None:
    import data_platform.connectors as c
    import data_platform.storage as s

    # Connectors must NOT expose snapshot primitives.
    assert not hasattr(c, "SnapshotIndex")
    # Storage must NOT expose connector primitives.
    assert not hasattr(s, "Connector")


def test_exit_dag_and_backfill_exist() -> None:
    from data_platform.pipelines import DAG, TaskSpec, backfill_windows  # noqa: F401


def test_exit_feature_definitions_are_versioned() -> None:
    from data_platform.feature_store import FactorDefinition, FactorRegistry

    reg = FactorRegistry()
    reg.register(
        FactorDefinition(
            factor_id="f",
            version="v1",
            description="",
            source_dependencies=(),
            universe="u",
            validation_status="candidate",
        )
    )
    reg.register(
        FactorDefinition(
            factor_id="f",
            version="v2",
            description="",
            source_dependencies=(),
            universe="u",
            validation_status="candidate",
        )
    )
    assert set(reg.versions("f")) == {"v1", "v2"}


def test_exit_vendor_routing_is_documented_in_registry() -> None:
    from data_platform.connectors import PROVIDER_REGISTRY

    # Alpaca/IBKR-style brokers must be marked `live`; historical vendors
    # marked `historical`. This is the machine-readable mirror of the
    # policy doc.
    assert PROVIDER_REGISTRY["alpaca"].role == "live"
    assert PROVIDER_REGISTRY["polygon"].role == "historical"
    assert PROVIDER_REGISTRY["tiingo"].role == "historical"
    assert PROVIDER_REGISTRY["databento"].role == "historical"
