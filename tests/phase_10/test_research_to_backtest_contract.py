"""Phase 10 Task 1 - research -> backtest contract test.

Chain under test:

1. A Phase 3 `Factor` subclass computes `FactorRecord`s from bars.
2. The Phase 3 `FactorLibrary` projects its metadata onto the Phase 2
   `data_platform.feature_store.FactorRegistry`.
3. The Phase 2 `FeatureStore` accepts the records and the Phase 4
   `Simulator` consumes the same bar stream to produce fills.

Everything crosses real package boundaries — no mocking of the
contracts.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal


def _bar(ts: datetime, close: str):
    from shared_lib.contracts import Bar

    return Bar(
        symbol="AAPL",
        interval="1d",
        timestamp=ts,
        open=Decimal(close),
        high=Decimal(close),
        low=Decimal(close),
        close=Decimal(close),
        volume=Decimal("100000"),
    )


def test_factor_library_exports_into_feature_store_registry() -> None:
    """Phase 3 FactorLibrary.export_to must populate the Phase 2 registry."""
    from alpha_research.factor_library import Factor, FactorLibrary, FactorMetadata
    from data_platform.feature_store import FactorRegistry

    class _MomFactor(Factor):
        metadata = FactorMetadata(
            factor_id="mom_12_1",
            version="v1",
            description="12-1 momentum reference",
            source_dependencies=("bars",),
            stationarity_assumption="mean-reverting monthly",
            universe_coverage="us_equities",
            leakage_review="reviewed: uses only t-1 close",
            validation_status="candidate",
        )

        def compute(self, bars):
            yield from ()

    lib = FactorLibrary()
    lib.register(_MomFactor())
    registry = FactorRegistry()
    lib.export_to(registry)

    # The Phase 2 feature store now knows about the factor and can
    # accept writes against the exact metadata the researcher declared.
    assert registry.has("mom_12_1", "v1")
    defn = registry.get("mom_12_1", "v1")
    assert defn.validation_status == "candidate"
    assert defn.source_dependencies == ("bars",)


def test_factor_records_flow_into_feature_store_and_out_to_callers() -> None:
    """A FactorRecord written into the FeatureStore is readable with the
    same factor_id + version + window filters."""
    from alpha_research.factor_library import Factor, FactorLibrary, FactorMetadata
    from data_platform.feature_store import FactorRegistry, FeatureStore
    from shared_lib.contracts import FactorRecord

    class _F(Factor):
        metadata = FactorMetadata(
            factor_id="ref",
            version="v1",
            description="d",
            source_dependencies=(),
            stationarity_assumption="s",
            universe_coverage="u",
            leakage_review="ok",
            validation_status="candidate",
        )

        def compute(self, bars):
            yield from ()

    lib = FactorLibrary()
    lib.register(_F())
    reg = FactorRegistry()
    lib.export_to(reg)
    store = FeatureStore(registry=reg)

    start = datetime(2026, 4, 1, tzinfo=UTC)
    records = [
        FactorRecord(
            factor_id="ref",
            as_of=start + timedelta(days=i),
            symbol="AAPL",
            value=Decimal("0.01"),
            version="v1",
        )
        for i in range(5)
    ]
    store.write(records)
    got = list(
        store.read(
            factor_id="ref",
            version="v1",
            start=start + timedelta(days=1),
            end=start + timedelta(days=4),
        )
    )
    assert len(got) == 3


def test_backtest_simulator_consumes_same_bar_stream_as_factor_compute() -> None:
    """The Phase 4 Simulator produces fills from the same `Bar` shape
    the Phase 3 Factor would consume; the research signal timestamp
    equals the bar timestamp (look-ahead prevention)."""
    from backtest_engine.market_mechanics import FixedBpsSlippage, PercentageFee
    from backtest_engine.simulator import Simulator
    from shared_lib.math_utils import Money

    class _AlwaysLong:
        def on_bar(self, bar, context):
            # Mirrors how a Phase 3 Factor would derive a signal: the
            # signal carries the bar's own timestamp.
            if len(list(context.history())) == 1:
                return [
                    context.make_signal(direction="long", strength=Decimal("1"))
                ]
            return []

    start = datetime(2026, 4, 1, tzinfo=UTC)
    bars = [_bar(start + timedelta(days=i), close=str(100 + i)) for i in range(3)]

    sim = Simulator(
        strategy=_AlwaysLong(),
        starting_cash=Money("10000", "USD"),
        slippage=FixedBpsSlippage(bps=Decimal("0")),
        fee=PercentageFee(rate=Decimal("0")),
        trade_size=Decimal("10"),
    )
    run = sim.run(run_id="rt-bt-1", bars=bars)
    assert run.fill_count == 1
    # Signal at bar[0] -> filled at bar[0].close = 100; final mark 102.
    # Equity = 10000 - 10*100 + 10*102 = 10020.
    assert run.final_equity == Money("10020", "USD")
    # The fill timestamp equals the originating bar timestamp (Phase 4
    # leakage invariant carried end-to-end).
    assert run.fills[0].filled_at == start


def test_promotion_result_is_the_same_validation_result_shape() -> None:
    """The Phase 3 promote_factor returns the same ValidationResult
    shape the Phase 2 quality checks + Phase 6 RMS + Phase 5 risk
    monitor all consume — proving the contract really is shared."""
    from alpha_research.factor_library import FactorMetadata
    from alpha_research.promotion import FactorPromotionRequest, promote_factor
    from shared_lib.contracts import ValidationResult

    m = FactorMetadata(
        factor_id="f1",
        version="v1",
        description="d",
        source_dependencies=(),
        stationarity_assumption="s",
        universe_coverage="u",
        leakage_review="ok",
        validation_status="candidate",
    )
    vr = promote_factor(
        FactorPromotionRequest(
            metadata=m,
            target_status="validated",
            coverage_pct=Decimal("0.9"),
            oos_sharpe=Decimal("1.0"),
            leakage_check_passed=True,
        )
    )
    assert isinstance(vr, ValidationResult)
    assert vr.passed is True
