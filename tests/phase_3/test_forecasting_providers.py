"""Phase 3 Task 2 - alpha_research.ml_models.providers.

Per the Kronos adoption rules (roadmap):

- Kronos-like models are *bounded providers*. They never define
  internal schemas.
- Every provider lands behind
  `alpha_research.ml_models.providers.ForecastingProvider`.
- Outputs are `shared_lib.contracts.PredictionArtifact` values so they
  can be persisted through the Phase 2 SnapshotIndex exactly like
  factors.
- Providers carry a `model_id`, `version`, and `universe` attribute
  so the registry can bind them to a specific training run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

# ---------------------------------------------------------------------------
# Protocol acceptance
# ---------------------------------------------------------------------------


def test_forecasting_provider_protocol_accepts_in_memory_impl() -> None:
    from alpha_research.ml_models.providers import ForecastingProvider
    from shared_lib.contracts import PredictionArtifact

    class _ZeroProvider:
        model_id = "zero-v0"
        version = "v0"
        universe = "us_equities"

        def forecast(self, bars):
            # Emit one prediction per bar, always zero.
            for b in bars:
                yield PredictionArtifact(
                    model_id=self.model_id,
                    symbol=b.symbol,
                    horizon="1d",
                    generated_at=b.timestamp,
                    value=Decimal("0"),
                    confidence=Decimal("0.5"),
                )

    provider: ForecastingProvider = _ZeroProvider()
    assert provider.model_id == "zero-v0"


# ---------------------------------------------------------------------------
# CacheableForecastingProvider - requires deterministic output for a given
# (symbol, interval, start, end) window.
# ---------------------------------------------------------------------------


def _bar(ts: datetime, symbol: str = "AAPL"):
    from shared_lib.contracts import Bar

    return Bar(
        symbol=symbol,
        interval="1d",
        timestamp=ts,
        open=Decimal("100"),
        high=Decimal("100"),
        low=Decimal("100"),
        close=Decimal("100"),
        volume=Decimal("0"),
    )


def test_cacheable_provider_returns_deterministic_output() -> None:
    from alpha_research.ml_models.providers import CacheableForecastingProvider
    from shared_lib.contracts import PredictionArtifact

    class _Det(CacheableForecastingProvider):
        model_id = "det-v0"
        version = "v0"
        universe = "us_equities"

        def forecast(self, bars):
            for b in bars:
                yield PredictionArtifact(
                    model_id=self.model_id,
                    symbol=b.symbol,
                    horizon="1d",
                    generated_at=b.timestamp,
                    value=Decimal("0.01"),
                    confidence=Decimal("0.9"),
                )

    p = _Det()
    start = datetime(2026, 4, 1, tzinfo=UTC)
    bars = [_bar(start + timedelta(days=i)) for i in range(3)]
    a = list(p.forecast(bars))
    b = list(p.forecast(bars))
    assert a == b


# ---------------------------------------------------------------------------
# BatchForecaster - produces predictions across an entire universe in one
# call so downstream persistence can run a single snapshot write.
# ---------------------------------------------------------------------------


def test_batch_forecaster_emits_by_symbol() -> None:
    from alpha_research.ml_models.providers import BatchForecaster
    from shared_lib.contracts import PredictionArtifact

    class _Batch(BatchForecaster):
        model_id = "batch-v0"
        version = "v0"
        universe = "us_equities"

        def forecast_universe(self, bars_by_symbol):
            for symbol, bars in bars_by_symbol.items():
                for b in bars:
                    yield PredictionArtifact(
                        model_id=self.model_id,
                        symbol=symbol,
                        horizon="1d",
                        generated_at=b.timestamp,
                        value=Decimal("0.01"),
                        confidence=Decimal("0.8"),
                    )

    p = _Batch()
    start = datetime(2026, 4, 1, tzinfo=UTC)
    bars_by_symbol = {
        "AAPL": [_bar(start, symbol="AAPL")],
        "MSFT": [_bar(start, symbol="MSFT")],
    }
    got = list(p.forecast_universe(bars_by_symbol))
    assert {g.symbol for g in got} == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# ProviderAdapter: external (upstream) providers must land behind an
# adapter so their types never leak into domain code.
# ---------------------------------------------------------------------------


def test_provider_adapter_wraps_external_model() -> None:
    from alpha_research.ml_models.providers import ProviderAdapter
    from shared_lib.contracts import PredictionArtifact

    # Simulate an external "Kronos-like" model with its own return shape.
    class _External:
        def predict(self, symbol, window):
            return [(window[-1], 0.02, 0.7)]

    class _Adapter(ProviderAdapter):
        model_id = "kronos-like-v0"
        version = "v0"
        universe = "us_equities"

        def __init__(self, external):
            self._external = external

        def forecast(self, bars):
            bars = list(bars)
            window = [b.timestamp for b in bars]
            for ts, value, conf in self._external.predict(bars[0].symbol, window):
                yield PredictionArtifact(
                    model_id=self.model_id,
                    symbol=bars[0].symbol,
                    horizon="1d",
                    generated_at=ts,
                    value=Decimal(str(value)),
                    confidence=Decimal(str(conf)),
                )

    bars = [_bar(datetime(2026, 4, 1, tzinfo=UTC))]
    adapter = _Adapter(_External())
    out = list(adapter.forecast(bars))
    assert len(out) == 1 and out[0].confidence == Decimal("0.7")


# ---------------------------------------------------------------------------
# Provider metadata validation.
# ---------------------------------------------------------------------------


def test_provider_requires_model_id_and_version() -> None:
    from alpha_research.ml_models.providers import validate_provider_attrs

    class _NoId:
        model_id = ""
        version = "v1"
        universe = "u"

        def forecast(self, bars):  # pragma: no cover
            yield from ()

    with pytest.raises(ValueError, match="model_id"):
        validate_provider_attrs(_NoId())


def test_provider_requires_universe() -> None:
    from alpha_research.ml_models.providers import validate_provider_attrs

    class _NoUniverse:
        model_id = "m"
        version = "v1"
        universe = ""

        def forecast(self, bars):  # pragma: no cover
            yield from ()

    with pytest.raises(ValueError, match="universe"):
        validate_provider_attrs(_NoUniverse())
