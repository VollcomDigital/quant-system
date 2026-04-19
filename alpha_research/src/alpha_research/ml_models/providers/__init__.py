"""Forecasting-provider interfaces.

Kronos-style bounded model providers land behind these abstractions so
their internal types never leak into `alpha_research`, `backtest_engine`,
or `trading_system`. Every provider emits
`shared_lib.contracts.PredictionArtifact` values that can be persisted
through the Phase 2 `SnapshotIndex` exactly like factors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from typing import Protocol, runtime_checkable

from shared_lib.contracts import Bar, PredictionArtifact

__all__ = [
    "BatchForecaster",
    "CacheableForecastingProvider",
    "ForecastingProvider",
    "ProviderAdapter",
    "validate_provider_attrs",
]


@runtime_checkable
class ForecastingProvider(Protocol):
    """Minimum contract every forecasting provider must honour."""

    model_id: str
    version: str
    universe: str

    def forecast(self, bars: Iterable[Bar]) -> Iterator[PredictionArtifact]:
        ...


class CacheableForecastingProvider(ABC):
    """Base class for deterministic forecasters.

    A provider is 'cacheable' when `forecast(bars)` produces the same
    output for the same `bars` input. This is what lets the feature
    store re-use predictions as cached features.
    """

    model_id: str
    version: str
    universe: str

    @abstractmethod
    def forecast(self, bars: Iterable[Bar]) -> Iterator[PredictionArtifact]:
        ...


class BatchForecaster(ABC):
    """Provider that scores an entire universe in one call."""

    model_id: str
    version: str
    universe: str

    @abstractmethod
    def forecast_universe(
        self, bars_by_symbol: Mapping[str, Iterable[Bar]]
    ) -> Iterator[PredictionArtifact]:
        ...


class ProviderAdapter(ABC):
    """Base class for adapters wrapping upstream (external) model SDKs.

    Upstream types must never cross this boundary. Adapters import the
    upstream library themselves and expose only `PredictionArtifact`
    values to callers.
    """

    model_id: str
    version: str
    universe: str

    @abstractmethod
    def forecast(self, bars: Iterable[Bar]) -> Iterator[PredictionArtifact]:
        ...


def validate_provider_attrs(provider: object) -> None:
    """Raise `ValueError` if the provider is missing required metadata."""
    model_id = getattr(provider, "model_id", None)
    version = getattr(provider, "version", None)
    universe = getattr(provider, "universe", None)
    if not model_id:
        raise ValueError("provider.model_id must be a non-empty string")
    if not version:
        raise ValueError("provider.version must be a non-empty string")
    if not universe:
        raise ValueError("provider.universe must be a non-empty string")
