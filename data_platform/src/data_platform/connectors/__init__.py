"""Connector contracts for the data platform.

A `Connector` composes five independent concerns:

- `ConnectorConfig`  - auth/config
- `RateLimitPolicy`  - request pacing
- `RetrievalClient`  - raw fetch (protocol)
- `Normalizer`       - raw payload -> canonical `shared_lib.contracts.Bar`
- `CachePolicy`      - dataset id / snapshot key

Domain code NEVER constructs these directly from `src/data/*`. The legacy
providers stay in `src/` during Phase 2 and will be migrated in Phase 10
behind compatibility adapters.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

from shared_lib.contracts import Bar

__all__ = [
    "PROVIDER_REGISTRY",
    "CachePolicy",
    "Connector",
    "ConnectorConfig",
    "Normalizer",
    "OHLCVNormalizer",
    "ProviderSpec",
    "RateLimitPolicy",
    "RetrievalClient",
]


# ---------------------------------------------------------------------------
# ConnectorConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConnectorConfig:
    provider: str
    requires_auth: bool
    api_key: str | None

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("ConnectorConfig requires a non-empty provider")
        if self.requires_auth and not self.api_key:
            raise ValueError(
                f"ConnectorConfig for provider {self.provider!r}: api_key is required"
            )

    def __repr__(self) -> str:
        redacted = "***REDACTED***" if self.api_key else None
        return (
            f"ConnectorConfig(provider={self.provider!r}, "
            f"requires_auth={self.requires_auth}, api_key={redacted!r})"
        )


# ---------------------------------------------------------------------------
# RateLimitPolicy
# ---------------------------------------------------------------------------


@dataclass
class RateLimitPolicy:
    min_interval_seconds: float
    burst: int
    _last: float = field(default=0.0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.burst <= 0:
            raise ValueError("burst must be > 0")
        if self.min_interval_seconds < 0.0:
            raise ValueError("min_interval_seconds must be >= 0")

    def acquire(self) -> float:
        """Sleep if needed to honour the rate limit. Returns actual wait."""
        with self._lock:
            now = time.perf_counter()
            delta = now - self._last
            wait = max(0.0, self.min_interval_seconds - delta)
            if wait > 0:
                time.sleep(wait)
            self._last = time.perf_counter()
            return wait


# ---------------------------------------------------------------------------
# RetrievalClient protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class RetrievalClient(Protocol):
    """Raw fetch surface. Implementations return vendor-shaped rows."""

    def fetch_raw(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> Sequence[dict[str, Any]]:
        ...


# ---------------------------------------------------------------------------
# Normalizer + OHLCVNormalizer
# ---------------------------------------------------------------------------


class Normalizer(Protocol):
    """Transform vendor rows into canonical contracts."""

    def normalize(self, rows: Iterable[dict[str, Any]]) -> Iterator[Bar]:
        ...


@dataclass(frozen=True, slots=True)
class OHLCVNormalizer:
    """Default normalizer for OHLCV bars.

    Expects rows with keys: t (ISO 8601 tz-aware), o, h, l, c, v.
    Any missing field raises KeyError; naive timestamps raise ValueError.
    """

    symbol: str
    interval: str

    def normalize(self, rows: Iterable[dict[str, Any]]) -> Iterator[Bar]:
        for row in rows:
            ts_str = row["t"]
            timestamp = datetime.fromisoformat(ts_str)
            if timestamp.tzinfo is None:
                raise ValueError(
                    f"timestamp {ts_str!r} is naive; require tz-aware ISO 8601"
                )
            yield Bar(
                symbol=self.symbol,
                interval=self.interval,
                timestamp=timestamp,
                open=Decimal(row["o"]),
                high=Decimal(row["h"]),
                low=Decimal(row["l"]),
                close=Decimal(row["c"]),
                volume=Decimal(row["v"]),
            )


# ---------------------------------------------------------------------------
# CachePolicy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CachePolicy:
    namespace: str

    def dataset_id(
        self,
        *,
        provider: str,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> str:
        fingerprint = "|".join(
            (
                self.namespace,
                provider,
                symbol,
                interval,
                start.astimezone().isoformat(),
                end.astimezone().isoformat(),
            )
        )
        digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
        return f"{self.namespace}::{provider}::{symbol}::{interval}::{digest}"


# ---------------------------------------------------------------------------
# Connector - composes all five concerns.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Connector:
    config: ConnectorConfig
    rate_limit: RateLimitPolicy
    retrieval: RetrievalClient
    normalizer: Normalizer
    cache: CachePolicy

    def fetch(
        self,
        *,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> Iterator[Bar]:
        self.rate_limit.acquire()
        rows = self.retrieval.fetch_raw(symbol, interval, start, end)
        yield from self.normalizer.normalize(rows)


# ---------------------------------------------------------------------------
# Provider registry - the canonical list of providers Phase 2 covers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    provider: str
    requires_auth: bool
    role: str  # "historical" | "live" | "mixed"


PROVIDER_REGISTRY: dict[str, ProviderSpec] = {
    "yfinance": ProviderSpec("yfinance", requires_auth=False, role="historical"),
    "ccxt": ProviderSpec("ccxt", requires_auth=False, role="mixed"),
    "alpaca": ProviderSpec("alpaca", requires_auth=True, role="live"),
    "polygon": ProviderSpec("polygon", requires_auth=True, role="historical"),
    "tiingo": ProviderSpec("tiingo", requires_auth=True, role="historical"),
    "finnhub": ProviderSpec("finnhub", requires_auth=True, role="historical"),
    "twelvedata": ProviderSpec("twelvedata", requires_auth=True, role="historical"),
    "alphavantage": ProviderSpec("alphavantage", requires_auth=True, role="historical"),
    "databento": ProviderSpec("databento", requires_auth=True, role="historical"),
}
