"""Phase 2 Task 1 - data_platform.connectors contracts.

The five connector concerns (ADR: `data-connector-concerns.md`) must be
strictly separated:

1. `ConnectorConfig`  - auth + endpoint + request shape
2. `RateLimitPolicy`  - min interval + burst budget
3. `RetrievalClient`  - fetch(symbol, interval, window) protocol
4. `Normalizer`       - raw vendor payload -> canonical `Bar`
5. `CachePolicy`      - dataset-id / snapshot resolution

A `Connector` is composed of one of each. The tests enforce that no
concern leaks into another and that the whole pipeline produces Phase 1
`shared_lib.contracts.Bar` records on the way out.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

# ---------------------------------------------------------------------------
# Edge case 1: ConnectorConfig enforces auth presence by kind.
# ---------------------------------------------------------------------------


def test_connector_config_requires_api_key_when_required() -> None:
    from data_platform.connectors import ConnectorConfig

    with pytest.raises(ValueError, match="api_key"):
        ConnectorConfig(provider="polygon", requires_auth=True, api_key=None)


def test_connector_config_accepts_unauth_provider() -> None:
    from data_platform.connectors import ConnectorConfig

    cfg = ConnectorConfig(provider="yfinance", requires_auth=False, api_key=None)
    assert cfg.provider == "yfinance"


def test_connector_config_redacts_api_key_in_repr() -> None:
    from data_platform.connectors import ConnectorConfig

    cfg = ConnectorConfig(provider="polygon", requires_auth=True, api_key="super-secret")
    assert "super-secret" not in repr(cfg)


# ---------------------------------------------------------------------------
# Edge case 2: RateLimitPolicy enforces sleep windows deterministically.
# ---------------------------------------------------------------------------


def test_rate_limit_policy_blocks_consecutive_calls() -> None:
    from data_platform.connectors import RateLimitPolicy

    policy = RateLimitPolicy(min_interval_seconds=0.05, burst=1)
    # First call goes through with wait=0.
    assert policy.acquire() == 0.0
    # Second call waits close to min_interval.
    waited = policy.acquire()
    assert 0.04 <= waited <= 0.10


def test_rate_limit_policy_rejects_zero_burst() -> None:
    from data_platform.connectors import RateLimitPolicy

    with pytest.raises(ValueError):
        RateLimitPolicy(min_interval_seconds=0.1, burst=0)


# ---------------------------------------------------------------------------
# Edge case 3: RetrievalClient protocol.
# ---------------------------------------------------------------------------


def test_retrieval_client_protocol_accepts_in_memory_impl() -> None:
    from data_platform.connectors import RetrievalClient

    class _InMemory:
        def fetch_raw(self, symbol: str, interval: str, start: datetime, end: datetime):
            return [
                {
                    "t": start.isoformat(),
                    "o": "1.0",
                    "h": "2.0",
                    "l": "0.5",
                    "c": "1.5",
                    "v": "100",
                }
            ]

    # Structural typing: this must satisfy the protocol.
    client: RetrievalClient = _InMemory()
    rows = client.fetch_raw("X", "1d", datetime(2026, 4, 1, tzinfo=UTC), datetime(2026, 4, 2, tzinfo=UTC))
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# Edge case 4: Normalizer turns raw vendor rows into canonical Bars.
# ---------------------------------------------------------------------------


def test_normalizer_emits_canonical_bar() -> None:
    from data_platform.connectors import OHLCVNormalizer
    from shared_lib.contracts import Bar

    raw = [
        {
            "t": "2026-04-01T00:00:00+00:00",
            "o": "100.0",
            "h": "102.0",
            "l": "99.0",
            "c": "101.0",
            "v": "10000",
        }
    ]
    norm = OHLCVNormalizer(symbol="AAPL", interval="1d")
    bars = list(norm.normalize(raw))
    assert len(bars) == 1
    assert isinstance(bars[0], Bar)
    assert bars[0].symbol == "AAPL"
    assert bars[0].close == Decimal("101.0")


def test_normalizer_rejects_missing_ohlc_field() -> None:
    from data_platform.connectors import OHLCVNormalizer

    norm = OHLCVNormalizer(symbol="AAPL", interval="1d")
    with pytest.raises(KeyError):
        list(
            norm.normalize(
                [{"t": "2026-04-01T00:00:00+00:00", "o": "1", "h": "2"}]
            )
        )


def test_normalizer_rejects_naive_timestamp_string() -> None:
    from data_platform.connectors import OHLCVNormalizer

    norm = OHLCVNormalizer(symbol="AAPL", interval="1d")
    with pytest.raises(ValueError):
        list(
            norm.normalize(
                [{"t": "2026-04-01T00:00:00", "o": "1", "h": "2", "l": "1", "c": "1.5", "v": "0"}]
            )
        )


# ---------------------------------------------------------------------------
# Edge case 5: CachePolicy produces deterministic dataset ids per window.
# ---------------------------------------------------------------------------


def test_cache_policy_dataset_id_is_deterministic() -> None:
    from data_platform.connectors import CachePolicy

    p = CachePolicy(namespace="data_platform.bars")
    a = p.dataset_id(provider="polygon", symbol="AAPL", interval="1d",
                     start=datetime(2026, 4, 1, tzinfo=UTC),
                     end=datetime(2026, 4, 30, tzinfo=UTC))
    b = p.dataset_id(provider="polygon", symbol="AAPL", interval="1d",
                     start=datetime(2026, 4, 1, tzinfo=UTC),
                     end=datetime(2026, 4, 30, tzinfo=UTC))
    assert a == b


def test_cache_policy_changes_with_interval() -> None:
    from data_platform.connectors import CachePolicy

    p = CachePolicy(namespace="data_platform.bars")
    a = p.dataset_id(provider="polygon", symbol="AAPL", interval="1d",
                     start=datetime(2026, 4, 1, tzinfo=UTC),
                     end=datetime(2026, 4, 30, tzinfo=UTC))
    b = p.dataset_id(provider="polygon", symbol="AAPL", interval="1h",
                     start=datetime(2026, 4, 1, tzinfo=UTC),
                     end=datetime(2026, 4, 30, tzinfo=UTC))
    assert a != b


# ---------------------------------------------------------------------------
# Edge case 6: Connector composes all concerns and yields Bars.
# ---------------------------------------------------------------------------


def test_connector_composes_concerns_and_yields_bars() -> None:
    from data_platform.connectors import (
        CachePolicy,
        Connector,
        ConnectorConfig,
        OHLCVNormalizer,
        RateLimitPolicy,
    )

    class _Source:
        def fetch_raw(self, symbol: str, interval: str, start: datetime, end: datetime):
            return [
                {
                    "t": (start + timedelta(days=i)).isoformat(),
                    "o": "1",
                    "h": "2",
                    "l": "0.5",
                    "c": "1.5",
                    "v": str(100 + i),
                }
                for i in range(3)
            ]

    conn = Connector(
        config=ConnectorConfig(provider="yfinance", requires_auth=False, api_key=None),
        rate_limit=RateLimitPolicy(min_interval_seconds=0.0, burst=1),
        retrieval=_Source(),
        normalizer=OHLCVNormalizer(symbol="AAPL", interval="1d"),
        cache=CachePolicy(namespace="data_platform.bars"),
    )
    bars = list(
        conn.fetch(
            symbol="AAPL",
            interval="1d",
            start=datetime(2026, 4, 1, tzinfo=UTC),
            end=datetime(2026, 4, 4, tzinfo=UTC),
        )
    )
    assert len(bars) == 3
    assert all(b.symbol == "AAPL" for b in bars)


# ---------------------------------------------------------------------------
# Edge case 7: Connector auth is not optional when the config says it is
# required. This guards against constructing a live connector without
# credentials during tests.
# ---------------------------------------------------------------------------


def test_connector_refuses_network_call_without_api_key_when_required() -> None:
    from data_platform.connectors import (
        CachePolicy,
        Connector,
        ConnectorConfig,
        OHLCVNormalizer,
        RateLimitPolicy,
    )

    class _FakeRetrieval:
        def fetch_raw(self, *args, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("retrieval must not be called without auth")

    # ConnectorConfig already rejects api_key=None when requires_auth=True, so
    # the invariant is enforced at construction time.
    with pytest.raises(ValueError):
        Connector(
            config=ConnectorConfig(provider="polygon", requires_auth=True, api_key=None),
            rate_limit=RateLimitPolicy(min_interval_seconds=0.0, burst=1),
            retrieval=_FakeRetrieval(),
            normalizer=OHLCVNormalizer(symbol="X", interval="1d"),
            cache=CachePolicy(namespace="data_platform.bars"),
        )


# ---------------------------------------------------------------------------
# Edge case 8: ProviderRegistry maps every required provider id to a
# ConnectorConfig.
# ---------------------------------------------------------------------------


REQUIRED_PROVIDERS = (
    "yfinance",
    "ccxt",
    "alpaca",
    "polygon",
    "tiingo",
    "finnhub",
    "twelvedata",
    "alphavantage",
)


@pytest.mark.parametrize("provider", REQUIRED_PROVIDERS)
def test_provider_registry_knows_every_required_provider(provider: str) -> None:
    from data_platform.connectors import PROVIDER_REGISTRY

    assert provider in PROVIDER_REGISTRY
    spec = PROVIDER_REGISTRY[provider]
    assert spec.provider == provider
    # Spec documents whether auth is required - that is all Phase 2 needs.
    assert isinstance(spec.requires_auth, bool)
