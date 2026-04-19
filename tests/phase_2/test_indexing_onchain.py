"""Phase 2 Task 5 - data_platform.indexing on-chain contracts.

Three pieces Phase 2 must nail down before Phase 7 gateways touch EVM
chains:

1. ABI Registry: versioned ABI storage keyed by (protocol, version), with
   an explicit immutability guarantee.
2. Protocol-normalized event schemas (swap, lend, borrow, liquidity,
   vault) so decoded on-chain data maps onto a stable contract.
3. Raw log record shape for Parquet-backed ETL of EVM logs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

# ---------------------------------------------------------------------------
# ABI registry
# ---------------------------------------------------------------------------


def test_abi_registry_round_trip() -> None:
    from data_platform.indexing import ABIRegistry

    reg = ABIRegistry()
    abi = [{"type": "function", "name": "swap", "inputs": []}]
    reg.register(protocol="uniswap_v3", version="1.0.0", abi=abi)
    assert reg.get(protocol="uniswap_v3", version="1.0.0") == abi


def test_abi_registry_refuses_overwrite() -> None:
    from data_platform.indexing import ABIRegistry

    reg = ABIRegistry()
    reg.register(protocol="p", version="v1", abi=[{"type": "event"}])
    with pytest.raises(ValueError, match="already"):
        reg.register(protocol="p", version="v1", abi=[{"type": "event", "new": True}])


def test_abi_registry_rejects_non_list_abi() -> None:
    from data_platform.indexing import ABIRegistry

    reg = ABIRegistry()
    with pytest.raises(TypeError):
        reg.register(protocol="p", version="v1", abi={"not": "a list"})  # type: ignore[arg-type]


def test_abi_registry_lookup_missing_raises() -> None:
    from data_platform.indexing import ABIRegistry

    reg = ABIRegistry()
    with pytest.raises(LookupError):
        reg.get(protocol="x", version="y")


# ---------------------------------------------------------------------------
# Normalized protocol events
# ---------------------------------------------------------------------------


def test_swap_event_requires_positive_amounts() -> None:
    from data_platform.indexing import SwapEvent

    with pytest.raises(ValueError):
        SwapEvent(
            chain_id=1,
            protocol="uniswap_v3",
            pool="0xpool",
            tx_hash="0xtx",
            block_number=100,
            block_timestamp=datetime(2026, 4, 1, tzinfo=UTC),
            token_in="0xA",
            token_out="0xB",
            amount_in=Decimal("0"),
            amount_out=Decimal("1"),
        )


def test_swap_event_round_trip() -> None:
    from data_platform.indexing import SwapEvent

    ev = SwapEvent(
        chain_id=1,
        protocol="uniswap_v3",
        pool="0xpool",
        tx_hash="0xtx",
        block_number=100,
        block_timestamp=datetime(2026, 4, 1, tzinfo=UTC),
        token_in="0xA",
        token_out="0xB",
        amount_in=Decimal("100"),
        amount_out=Decimal("99.5"),
    )
    restored = SwapEvent.model_validate_json(ev.model_dump_json())
    assert restored == ev


def test_lend_event_negative_amount_rejected() -> None:
    from data_platform.indexing import LendEvent

    with pytest.raises(ValueError):
        LendEvent(
            chain_id=1,
            protocol="aave_v3",
            pool="0xpool",
            tx_hash="0xtx",
            block_number=1,
            block_timestamp=datetime(2026, 4, 1, tzinfo=UTC),
            asset="0xA",
            amount=Decimal("-1"),
            user="0xuser",
        )


def test_liquidity_event_rejects_zero_amounts() -> None:
    from data_platform.indexing import LiquidityEvent

    with pytest.raises(ValueError):
        LiquidityEvent(
            chain_id=1,
            protocol="uniswap_v3",
            pool="0xpool",
            tx_hash="0xtx",
            block_number=1,
            block_timestamp=datetime(2026, 4, 1, tzinfo=UTC),
            action="mint",
            amount0=Decimal("0"),
            amount1=Decimal("0"),
            provider="0xprovider",
        )


def test_vault_event_action_enum_closed() -> None:
    from data_platform.indexing import VaultEvent

    with pytest.raises(ValueError):
        VaultEvent(
            chain_id=1,
            protocol="yearn",
            vault="0xv",
            tx_hash="0xtx",
            block_number=1,
            block_timestamp=datetime(2026, 4, 1, tzinfo=UTC),
            action="shenanigans",
            shares=Decimal("1"),
            assets=Decimal("1"),
            user="0xu",
        )


# ---------------------------------------------------------------------------
# Raw log record
# ---------------------------------------------------------------------------


def test_raw_log_record_requires_topics() -> None:
    from data_platform.indexing import RawLogRecord

    with pytest.raises(ValueError):
        RawLogRecord(
            chain_id=1,
            address="0x0",
            block_number=1,
            block_timestamp=datetime(2026, 4, 1, tzinfo=UTC),
            tx_hash="0x",
            log_index=0,
            topics=(),  # empty -> rejected
            data="0x",
        )
