"""On-chain indexing contracts.

Three pieces Phase 2 commits to:

1. `ABIRegistry` - versioned storage of protocol ABIs; immutable per
   `(protocol, version)`.
2. Protocol-normalized event schemas: `SwapEvent`, `LendEvent`,
   `BorrowEvent`, `LiquidityEvent`, `VaultEvent` - all pydantic v2
   contracts that survive Parquet round-trip.
3. `RawLogRecord` - the Parquet-shaped record written by custom EVM ETL.

These are stable for every chain we later target. A Graph adapter and a
custom-ETL adapter are both valid producers of these records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import Field
from shared_lib.contracts._base import Schema, aware_datetime_validator

__all__ = [
    "ABIRegistry",
    "BorrowEvent",
    "LendEvent",
    "LiquidityEvent",
    "RawLogRecord",
    "SwapEvent",
    "VaultEvent",
]


@dataclass
class ABIRegistry:
    _abis: dict[tuple[str, str], list[dict[str, Any]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def register(
        self, *, protocol: str, version: str, abi: list[dict[str, Any]]
    ) -> None:
        if not isinstance(abi, list):
            raise TypeError("abi must be a list of entries")
        key = (protocol, version)
        if key in self._abis:
            raise ValueError(f"ABI for {key} already registered")
        self._abis[key] = list(abi)

    def get(self, *, protocol: str, version: str) -> list[dict[str, Any]]:
        try:
            return self._abis[(protocol, version)]
        except KeyError as exc:
            raise LookupError(
                f"no ABI registered for {(protocol, version)!r}"
            ) from exc


class _OnChainBase(Schema):
    chain_id: int = Field(gt=0)
    protocol: str = Field(min_length=1)
    tx_hash: str = Field(min_length=1)
    block_number: int = Field(ge=0)
    block_timestamp: datetime

    _ts = aware_datetime_validator("block_timestamp")


class SwapEvent(_OnChainBase):
    pool: str = Field(min_length=1)
    token_in: str = Field(min_length=1)
    token_out: str = Field(min_length=1)
    amount_in: Decimal = Field(gt=0)
    amount_out: Decimal = Field(gt=0)


class LendEvent(_OnChainBase):
    pool: str = Field(min_length=1)
    asset: str = Field(min_length=1)
    amount: Decimal = Field(gt=0)
    user: str = Field(min_length=1)


class BorrowEvent(_OnChainBase):
    pool: str = Field(min_length=1)
    asset: str = Field(min_length=1)
    amount: Decimal = Field(gt=0)
    user: str = Field(min_length=1)


LiquidityAction = Literal["mint", "burn"]


class LiquidityEvent(_OnChainBase):
    pool: str = Field(min_length=1)
    action: LiquidityAction
    amount0: Decimal = Field(gt=0)
    amount1: Decimal = Field(gt=0)
    provider: str = Field(min_length=1)


VaultAction = Literal["deposit", "withdraw", "harvest"]


class VaultEvent(_OnChainBase):
    vault: str = Field(min_length=1)
    action: VaultAction
    shares: Decimal = Field(gt=0)
    assets: Decimal = Field(gt=0)
    user: str = Field(min_length=1)


class RawLogRecord(Schema):
    chain_id: int = Field(gt=0)
    address: str = Field(min_length=1)
    block_number: int = Field(ge=0)
    block_timestamp: datetime
    tx_hash: str = Field(min_length=1)
    log_index: int = Field(ge=0)
    topics: tuple[str, ...] = Field(min_length=1)
    data: str

    _ts = aware_datetime_validator("block_timestamp")
