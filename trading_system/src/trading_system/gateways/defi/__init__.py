"""DeFi-specific kill-switch helpers.

ADR-0004 Layer 4: pause + revoke + protocol-denylist controls. These
helpers build `UnsignedTransaction` objects that the Phase 7 Web3
gateway broadcasts. Phase 7 does not hold private keys; the Phase 9
KMS-backed signer does the actual signing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from trading_system.gateways.web3 import UnsignedTransaction

__all__ = [
    "ProtocolDenylist",
    "request_pause",
    "request_revoke_allowances",
]


def request_pause(
    *,
    chain_id: int,
    from_address: str,
    target_contract: str,
    nonce: int,
    gas_limit: int,
) -> UnsignedTransaction:
    if gas_limit <= 0:
        raise ValueError("gas_limit must be > 0")
    return UnsignedTransaction(
        chain_id=chain_id,
        from_address=from_address,
        to_address=target_contract,
        protocol="pausable",
        version="v1",
        function_name="pause",
        args={},
        gas_limit=gas_limit,
        max_fee_per_gas=Decimal("100"),
        max_priority_fee_per_gas=Decimal("2"),
        nonce=nonce,
    )


def request_revoke_allowances(
    *,
    chain_id: int,
    from_address: str,
    token_address: str,
    spenders: Sequence[str],
    nonce_start: int,
    gas_limit: int,
) -> list[UnsignedTransaction]:
    if not spenders:
        raise ValueError("spenders must be non-empty")
    if gas_limit <= 0:
        raise ValueError("gas_limit must be > 0")
    return [
        UnsignedTransaction(
            chain_id=chain_id,
            from_address=from_address,
            to_address=token_address,
            protocol="erc20",
            version="v1",
            function_name="approve",
            args={"spender": spender, "amount": "0"},
            gas_limit=gas_limit,
            max_fee_per_gas=Decimal("100"),
            max_priority_fee_per_gas=Decimal("2"),
            nonce=nonce_start + i,
        )
        for i, spender in enumerate(spenders)
    ]


@dataclass
class ProtocolDenylist:
    _entries: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def add(self, protocol: str, *, reason: str) -> None:
        if not reason:
            raise ValueError("reason must be a non-empty string")
        self._entries[protocol] = reason

    def remove(self, protocol: str) -> None:
        self._entries.pop(protocol, None)

    def is_blocked(self, protocol: str) -> bool:
        return protocol in self._entries

    def reason(self, protocol: str) -> str | None:
        return self._entries.get(protocol)
