"""Web3 / DeFi gateway.

Builds + simulates + signs + broadcasts EVM transactions. The gateway
never holds a private key: signing happens through a `SigningClient`
implementation that lives behind KMS / Vault (ADR-0006). Calldata
encoding is ABI-driven via `data_platform.indexing.ABIRegistry`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

from data_platform.indexing import ABIRegistry
from shared_lib.contracts import HealthStatus

__all__ = [
    "FakeRpcClient",
    "FakeSigningClient",
    "RpcClient",
    "SigningClient",
    "TxReceipt",
    "UnsignedTransaction",
    "Web3Gateway",
    "build_unsigned_tx",
]


@dataclass(frozen=True, slots=True)
class UnsignedTransaction:
    chain_id: int
    from_address: str
    to_address: str
    protocol: str
    version: str
    function_name: str
    args: dict[str, Any]
    gas_limit: int
    max_fee_per_gas: Decimal
    max_priority_fee_per_gas: Decimal
    nonce: int

    def __post_init__(self) -> None:
        if self.nonce < 0:
            raise ValueError("nonce must be >= 0")
        if self.gas_limit <= 0:
            raise ValueError("gas_limit must be > 0")
        if self.max_fee_per_gas < 0 or self.max_priority_fee_per_gas < 0:
            raise ValueError("gas fees must be >= 0")


def build_unsigned_tx(
    *,
    chain_id: int,
    from_address: str,
    to_address: str,
    protocol: str,
    version: str,
    function_name: str,
    args: dict[str, Any],
    gas_limit: int,
    max_fee_per_gas: Decimal,
    max_priority_fee_per_gas: Decimal,
    nonce: int,
    abi_registry: ABIRegistry,
) -> UnsignedTransaction:
    abi = abi_registry.get(protocol=protocol, version=version)
    fns = {entry["name"] for entry in abi if entry.get("type") == "function"}
    if function_name not in fns:
        raise ValueError(
            f"function {function_name!r} not in ABI for {(protocol, version)!r}"
        )
    return UnsignedTransaction(
        chain_id=chain_id,
        from_address=from_address,
        to_address=to_address,
        protocol=protocol,
        version=version,
        function_name=function_name,
        args=args,
        gas_limit=gas_limit,
        max_fee_per_gas=max_fee_per_gas,
        max_priority_fee_per_gas=max_priority_fee_per_gas,
        nonce=nonce,
    )


@dataclass(frozen=True, slots=True)
class TxReceipt:
    tx_hash: str
    broadcast_ok: bool


@runtime_checkable
class RpcClient(Protocol):
    connected: bool

    def simulate(self, tx: UnsignedTransaction) -> tuple[bool, str | None]: ...
    def broadcast(self, signed_blob: bytes) -> str: ...


@runtime_checkable
class SigningClient(Protocol):
    def sign(self, tx: UnsignedTransaction, *, signer_role: str) -> bytes: ...


@dataclass
class FakeRpcClient:
    simulate_ok: bool
    broadcast_hash: str = "0xfake"
    simulate_reason: str | None = None
    connected: bool = True

    def simulate(self, tx: UnsignedTransaction) -> tuple[bool, str | None]:
        return (self.simulate_ok, self.simulate_reason)

    def broadcast(self, signed_blob: bytes) -> str:
        return self.broadcast_hash


@dataclass
class FakeSigningClient:
    signature: str
    allowed_roles: tuple[str, ...] = ("trading_signer",)

    def sign(self, tx: UnsignedTransaction, *, signer_role: str) -> bytes:
        if signer_role not in self.allowed_roles:
            raise PermissionError(
                f"signer_role {signer_role!r} not in allowed roles {self.allowed_roles}"
            )
        return self.signature.encode("utf-8")


@dataclass
class Web3Gateway:
    rpc: RpcClient
    signer: SigningClient
    gas_estimator: Callable[[UnsignedTransaction], int]
    _broadcast_count: int = field(default=0, init=False, repr=False)

    def execute(
        self, tx: UnsignedTransaction, *, signer_role: str
    ) -> TxReceipt:
        ok, reason = self.rpc.simulate(tx)
        if not ok:
            raise RuntimeError(f"simulation failed: {reason}")
        # Honour gas estimator (the caller can re-price before signing).
        _ = self.gas_estimator(tx)
        signed = self.signer.sign(tx, signer_role=signer_role)
        tx_hash = self.rpc.broadcast(signed)
        self._broadcast_count += 1
        return TxReceipt(tx_hash=tx_hash, broadcast_ok=True)

    def health(self) -> HealthStatus:
        connected = bool(getattr(self.rpc, "connected", False))
        checks = {"rpc_connected": connected}
        return HealthStatus(
            service="gateways.web3",
            ok=all(checks.values()),
            checks=checks,
        )
