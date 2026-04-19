"""Phase 7 Task 3 - Web3 gateway: EVM tx construction + signing requests.

The Web3 gateway never holds private keys. Instead it builds an
`UnsignedTransaction`, simulates it via an `RpcClient`, requests a
signature via a `SigningClient` (Phase 9 KMS adapter), and broadcasts
the resulting `SignedTransaction`. ABI-driven calldata encoding goes
through `data_platform.indexing.ABIRegistry`.
"""

from __future__ import annotations

from decimal import Decimal

import pytest


def _registry_with_swap_abi():
    from data_platform.indexing import ABIRegistry

    reg = ABIRegistry()
    reg.register(
        protocol="uniswap_v3",
        version="1.0.0",
        abi=[
            {
                "type": "function",
                "name": "exactInputSingle",
                "inputs": [{"name": "params", "type": "tuple"}],
                "outputs": [{"name": "amountOut", "type": "uint256"}],
            }
        ],
    )
    return reg


# ---------------------------------------------------------------------------
# UnsignedTransaction construction
# ---------------------------------------------------------------------------


def test_build_unsigned_tx_uses_abi_registry() -> None:
    from trading_system.gateways.web3 import build_unsigned_tx

    reg = _registry_with_swap_abi()
    tx = build_unsigned_tx(
        chain_id=1,
        from_address="0xfrom",
        to_address="0xrouter",
        protocol="uniswap_v3",
        version="1.0.0",
        function_name="exactInputSingle",
        args={"params": "0xabcd"},
        gas_limit=200_000,
        max_fee_per_gas=Decimal("50"),
        max_priority_fee_per_gas=Decimal("2"),
        nonce=42,
        abi_registry=reg,
    )
    assert tx.chain_id == 1
    assert tx.to_address == "0xrouter"
    assert tx.function_name == "exactInputSingle"
    assert tx.nonce == 42


def test_build_unsigned_tx_rejects_unknown_function() -> None:
    from trading_system.gateways.web3 import build_unsigned_tx

    reg = _registry_with_swap_abi()
    with pytest.raises(ValueError, match="function"):
        build_unsigned_tx(
            chain_id=1,
            from_address="0x",
            to_address="0x",
            protocol="uniswap_v3",
            version="1.0.0",
            function_name="ghost",
            args={},
            gas_limit=1,
            max_fee_per_gas=Decimal("1"),
            max_priority_fee_per_gas=Decimal("1"),
            nonce=0,
            abi_registry=reg,
        )


def test_build_unsigned_tx_rejects_negative_nonce() -> None:
    from trading_system.gateways.web3 import build_unsigned_tx

    reg = _registry_with_swap_abi()
    with pytest.raises(ValueError):
        build_unsigned_tx(
            chain_id=1,
            from_address="0x",
            to_address="0x",
            protocol="uniswap_v3",
            version="1.0.0",
            function_name="exactInputSingle",
            args={},
            gas_limit=1,
            max_fee_per_gas=Decimal("1"),
            max_priority_fee_per_gas=Decimal("1"),
            nonce=-1,
            abi_registry=reg,
        )


# ---------------------------------------------------------------------------
# Simulation + signing + broadcast
# ---------------------------------------------------------------------------


def test_web3_gateway_full_flow_simulate_sign_broadcast() -> None:
    from trading_system.gateways.web3 import (
        FakeRpcClient,
        FakeSigningClient,
        UnsignedTransaction,
        Web3Gateway,
    )

    rpc = FakeRpcClient(simulate_ok=True, broadcast_hash="0xtxhash")
    signer = FakeSigningClient(signature="0xsig")
    gw = Web3Gateway(rpc=rpc, signer=signer, gas_estimator=lambda _tx: 100_000)

    tx = UnsignedTransaction(
        chain_id=1,
        from_address="0xfrom",
        to_address="0xrouter",
        protocol="uniswap_v3",
        version="1.0.0",
        function_name="exactInputSingle",
        args={"params": "0xabcd"},
        gas_limit=200_000,
        max_fee_per_gas=Decimal("50"),
        max_priority_fee_per_gas=Decimal("2"),
        nonce=42,
    )
    receipt = gw.execute(tx, signer_role="trading_signer")
    assert receipt.tx_hash == "0xtxhash"
    assert receipt.broadcast_ok is True


def test_web3_gateway_refuses_when_simulation_fails() -> None:
    from trading_system.gateways.web3 import (
        FakeRpcClient,
        FakeSigningClient,
        UnsignedTransaction,
        Web3Gateway,
    )

    rpc = FakeRpcClient(simulate_ok=False, simulate_reason="reverted: stale price")
    gw = Web3Gateway(rpc=rpc, signer=FakeSigningClient(signature="0x"), gas_estimator=lambda _tx: 1)
    tx = UnsignedTransaction(
        chain_id=1,
        from_address="0x",
        to_address="0x",
        protocol="uniswap_v3",
        version="1.0.0",
        function_name="exactInputSingle",
        args={},
        gas_limit=1,
        max_fee_per_gas=Decimal("1"),
        max_priority_fee_per_gas=Decimal("1"),
        nonce=0,
    )
    with pytest.raises(RuntimeError, match="simulation"):
        gw.execute(tx, signer_role="trading_signer")


def test_web3_gateway_refuses_signer_without_role() -> None:
    from trading_system.gateways.web3 import (
        FakeRpcClient,
        FakeSigningClient,
        UnsignedTransaction,
        Web3Gateway,
    )

    rpc = FakeRpcClient(simulate_ok=True, broadcast_hash="0xa")
    signer = FakeSigningClient(signature="0x", allowed_roles=("treasury_signer",))
    gw = Web3Gateway(rpc=rpc, signer=signer, gas_estimator=lambda _tx: 1)
    tx = UnsignedTransaction(
        chain_id=1, from_address="0x", to_address="0x", protocol="p",
        version="v", function_name="f", args={}, gas_limit=1,
        max_fee_per_gas=Decimal("1"), max_priority_fee_per_gas=Decimal("1"),
        nonce=0,
    )
    with pytest.raises(PermissionError):
        gw.execute(tx, signer_role="trading_signer")


def test_web3_gateway_health_reports_rpc_disconnect() -> None:
    from trading_system.gateways.web3 import FakeRpcClient, FakeSigningClient, Web3Gateway

    rpc = FakeRpcClient(simulate_ok=True, broadcast_hash="0x", connected=False)
    gw = Web3Gateway(rpc=rpc, signer=FakeSigningClient(signature="0x"), gas_estimator=lambda _: 1)
    assert gw.health().ok is False
