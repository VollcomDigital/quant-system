"""Phase 10 Task 3 - OMS/EMS -> gateways contract test.

Chain under test:

1. Phase 6 `EMS.schedule` slices a parent `Order`.
2. `OrderRouter.to_payload` projects each child to the Phase 4
   `OrderPayload` wire shape that every Phase 7 gateway consumes.
3. The Phase 7 `SimulatedGateway` and `AlpacaGateway` both satisfy the
   `Gateway` Protocol and accept the same payload.
4. The Phase 7 `Web3Gateway` composes orthogonally: a DeFi-flavoured
   `UnsignedTransaction` goes through `build_unsigned_tx` +
   `simulate -> sign -> broadcast` without touching the TradFi seam.
5. The Phase 6 `PanicPlaybook` cancels every open gateway order when
   triggered, binding the OMS/EMS -> gateway path to the kill-switch
   escalation path.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal


def _parent_order() -> object:
    from shared_lib.contracts import Order

    return Order(
        order_id="p-1",
        idempotency_key="idem-p-1",
        symbol="AAPL",
        side="buy",
        quantity=Decimal("20"),
        limit_price=Decimal("150"),
        time_in_force="day",
        placed_at=datetime(2026, 4, 20, tzinfo=UTC),
    )


def test_ems_child_payloads_satisfy_gateway_protocol_on_simulated_gateway() -> None:
    from trading_system.ems import EMS, EqualSliceSchedule, OrderRouter
    from trading_system.shared_gateways import Gateway, SimulatedGateway

    ems = EMS()
    router = OrderRouter()
    children = ems.schedule(_parent_order(), slicer=EqualSliceSchedule(num_slices=4))
    gateway: Gateway = SimulatedGateway()
    assert isinstance(gateway, Gateway)
    acks = [gateway.submit(router.to_payload(child)) for child in children]
    assert all(ack.accepted for ack in acks)
    assert len(list(gateway.open_orders())) == 4


def test_ems_children_route_identically_to_alpaca_gateway() -> None:
    """The TradFi AlpacaGateway satisfies the same `Gateway` Protocol."""
    from trading_system.ems import EMS, EqualSliceSchedule, OrderRouter
    from trading_system.gateways.tradfi import AlpacaGateway, FakeBrokerClient
    from trading_system.shared_gateways import Gateway

    ems = EMS()
    router = OrderRouter()
    children = ems.schedule(_parent_order(), slicer=EqualSliceSchedule(num_slices=2))
    gateway = AlpacaGateway(client=FakeBrokerClient())
    assert isinstance(gateway, Gateway)
    for child in children:
        ack = gateway.submit(router.to_payload(child))
        assert ack.accepted
    assert len(list(gateway.open_orders())) == 2


def test_web3_gateway_roundtrips_build_sign_broadcast() -> None:
    """Web3 seam composes orthogonally to the TradFi seam."""
    from data_platform.indexing import ABIRegistry
    from trading_system.gateways.web3 import (
        FakeRpcClient,
        FakeSigningClient,
        Web3Gateway,
        build_unsigned_tx,
    )

    registry = ABIRegistry()
    registry.register(
        protocol="uniswap",
        version="v3",
        abi=[{"type": "function", "name": "swap"}],
    )
    tx = build_unsigned_tx(
        chain_id=1,
        from_address="0xabc",
        to_address="0xdef",
        protocol="uniswap",
        version="v3",
        function_name="swap",
        args={"tokenIn": "USDC", "tokenOut": "WETH", "amountIn": "1000"},
        gas_limit=250_000,
        max_fee_per_gas=Decimal("30"),
        max_priority_fee_per_gas=Decimal("2"),
        nonce=0,
        abi_registry=registry,
    )
    gw = Web3Gateway(
        rpc=FakeRpcClient(simulate_ok=True, broadcast_hash="0x1234"),
        signer=FakeSigningClient(signature="sig"),
        gas_estimator=lambda _t: 250_000,
    )
    receipt = gw.execute(tx, signer_role="trading_signer")
    assert receipt.tx_hash == "0x1234"
    assert receipt.broadcast_ok is True


def test_panic_playbook_cancels_every_open_gateway_order() -> None:
    """Kill-switch escalation cancels orders that EMS routed to gateway."""
    from trading_system.ems import EMS, EqualSliceSchedule, OrderRouter
    from trading_system.kill_switch import KillSwitch, PanicPlaybook
    from trading_system.shared_gateways import SimulatedGateway

    ems = EMS()
    router = OrderRouter()
    gateway = SimulatedGateway()
    for child in ems.schedule(_parent_order(), slicer=EqualSliceSchedule(num_slices=3)):
        gateway.submit(router.to_payload(child))
    assert len(list(gateway.open_orders())) == 3

    ks = KillSwitch()
    playbook = PanicPlaybook(kill_switch=ks)
    result = playbook.execute(
        reason="ops drill",
        actor="ops-lead",
        at=datetime(2026, 4, 20, tzinfo=UTC),
        cancel_all_orders=gateway.cancel_all,
    )
    assert ks.trading_halted is True
    assert result.ai_signal_intake_halted is True
    assert len(result.cancelled_orders) == 3
    assert list(gateway.open_orders()) == []


def test_gateway_health_reports_simulated_disconnect() -> None:
    """The gateway health surface is the seam between Phase 7 adapters
    and the Phase 5 web control plane / Phase 9 observability stack."""
    from trading_system.shared_gateways import SimulatedGateway

    gateway = SimulatedGateway()
    assert gateway.health().ok is True
    gateway.simulate_disconnect()
    assert gateway.health().ok is False
