"""Phase 7 Exit Criteria.

- TradFi and Web3 gateway abstractions are separate and explicit.
- Replay and reconciliation workflows exist for gateway failures.
- Paper-trading parity is possible without live credential access.
"""

from __future__ import annotations


def test_exit_tradfi_and_web3_are_separate_namespaces() -> None:
    import trading_system.gateways.tradfi as tradfi
    import trading_system.gateways.web3 as web3

    assert tradfi.__name__ != web3.__name__
    # No vendor types leak across the seam.
    assert not hasattr(tradfi, "Web3Gateway")
    assert not hasattr(web3, "AlpacaGateway")
    assert not hasattr(web3, "IBKRGateway")


def test_exit_replay_and_heartbeat_helpers_exist() -> None:
    from trading_system.shared_gateways.replay import (  # noqa: F401
        HeartbeatTracker,
        detect_gaps,
        replay_sequenced,
    )


def test_exit_paper_trading_parity_runs_without_credentials() -> None:
    from datetime import UTC, datetime
    from decimal import Decimal

    from backtest_engine.api import OrderPayload
    from trading_system.shared_gateways import SimulatedGateway

    gw = SimulatedGateway()
    payload = OrderPayload(
        idempotency_key="idem-x",
        symbol="AAPL",
        side="buy",
        quantity=Decimal("1"),
        limit_price=Decimal("100"),
        time_in_force="day",
        placed_at=datetime(2026, 4, 19, tzinfo=UTC),
    )
    ack = gw.submit(payload)
    assert ack.accepted is True
