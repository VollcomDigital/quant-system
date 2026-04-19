"""Phase 7 Task 4 - DeFi kill-switch controls (ADR-0004 Layer 4).

Three controls:

- `request_pause(target_contract)` - emits a Pausable / Safe-module
  pause() request that the Phase 9 signer executes.
- `request_revoke_allowances(token, spenders)` - emits ERC-20
  approve(spender, 0) requests for each spender.
- `ProtocolDenylist` - in-memory deny set the Web3Gateway can consult
  before broadcasting.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Pause request
# ---------------------------------------------------------------------------


def test_request_pause_returns_unsigned_tx_envelope() -> None:
    from trading_system.gateways.defi import request_pause

    req = request_pause(
        chain_id=1,
        from_address="0xops",
        target_contract="0xpool",
        nonce=0,
        gas_limit=80_000,
    )
    assert req.to_address == "0xpool"
    assert req.function_name == "pause"
    assert req.protocol == "pausable"


def test_request_pause_rejects_negative_gas() -> None:
    from trading_system.gateways.defi import request_pause

    with pytest.raises(ValueError):
        request_pause(
            chain_id=1,
            from_address="0xops",
            target_contract="0xpool",
            nonce=0,
            gas_limit=0,
        )


# ---------------------------------------------------------------------------
# Revoke allowances
# ---------------------------------------------------------------------------


def test_request_revoke_allowances_emits_one_tx_per_spender() -> None:
    from trading_system.gateways.defi import request_revoke_allowances

    reqs = request_revoke_allowances(
        chain_id=1,
        from_address="0xops",
        token_address="0xUSDC",
        spenders=("0xrouter1", "0xrouter2", "0xrouter3"),
        nonce_start=10,
        gas_limit=60_000,
    )
    assert len(reqs) == 3
    nonces = [r.nonce for r in reqs]
    assert nonces == [10, 11, 12]
    # Every request targets the token contract with approve(spender, 0).
    for r in reqs:
        assert r.to_address == "0xUSDC"
        assert r.function_name == "approve"
        assert r.args["amount"] == "0"


def test_request_revoke_allowances_refuses_empty_spender_set() -> None:
    from trading_system.gateways.defi import request_revoke_allowances

    with pytest.raises(ValueError):
        request_revoke_allowances(
            chain_id=1,
            from_address="0x",
            token_address="0x",
            spenders=(),
            nonce_start=0,
            gas_limit=1,
        )


# ---------------------------------------------------------------------------
# Protocol denylist
# ---------------------------------------------------------------------------


def test_protocol_denylist_blocks_listed_protocol() -> None:
    from trading_system.gateways.defi import ProtocolDenylist

    deny = ProtocolDenylist()
    deny.add("uniswap_v3", reason="oracle exploit drill")
    assert deny.is_blocked("uniswap_v3") is True
    assert deny.is_blocked("aave_v3") is False
    assert deny.reason("uniswap_v3") == "oracle exploit drill"


def test_protocol_denylist_remove_unblocks() -> None:
    from trading_system.gateways.defi import ProtocolDenylist

    deny = ProtocolDenylist()
    deny.add("aave_v3", reason="x")
    deny.remove("aave_v3")
    assert deny.is_blocked("aave_v3") is False


def test_protocol_denylist_reason_for_unknown_returns_none() -> None:
    from trading_system.gateways.defi import ProtocolDenylist

    assert ProtocolDenylist().reason("never_listed") is None


def test_protocol_denylist_add_requires_reason() -> None:
    from trading_system.gateways.defi import ProtocolDenylist

    with pytest.raises(ValueError, match="reason"):
        ProtocolDenylist().add("uniswap_v3", reason="")
