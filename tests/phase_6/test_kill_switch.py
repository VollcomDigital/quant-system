"""Phase 6 Task 6 - trading_system.kill_switch panic-button playbooks.

Multi-layer kill-switch (ADR-0004 Layers 2/3):

- `KillSwitch.trigger(reason, actor)` sets `TRADING_HALTED` and emits
  an `AuditEvent`.
- `KillSwitch.reset(approval_id, actor)` flips the flag off only if an
  approval for `subject='kill_switch_reset'` has been recorded.
- `PanicPlaybook` models the TradFi + DeFi flow: halt new AI signals,
  cancel all working orders, optionally flatten or delta-hedge.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

# ---------------------------------------------------------------------------
# Kill-switch state
# ---------------------------------------------------------------------------


def test_kill_switch_default_is_disarmed() -> None:
    from trading_system.kill_switch import KillSwitch

    ks = KillSwitch()
    assert ks.trading_halted is False


def test_kill_switch_trigger_sets_flag_and_emits_audit() -> None:
    from trading_system.kill_switch import KillSwitch

    ks = KillSwitch()
    ks.trigger(
        reason="drawdown breach",
        actor="risk_monitor",
        at=datetime(2026, 4, 19, tzinfo=UTC),
    )
    assert ks.trading_halted is True
    audits = ks.audit_log()
    assert len(audits) == 1
    assert audits[0].action == "kill_switch.triggered"


def test_kill_switch_reset_requires_approval() -> None:
    from trading_system.kill_switch import KillSwitch

    ks = KillSwitch()
    ks.trigger(reason="x", actor="operator", at=datetime(2026, 4, 19, tzinfo=UTC))
    with pytest.raises(PermissionError):
        ks.reset(approval_id="", actor="operator", at=datetime(2026, 4, 19, tzinfo=UTC))
    ks.reset(
        approval_id="app-123",
        actor="operator",
        at=datetime(2026, 4, 19, tzinfo=UTC),
    )
    assert ks.trading_halted is False
    assert [a.action for a in ks.audit_log()] == [
        "kill_switch.triggered",
        "kill_switch.reset",
    ]


def test_kill_switch_trigger_when_already_halted_is_idempotent() -> None:
    from trading_system.kill_switch import KillSwitch

    ks = KillSwitch()
    ks.trigger(reason="a", actor="o", at=datetime(2026, 4, 19, tzinfo=UTC))
    ks.trigger(reason="b", actor="o", at=datetime(2026, 4, 19, tzinfo=UTC))
    assert ks.trading_halted is True
    # Only the first trigger produces an audit event; subsequent calls
    # are no-ops so the audit log isn't spammed.
    assert len(ks.audit_log()) == 1


# ---------------------------------------------------------------------------
# Panic playbook
# ---------------------------------------------------------------------------


def test_panic_playbook_halts_ai_and_cancels_orders() -> None:
    from trading_system.kill_switch import KillSwitch, PanicPlaybook

    ks = KillSwitch()
    playbook = PanicPlaybook(kill_switch=ks)

    cancelled: list[str] = []

    def cancel_all():
        cancelled.extend(["o-1", "o-2"])
        return ("o-1", "o-2")

    result = playbook.execute(
        reason="vol spike",
        actor="operator",
        at=datetime(2026, 4, 19, tzinfo=UTC),
        cancel_all_orders=cancel_all,
    )
    assert ks.trading_halted is True
    assert result.cancelled_orders == ("o-1", "o-2")
    assert result.ai_signal_intake_halted is True
    assert "cancel" in [a.action for a in ks.audit_log()] or any(
        a.action == "kill_switch.triggered" for a in ks.audit_log()
    )


def test_panic_playbook_result_records_actor() -> None:
    from trading_system.kill_switch import KillSwitch, PanicPlaybook

    ks = KillSwitch()
    playbook = PanicPlaybook(kill_switch=ks)
    result = playbook.execute(
        reason="r",
        actor="alice",
        at=datetime(2026, 4, 19, tzinfo=UTC),
        cancel_all_orders=lambda: (),
    )
    assert result.actor == "alice"


# ---------------------------------------------------------------------------
# Defensive: the kill switch enforces tz-aware timestamps.
# ---------------------------------------------------------------------------


def test_trigger_rejects_naive_timestamp() -> None:
    from trading_system.kill_switch import KillSwitch

    ks = KillSwitch()
    with pytest.raises(ValueError, match="aware"):
        ks.trigger(reason="x", actor="o", at=datetime(2026, 4, 19))


# Suppress unused import warnings for pytest-cov.
_ = Decimal
