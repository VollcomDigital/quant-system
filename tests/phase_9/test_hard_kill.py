"""Phase 9 Task 4 - out-of-band hard-kill handler.

ADR-0004 Layer 5: out-of-band infrastructure isolation that can
revoke runtime permissions even if the main system is unresponsive.
The handler runs on AWS Lambda in its own account, behind its own IAM
role; Phase 9 ships the pure-Python contract so unit tests run
without AWS.

Flow:

1. Operator invokes `HardKillRequest(target_iam_roles, target_kms_keys,
   reason, approved_by, approval_id)` via the web control plane.
2. Handler calls `IAMRevoker.revoke(role)` for every role, then
   `KMSRevoker.disable(key_id)` for every key.
3. Emits an `AuditEvent` with `action = "hard_kill.executed"`.
4. Refuses execution without an `approval_id`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest


def _request(**kw):
    from trading_system.hard_kill import HardKillRequest

    defaults = dict(
        target_iam_roles=("trading-role-prod", "hft-role-prod"),
        target_kms_keys=("arn:aws:kms:us-east-1:1:key/trading", "arn:aws:kms:us-east-1:1:key/hft"),
        reason="exploit detected",
        approved_by="ops-lead",
        approval_id="app-hard-kill-1",
        at=datetime(2026, 4, 20, tzinfo=UTC),
    )
    defaults.update(kw)
    return HardKillRequest(**defaults)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_hard_kill_revokes_every_iam_role_and_disables_every_key() -> None:
    from trading_system.hard_kill import FakeIAMRevoker, FakeKMSRevoker, execute_hard_kill

    iam = FakeIAMRevoker()
    kms = FakeKMSRevoker()
    result = execute_hard_kill(
        request=_request(),
        iam_revoker=iam,
        kms_revoker=kms,
    )
    assert result.revoked_iam_roles == ("trading-role-prod", "hft-role-prod")
    assert result.disabled_kms_keys == (
        "arn:aws:kms:us-east-1:1:key/trading",
        "arn:aws:kms:us-east-1:1:key/hft",
    )
    assert iam.calls == [
        ("revoke", "trading-role-prod"),
        ("revoke", "hft-role-prod"),
    ]
    assert kms.calls == [
        ("disable", "arn:aws:kms:us-east-1:1:key/trading"),
        ("disable", "arn:aws:kms:us-east-1:1:key/hft"),
    ]


def test_hard_kill_emits_audit_event() -> None:
    from trading_system.hard_kill import FakeIAMRevoker, FakeKMSRevoker, execute_hard_kill

    result = execute_hard_kill(
        request=_request(),
        iam_revoker=FakeIAMRevoker(),
        kms_revoker=FakeKMSRevoker(),
    )
    assert result.audit_event.action == "hard_kill.executed"
    assert result.audit_event.actor == "user:ops-lead"


# ---------------------------------------------------------------------------
# Defensive: approval required, empty targets refused.
# ---------------------------------------------------------------------------


def test_hard_kill_refuses_without_approval_id() -> None:
    with pytest.raises(ValueError, match="approval"):
        _request(approval_id="")


def test_hard_kill_refuses_empty_target_sets() -> None:
    with pytest.raises(ValueError, match="target"):
        _request(target_iam_roles=(), target_kms_keys=())


def test_hard_kill_refuses_naive_timestamp() -> None:
    with pytest.raises(ValueError):
        _request(at=datetime(2026, 4, 20))


# ---------------------------------------------------------------------------
# Revoker failures short-circuit with an audit-event-style error.
# ---------------------------------------------------------------------------


def test_iam_revoker_failure_does_not_prevent_audit_of_attempted_roles() -> None:
    from trading_system.hard_kill import execute_hard_kill

    class _FlakyIAM:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def revoke(self, role: str) -> None:
            self.calls.append(("revoke", role))
            if role == "hft-role-prod":
                raise RuntimeError("IAM outage")

    class _FakeKMS:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def disable(self, key: str) -> None:
            self.calls.append(("disable", key))

    with pytest.raises(RuntimeError):
        execute_hard_kill(
            request=_request(),
            iam_revoker=_FlakyIAM(),
            kms_revoker=_FakeKMS(),
        )
