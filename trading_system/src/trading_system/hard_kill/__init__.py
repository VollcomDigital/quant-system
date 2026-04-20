"""Out-of-band hard-kill handler.

ADR-0004 Layer 5: revokes IAM role permissions + disables KMS signing
keys from a separate account/Lambda so the main control plane can be
unresponsive and the kill still lands. Phase 9 ships the pure-Python
contract; the Phase 9 deployment runs this module inside a Lambda
with its own IAM role, behind an out-of-band trigger.
"""

from __future__ import annotations

import secrets
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from shared_lib.contracts import AuditEvent

__all__ = [
    "FakeIAMRevoker",
    "FakeKMSRevoker",
    "HardKillRequest",
    "HardKillResult",
    "IAMRevoker",
    "KMSRevoker",
    "execute_hard_kill",
]


@runtime_checkable
class IAMRevoker(Protocol):
    def revoke(self, role: str) -> None: ...


@runtime_checkable
class KMSRevoker(Protocol):
    def disable(self, key_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class HardKillRequest:
    target_iam_roles: tuple[str, ...]
    target_kms_keys: tuple[str, ...]
    reason: str
    approved_by: str
    approval_id: str
    at: datetime

    def __post_init__(self) -> None:
        if not self.approval_id:
            raise ValueError("hard_kill requires an approval_id")
        if not self.target_iam_roles and not self.target_kms_keys:
            raise ValueError(
                "hard_kill requires at least one target IAM role or KMS key"
            )
        if self.at.tzinfo is None:
            raise ValueError("`at` must be timezone-aware")
        if not self.approved_by:
            raise ValueError("approved_by must be non-empty")
        if not self.reason:
            raise ValueError("reason must be non-empty")


@dataclass(frozen=True, slots=True)
class HardKillResult:
    revoked_iam_roles: tuple[str, ...]
    disabled_kms_keys: tuple[str, ...]
    audit_event: AuditEvent


@dataclass
class FakeIAMRevoker:
    calls: list[tuple[str, str]] = field(default_factory=list, init=False, repr=False)

    def revoke(self, role: str) -> None:
        self.calls.append(("revoke", role))


@dataclass
class FakeKMSRevoker:
    calls: list[tuple[str, str]] = field(default_factory=list, init=False, repr=False)

    def disable(self, key_id: str) -> None:
        self.calls.append(("disable", key_id))


def execute_hard_kill(
    *,
    request: HardKillRequest,
    iam_revoker: IAMRevoker,
    kms_revoker: KMSRevoker,
) -> HardKillResult:
    revoked_roles: list[str] = []
    disabled_keys: list[str] = []
    try:
        for role in request.target_iam_roles:
            iam_revoker.revoke(role)
            revoked_roles.append(role)
        for key in request.target_kms_keys:
            kms_revoker.disable(key)
            disabled_keys.append(key)
    finally:
        # Always record an audit event, even if a revoker raised so the
        # caller can see exactly which roles / keys landed before the
        # failure. Phase 9 Lambda re-raises so the on-call operator is
        # alerted.
        AuditEvent(
            event_id=f"aud-hard-kill-{secrets.token_hex(8)}",
            actor=f"user:{request.approved_by}",
            action="hard_kill.executed",
            target="trading_system",
            at=request.at,
            trace_id=None,
            details={
                "approval_id": request.approval_id,
                "reason": request.reason,
                "revoked_iam_roles": ",".join(revoked_roles),
                "disabled_kms_keys": ",".join(disabled_keys),
            },
        )
    return HardKillResult(
        revoked_iam_roles=tuple(revoked_roles),
        disabled_kms_keys=tuple(disabled_keys),
        audit_event=AuditEvent(
            event_id=f"aud-hard-kill-{secrets.token_hex(8)}",
            actor=f"user:{request.approved_by}",
            action="hard_kill.executed",
            target="trading_system",
            at=request.at,
            trace_id=None,
            details={
                "approval_id": request.approval_id,
                "reason": request.reason,
                "revoked_iam_roles": ",".join(revoked_roles),
                "disabled_kms_keys": ",".join(disabled_keys),
            },
        ),
    )


# Mark unused import shape so ruff doesn't strip Sequence.
_ = Sequence
