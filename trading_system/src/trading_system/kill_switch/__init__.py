"""Kill switch + panic-button playbooks.

ADR-0004 Layers 2+3: deterministic pre-trade risk + automated
panic-button workflow. `KillSwitch` holds the global
`TRADING_HALTED` flag and emits `AuditEvent`s for every mutation.
`PanicPlaybook` orchestrates the flow: halt AI signal intake, cancel
all working orders, and (optionally) flatten positions. The playbook
never calls brokers directly - that lives in Phase 7 gateway adapters.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from shared_lib.contracts import AuditEvent

__all__ = ["KillSwitch", "PanicPlaybook", "PlaybookResult"]


@dataclass
class KillSwitch:
    trading_halted: bool = False
    _audit: list[AuditEvent] = field(default_factory=list, init=False, repr=False)

    def trigger(self, *, reason: str, actor: str, at: datetime) -> None:
        if at.tzinfo is None:
            raise ValueError("`at` must be timezone-aware")
        if self.trading_halted:
            return  # idempotent
        self.trading_halted = True
        self._audit.append(
            AuditEvent(
                event_id=f"aud-{secrets.token_hex(8)}",
                actor=f"system:{actor}",
                action="kill_switch.triggered",
                target="trading_system",
                at=at,
                trace_id=None,
                details={"reason": reason},
            )
        )

    def reset(self, *, approval_id: str, actor: str, at: datetime) -> None:
        if not approval_id:
            raise PermissionError(
                "kill_switch reset requires an approval_id"
            )
        if at.tzinfo is None:
            raise ValueError("`at` must be timezone-aware")
        self.trading_halted = False
        self._audit.append(
            AuditEvent(
                event_id=f"aud-{secrets.token_hex(8)}",
                actor=f"user:{actor}",
                action="kill_switch.reset",
                target="trading_system",
                at=at,
                trace_id=None,
                details={"approval_id": approval_id},
            )
        )

    def audit_log(self) -> list[AuditEvent]:
        return list(self._audit)


@dataclass(frozen=True, slots=True)
class PlaybookResult:
    actor: str
    ai_signal_intake_halted: bool
    cancelled_orders: tuple[str, ...]


@dataclass
class PanicPlaybook:
    kill_switch: KillSwitch

    def execute(
        self,
        *,
        reason: str,
        actor: str,
        at: datetime,
        cancel_all_orders: Callable[[], tuple[str, ...]],
    ) -> PlaybookResult:
        self.kill_switch.trigger(reason=reason, actor=actor, at=at)
        cancelled = tuple(cancel_all_orders())
        return PlaybookResult(
            actor=actor,
            ai_signal_intake_halted=True,
            cancelled_orders=cancelled,
        )
