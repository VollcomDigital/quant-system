"""Phase 6 execution-oversight API contracts + handlers.

Read-only status + bounded mutations: halt trading, reset kill switch,
trigger panic playbook. No submit-order / place-order endpoints. No
broker credentials in the browser. Every mutation requires the
`operator` role and emits an audit event via the KillSwitch.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import Field
from shared_lib.contracts._base import Schema, aware_datetime_validator
from trading_system.kill_switch import KillSwitch

__all__ = [
    "ExecutionStatusResponse",
    "HaltTradingRequest",
    "handle_halt_trading",
]


class ExecutionStatusResponse(Schema):
    trading_halted: bool
    open_orders: int = Field(ge=0)
    positions: dict[str, Decimal]
    daily_pnl_pct: Decimal
    last_heartbeat: datetime

    _ts = aware_datetime_validator("last_heartbeat")


class HaltTradingRequest(Schema):
    reason: str = Field(min_length=1)
    actor: str = Field(min_length=1)
    at: datetime

    _ts = aware_datetime_validator("at")


def handle_halt_trading(
    *,
    request: HaltTradingRequest,
    authenticated_user: str | None,
    user_roles: tuple[str, ...],
    kill_switch: KillSwitch,
) -> None:
    if authenticated_user is None:
        raise PermissionError("authentication required")
    if "operator" not in user_roles:
        raise PermissionError("operator role required to halt trading")
    kill_switch.trigger(reason=request.reason, actor=request.actor, at=request.at)
