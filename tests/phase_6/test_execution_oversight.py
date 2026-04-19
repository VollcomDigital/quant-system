"""Phase 6 Task 7 - web_control_plane execution-oversight API.

Read-only status endpoints + two mutating endpoints for halt / panic.
Every mutating endpoint requires authentication + `operator` role and
produces audit events. No direct trade entry. Browser never holds
broker credentials.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

DOC_PATH = Path("docs") / "architecture" / "web-control-plane-phase-6.md"


REQUIRED_SECTIONS = (
    "## Purpose",
    "## In Scope for Phase 6",
    "## Out of Scope",
    "## Authentication and RBAC",
    "## Read-Only Status Endpoints",
    "## Bounded Mutating Endpoints",
    "## Audit Requirements",
    "## Enforcement",
)


def _read(repo_root: Path) -> str:
    path = repo_root / DOC_PATH
    assert path.is_file(), f"Doc missing at {DOC_PATH}"
    return path.read_text(encoding="utf-8")


def test_doc_has_required_sections(repo_root: Path) -> None:
    text = _read(repo_root)
    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    assert not missing, f"Doc missing sections: {missing}"


def test_doc_explicitly_forbids_raw_trade_entry(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    assert "raw browser-driven trade entry" in text or "no trade entry" in text
    assert "operator" in text


# ---------------------------------------------------------------------------
# ExecutionStatusResponse
# ---------------------------------------------------------------------------


def test_execution_status_response_serialises() -> None:
    from web_control_plane.backend.api.execution import ExecutionStatusResponse

    r = ExecutionStatusResponse(
        trading_halted=False,
        open_orders=3,
        positions={"AAPL": Decimal("10")},
        daily_pnl_pct=Decimal("0.01"),
        last_heartbeat=datetime(2026, 4, 19, 14, tzinfo=UTC),
    )
    assert r.open_orders == 3


# ---------------------------------------------------------------------------
# handle_halt_trading
# ---------------------------------------------------------------------------


def test_handle_halt_trading_requires_operator_role() -> None:
    from trading_system.kill_switch import KillSwitch
    from web_control_plane.backend.api.execution import (
        HaltTradingRequest,
        handle_halt_trading,
    )

    ks = KillSwitch()
    req = HaltTradingRequest(
        reason="vol spike",
        actor="alice",
        at=datetime(2026, 4, 19, tzinfo=UTC),
    )
    with pytest.raises(PermissionError):
        handle_halt_trading(
            request=req,
            authenticated_user="alice",
            user_roles=("viewer",),
            kill_switch=ks,
        )


def test_handle_halt_trading_happy_path_sets_flag() -> None:
    from trading_system.kill_switch import KillSwitch
    from web_control_plane.backend.api.execution import (
        HaltTradingRequest,
        handle_halt_trading,
    )

    ks = KillSwitch()
    handle_halt_trading(
        request=HaltTradingRequest(
            reason="halt",
            actor="alice",
            at=datetime(2026, 4, 19, tzinfo=UTC),
        ),
        authenticated_user="alice",
        user_roles=("operator",),
        kill_switch=ks,
    )
    assert ks.trading_halted is True


def test_handle_halt_trading_rejects_unauthenticated() -> None:
    from trading_system.kill_switch import KillSwitch
    from web_control_plane.backend.api.execution import (
        HaltTradingRequest,
        handle_halt_trading,
    )

    with pytest.raises(PermissionError):
        handle_halt_trading(
            request=HaltTradingRequest(
                reason="x",
                actor="a",
                at=datetime(2026, 4, 19, tzinfo=UTC),
            ),
            authenticated_user=None,
            user_roles=("operator",),
            kill_switch=KillSwitch(),
        )


# ---------------------------------------------------------------------------
# No trade entry
# ---------------------------------------------------------------------------


def test_execution_api_has_no_submit_order_endpoint() -> None:
    import web_control_plane.backend.api.execution as ex_api

    # The module must not expose any handler that creates/submits an
    # order. This prevents a future PR from smuggling one in.
    for name in dir(ex_api):
        lowered = name.lower()
        assert "submit_order" not in lowered
        assert "place_order" not in lowered
