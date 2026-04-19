"""Phase 5 Task 9 - web_control_plane approval console contract.

Phase 5 ships a scope doc + typed request/response models that the
Phase 9 FastAPI backend will implement. No live HTTP server yet; the
Phase 4 web-shell doc covers read-only browsing, and Phase 5 adds the
approval mutating endpoints which must all require authentication and
emit audit events.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

DOC_PATH = Path("docs") / "architecture" / "web-control-plane-phase-5.md"


REQUIRED_SECTIONS = (
    "## Purpose",
    "## In Scope for Phase 5",
    "## Out of Scope",
    "## Authentication and RBAC",
    "## Mutating Endpoints",
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


def test_doc_addresses_audit_requirement(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    assert "audit" in text and "auditevent" in text


def test_doc_explicitly_forbids_broker_credentials_in_browser(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    assert "browser" in text and (
        "never holds broker credentials" in text
        or "no broker credentials in the browser" in text
    )


# ---------------------------------------------------------------------------
# API request/response contracts
# ---------------------------------------------------------------------------


def test_submit_approval_request_rejects_unauthenticated_user() -> None:
    from ai_agents.approvals import ApprovalQueue
    from web_control_plane.backend.api import (
        SubmitApprovalRequest,
        handle_submit_approval,
    )

    queue = ApprovalQueue()
    req = SubmitApprovalRequest(
        approval_id="app-1",
        subject="factor_promotion",
        target_id="mom_12_1",
        requested_by="alice",
        requested_at=datetime(2026, 4, 19, tzinfo=UTC),
        context={"k": "v"},
    )
    with pytest.raises(PermissionError):
        handle_submit_approval(
            request=req,
            authenticated_user=None,
            approval_queue=queue,
        )


def test_submit_approval_happy_path_emits_audit() -> None:
    from ai_agents.approvals import ApprovalQueue
    from web_control_plane.backend.api import (
        SubmitApprovalRequest,
        handle_submit_approval,
    )

    queue = ApprovalQueue()
    req = SubmitApprovalRequest(
        approval_id="app-2",
        subject="factor_promotion",
        target_id="mom_12_1",
        requested_by="alice",
        requested_at=datetime(2026, 4, 19, tzinfo=UTC),
        context={},
    )
    handle_submit_approval(
        request=req,
        authenticated_user="alice",
        approval_queue=queue,
    )
    assert len(queue.audit_log()) == 1


def test_decide_approval_rejects_without_approver_role() -> None:
    from ai_agents.approvals import ApprovalQueue
    from web_control_plane.backend.api import (
        DecideApprovalRequest,
        handle_decide_approval,
    )

    queue = ApprovalQueue()
    with pytest.raises(PermissionError):
        handle_decide_approval(
            request=DecideApprovalRequest(
                approval_id="app-x",
                decision="approved",
                decided_by="alice",
                decided_at=datetime(2026, 4, 19, tzinfo=UTC),
                notes="",
            ),
            authenticated_user="alice",
            user_roles=("viewer",),  # not approver
            approval_queue=queue,
        )


def test_request_models_reject_naive_timestamps() -> None:
    from web_control_plane.backend.api import SubmitApprovalRequest

    with pytest.raises(ValueError):
        SubmitApprovalRequest(
            approval_id="x",
            subject="factor_promotion",
            target_id="y",
            requested_by="alice",
            requested_at=datetime(2026, 4, 19),
            context={},
        )
