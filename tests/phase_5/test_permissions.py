"""Phase 5 Task 3 - ai_agents.permissions scoped permission model.

ADR-0004 says agents must receive only the minimum API + file access
needed for each workflow. Permissions are strings (e.g. `factor_library.read`,
`backtest_engine.run`, `notifications.slack.post`) with optional resource
scopes (e.g. `factor_library.read:mom_12_1`).

Forbidden surfaces are explicit: no agent may declare
`oms.submit_order`, `kms.sign`, `treasury.transfer`, `src.*`.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# PermissionScope
# ---------------------------------------------------------------------------


def test_permission_scope_parses_name_and_resource() -> None:
    from ai_agents.permissions import PermissionScope

    p = PermissionScope.parse("factor_library.read:mom_12_1")
    assert p.namespace == "factor_library"
    assert p.action == "read"
    assert p.resource == "mom_12_1"


def test_permission_scope_allows_wildcard_resource() -> None:
    from ai_agents.permissions import PermissionScope

    p = PermissionScope.parse("factor_library.read")
    assert p.resource is None  # no explicit resource = generic


def test_permission_scope_rejects_malformed_string() -> None:
    from ai_agents.permissions import PermissionScope

    with pytest.raises(ValueError):
        PermissionScope.parse("justanoun")


def test_permission_scope_rejects_forbidden_namespace() -> None:
    from ai_agents.permissions import PermissionScope

    for bad in (
        "oms.submit_order",
        "kms.sign",
        "treasury.transfer",
        "src.data.fetch",
    ):
        with pytest.raises(ValueError, match="forbidden"):
            PermissionScope.parse(bad)


# ---------------------------------------------------------------------------
# AgentPermissions
# ---------------------------------------------------------------------------


def test_agent_permissions_allows_declared_action() -> None:
    from ai_agents.permissions import AgentPermissions

    perms = AgentPermissions.from_strings(
        ("factor_library.read", "notifications.slack.post")
    )
    perms.ensure_allowed("factor_library.read:any_factor")


def test_agent_permissions_wildcard_does_not_grant_write() -> None:
    from ai_agents.permissions import AgentPermissions

    perms = AgentPermissions.from_strings(("factor_library.read",))
    with pytest.raises(PermissionError):
        perms.ensure_allowed("factor_library.write:mom_12_1")


def test_agent_permissions_resource_scope_is_enforced() -> None:
    from ai_agents.permissions import AgentPermissions

    # Agent has read permission only for "mom_12_1".
    perms = AgentPermissions.from_strings(
        ("factor_library.read:mom_12_1",)
    )
    perms.ensure_allowed("factor_library.read:mom_12_1")
    with pytest.raises(PermissionError):
        perms.ensure_allowed("factor_library.read:reversal_5")


def test_agent_permissions_wildcard_resource_grants_any_resource() -> None:
    from ai_agents.permissions import AgentPermissions

    perms = AgentPermissions.from_strings(("factor_library.read",))
    perms.ensure_allowed("factor_library.read:any_specific_thing")


# ---------------------------------------------------------------------------
# Denylist: no permission set may include forbidden namespaces at all.
# ---------------------------------------------------------------------------


def test_agent_permissions_rejects_any_forbidden_permission() -> None:
    from ai_agents.permissions import AgentPermissions

    with pytest.raises(ValueError, match="forbidden"):
        AgentPermissions.from_strings(("oms.submit_order",))
