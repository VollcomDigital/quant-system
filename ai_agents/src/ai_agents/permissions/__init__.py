"""Scoped permission model for agents.

Permission strings follow this grammar::

    <namespace>.<action>[:<resource>]

Examples::

    factor_library.read
    factor_library.read:mom_12_1
    backtest_engine.run
    notifications.slack.post

The namespaces below are always forbidden and will be refused at
construction time (ADR-0004 Non-Negotiable Guardrails):

- `oms.*`
- `kms.*`
- `treasury.*`
- `src.*` (legacy)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

__all__ = ["AgentPermissions", "PermissionScope"]


FORBIDDEN_NAMESPACES: frozenset[str] = frozenset({"oms", "kms", "treasury", "src"})


@dataclass(frozen=True, slots=True)
class PermissionScope:
    namespace: str
    action: str
    resource: str | None

    @classmethod
    def parse(cls, raw: str) -> PermissionScope:
        if ":" in raw:
            left, resource = raw.split(":", 1)
        else:
            left, resource = raw, None
        if "." not in left:
            raise ValueError(f"malformed permission: {raw!r}")
        namespace, _, action = left.partition(".")
        # The action may itself contain dots (e.g. `slack.post`); keep
        # everything after the first `.` as the action name.
        root_ns = namespace
        if root_ns in FORBIDDEN_NAMESPACES:
            raise ValueError(
                f"forbidden namespace for agents: {root_ns!r} "
                f"(oms/kms/treasury/src are reserved)"
            )
        if not namespace or not action:
            raise ValueError(f"malformed permission: {raw!r}")
        return cls(namespace=namespace, action=action, resource=resource)


def _covers(grant: PermissionScope, request: PermissionScope) -> bool:
    if grant.namespace != request.namespace:
        return False
    if grant.action != request.action:
        return False
    if grant.resource is None:
        # wildcard on resource
        return True
    return grant.resource == request.resource


@dataclass(frozen=True, slots=True)
class AgentPermissions:
    scopes: tuple[PermissionScope, ...]

    @classmethod
    def from_strings(cls, raws: Iterable[str]) -> AgentPermissions:
        return cls(scopes=tuple(PermissionScope.parse(r) for r in raws))

    def ensure_allowed(self, raw: str) -> None:
        request = PermissionScope.parse(raw)
        for grant in self.scopes:
            if _covers(grant, request):
                return
        raise PermissionError(f"permission denied: {raw!r}")
