"""Agent runtime primitives.

- `AgentRegistry` / `AgentSpec`  - registry of deployed agents.
- `PromptRegistry`               - versioned, immutable prompt templates.
- `JobQueue`                     - idempotent agent-job queue.
- `Tool`                         - Protocol for restricted tool surfaces.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    "AgentJob",
    "AgentRegistry",
    "AgentRole",
    "AgentSpec",
    "JobQueue",
    "PromptRegistry",
    "PromptTemplate",
    "Tool",
]


AgentRole = Literal[
    "researcher",
    "reviewer",
    "risk_monitor",
    "latency_monitor",
    "allocation",
    "routing",
    "gateway_health",
]
_VALID_ROLES = {
    "researcher",
    "reviewer",
    "risk_monitor",
    "latency_monitor",
    "allocation",
    "routing",
    "gateway_health",
}


@dataclass(frozen=True, slots=True)
class AgentSpec:
    agent_id: str
    version: str
    role: AgentRole
    tools: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ValueError("agent_id must be non-empty")
        if not self.version:
            raise ValueError("version must be non-empty")
        if self.role not in _VALID_ROLES:
            raise ValueError(f"unknown role: {self.role!r}")


@dataclass
class AgentRegistry:
    _entries: dict[tuple[str, str], AgentSpec] = field(
        default_factory=dict, init=False, repr=False
    )

    def register(self, spec: AgentSpec) -> None:
        key = (spec.agent_id, spec.version)
        if key in self._entries:
            raise ValueError(f"agent {key} already registered")
        self._entries[key] = spec

    def get(self, agent_id: str, version: str) -> AgentSpec:
        try:
            return self._entries[(agent_id, version)]
        except KeyError as exc:
            raise LookupError(
                f"no agent registered for {(agent_id, version)!r}"
            ) from exc


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    prompt_id: str
    version: str
    template: str


@dataclass
class PromptRegistry:
    _entries: dict[tuple[str, str], PromptTemplate] = field(
        default_factory=dict, init=False, repr=False
    )

    def register(self, *, prompt_id: str, version: str, template: str) -> None:
        if not template:
            raise ValueError("template must be non-empty")
        key = (prompt_id, version)
        if key in self._entries:
            raise ValueError(f"prompt {key} already registered")
        self._entries[key] = PromptTemplate(
            prompt_id=prompt_id, version=version, template=template
        )

    def get(self, *, prompt_id: str, version: str) -> PromptTemplate:
        try:
            return self._entries[(prompt_id, version)]
        except KeyError as exc:
            raise LookupError(f"no prompt for {(prompt_id, version)!r}") from exc

    def render(
        self, *, prompt_id: str, version: str, variables: dict[str, str]
    ) -> str:
        template = self.get(prompt_id=prompt_id, version=version).template
        placeholders = set(re.findall(r"\{(\w+)\}", template))
        missing = placeholders - variables.keys()
        if missing:
            raise KeyError(f"missing prompt variables: {sorted(missing)}")
        return template.format(**variables)


@dataclass(frozen=True, slots=True)
class AgentJob:
    agent_id: str
    idempotency_key: str
    payload: dict[str, Any]


@dataclass
class JobQueue:
    _items: deque[AgentJob] = field(default_factory=deque, init=False, repr=False)
    _seen_keys: set[str] = field(default_factory=set, init=False, repr=False)

    def enqueue(
        self, *, agent_id: str, idempotency_key: str, payload: dict[str, Any]
    ) -> AgentJob:
        if not idempotency_key:
            raise ValueError("idempotency_key must be non-empty")
        if idempotency_key in self._seen_keys:
            raise ValueError(
                f"idempotency key {idempotency_key!r} already enqueued"
            )
        job = AgentJob(
            agent_id=agent_id,
            idempotency_key=idempotency_key,
            payload=payload,
        )
        self._items.append(job)
        self._seen_keys.add(idempotency_key)
        return job

    def dequeue(self) -> AgentJob | None:
        if not self._items:
            return None
        return self._items.popleft()

    def size(self) -> int:
        return len(self._items)


@runtime_checkable
class Tool(Protocol):
    name: str
    required_permissions: tuple[str, ...]

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...
