"""Phase 5 Task 1 - ai_agents.runtime primitives.

- `AgentRegistry` - registers `AgentSpec` with id/version/role/tools;
  append-only per (agent_id, version).
- `PromptRegistry` - registers prompt templates per
  (prompt_id, version); immutable.
- `JobQueue` - enqueue/dequeue agent jobs with idempotency keys; refuses
  duplicate keys.
- `Tool` - protocol for restricted tool surfaces; tools declare name +
  required permissions.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# AgentRegistry
# ---------------------------------------------------------------------------


def test_agent_registry_registers_and_retrieves() -> None:
    from ai_agents.runtime import AgentRegistry, AgentSpec

    reg = AgentRegistry()
    spec = AgentSpec(
        agent_id="alpha_researcher",
        version="v1",
        role="researcher",
        tools=("read_papers", "propose_factor"),
    )
    reg.register(spec)
    assert reg.get("alpha_researcher", "v1") == spec


def test_agent_registry_refuses_duplicate_version() -> None:
    from ai_agents.runtime import AgentRegistry, AgentSpec

    reg = AgentRegistry()
    spec = AgentSpec(
        agent_id="r", version="v1", role="researcher", tools=()
    )
    reg.register(spec)
    with pytest.raises(ValueError, match="already"):
        reg.register(spec)


def test_agent_spec_rejects_unknown_role() -> None:
    from ai_agents.runtime import AgentSpec

    with pytest.raises(ValueError):
        AgentSpec(
            agent_id="x",
            version="v1",
            role="archon",  # not in closed enum
            tools=(),
        )


def test_agent_spec_rejects_empty_agent_id() -> None:
    from ai_agents.runtime import AgentSpec

    with pytest.raises(ValueError):
        AgentSpec(agent_id="", version="v1", role="researcher", tools=())


# ---------------------------------------------------------------------------
# PromptRegistry
# ---------------------------------------------------------------------------


def test_prompt_registry_round_trip() -> None:
    from ai_agents.runtime import PromptRegistry

    reg = PromptRegistry()
    reg.register(
        prompt_id="summarize-paper",
        version="v1",
        template="Summarize: {abstract}",
    )
    got = reg.get(prompt_id="summarize-paper", version="v1")
    assert got.template == "Summarize: {abstract}"


def test_prompt_registry_refuses_overwrite() -> None:
    from ai_agents.runtime import PromptRegistry

    reg = PromptRegistry()
    reg.register(prompt_id="p", version="v1", template="x")
    with pytest.raises(ValueError):
        reg.register(prompt_id="p", version="v1", template="y")


def test_prompt_registry_rejects_empty_template() -> None:
    from ai_agents.runtime import PromptRegistry

    reg = PromptRegistry()
    with pytest.raises(ValueError):
        reg.register(prompt_id="p", version="v1", template="")


def test_prompt_registry_render_applies_variables() -> None:
    from ai_agents.runtime import PromptRegistry

    reg = PromptRegistry()
    reg.register(
        prompt_id="p",
        version="v1",
        template="Hello, {name}.",
    )
    rendered = reg.render(prompt_id="p", version="v1", variables={"name": "World"})
    assert rendered == "Hello, World."


def test_prompt_registry_render_rejects_unfilled_placeholders() -> None:
    from ai_agents.runtime import PromptRegistry

    reg = PromptRegistry()
    reg.register(prompt_id="p", version="v1", template="Hi {who}")
    with pytest.raises(KeyError):
        reg.render(prompt_id="p", version="v1", variables={})


# ---------------------------------------------------------------------------
# JobQueue
# ---------------------------------------------------------------------------


def test_job_queue_enqueue_dequeue_fifo() -> None:
    from ai_agents.runtime import JobQueue

    q = JobQueue()
    q.enqueue(agent_id="r", idempotency_key="k1", payload={"x": 1})
    q.enqueue(agent_id="r", idempotency_key="k2", payload={"x": 2})
    assert q.size() == 2
    first = q.dequeue()
    assert first.idempotency_key == "k1"


def test_job_queue_refuses_duplicate_idempotency_key() -> None:
    from ai_agents.runtime import JobQueue

    q = JobQueue()
    q.enqueue(agent_id="r", idempotency_key="k1", payload={})
    with pytest.raises(ValueError, match="idempotency"):
        q.enqueue(agent_id="r", idempotency_key="k1", payload={})


def test_job_queue_dequeue_on_empty_returns_none() -> None:
    from ai_agents.runtime import JobQueue

    q = JobQueue()
    assert q.dequeue() is None


# ---------------------------------------------------------------------------
# Tool protocol
# ---------------------------------------------------------------------------


def test_tool_declares_required_permissions() -> None:
    from ai_agents.runtime import Tool

    class _ReadPapers:
        name = "read_papers"
        required_permissions = ("arxiv.read",)

        def invoke(self, payload):
            return {"ok": True}

    tool: Tool = _ReadPapers()
    assert tool.required_permissions == ("arxiv.read",)
    assert tool.name == "read_papers"
