"""Phase 5 Task 2 - ai_agents.memory research-memory abstraction.

MemPalace-inspired store over `shared_lib.contracts.ResearchMemoryRecord`.
Agents can store and retrieve past findings so researchers don't
re-propose rejected ideas and reviewers can cite prior dataset caveats.

Phase 5 ships the contract; a backend (vector DB, SQLite, etc.) lands
in Phase 9.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest


def _rec(record_id: str, kind: str = "factor_hypothesis", tags=()):
    from shared_lib.contracts import ResearchMemoryRecord

    return ResearchMemoryRecord(
        record_id=record_id,
        kind=kind,
        title=f"title-{record_id}",
        body=f"body-{record_id}",
        tags=tuple(tags),
        created_at=datetime(2026, 4, 19, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def test_memory_store_round_trip() -> None:
    from ai_agents.memory import ResearchMemoryStore

    store = ResearchMemoryStore()
    rec = _rec("m-1")
    store.add(rec)
    assert store.get("m-1") == rec


def test_memory_store_rejects_duplicate_record_id() -> None:
    from ai_agents.memory import ResearchMemoryStore

    store = ResearchMemoryStore()
    store.add(_rec("m-1"))
    with pytest.raises(ValueError, match="already"):
        store.add(_rec("m-1"))


def test_memory_store_lookup_missing_raises() -> None:
    from ai_agents.memory import ResearchMemoryStore

    with pytest.raises(LookupError):
        ResearchMemoryStore().get("nope")


# ---------------------------------------------------------------------------
# Retrieval by kind / tags
# ---------------------------------------------------------------------------


def test_memory_search_by_kind() -> None:
    from ai_agents.memory import ResearchMemoryStore

    store = ResearchMemoryStore()
    store.add(_rec("a", kind="factor_hypothesis"))
    store.add(_rec("b", kind="rejected_idea"))
    store.add(_rec("c", kind="factor_hypothesis"))
    ids = {r.record_id for r in store.search(kind="factor_hypothesis")}
    assert ids == {"a", "c"}


def test_memory_search_by_tags_matches_any() -> None:
    from ai_agents.memory import ResearchMemoryStore

    store = ResearchMemoryStore()
    store.add(_rec("a", tags=("momentum",)))
    store.add(_rec("b", tags=("small_cap", "momentum")))
    store.add(_rec("c", tags=("reversal",)))
    ids = {r.record_id for r in store.search(tags=("momentum",))}
    assert ids == {"a", "b"}


def test_memory_search_combines_kind_and_tags() -> None:
    from ai_agents.memory import ResearchMemoryStore

    store = ResearchMemoryStore()
    store.add(_rec("a", kind="factor_hypothesis", tags=("mom",)))
    store.add(_rec("b", kind="rejected_idea", tags=("mom",)))
    store.add(_rec("c", kind="factor_hypothesis", tags=("rev",)))
    ids = {
        r.record_id
        for r in store.search(kind="factor_hypothesis", tags=("mom",))
    }
    assert ids == {"a"}


# ---------------------------------------------------------------------------
# Pagination & ordering
# ---------------------------------------------------------------------------


def test_memory_search_is_ordered_by_insertion() -> None:
    from ai_agents.memory import ResearchMemoryStore

    store = ResearchMemoryStore()
    for i in range(5):
        store.add(_rec(f"m-{i}"))
    results = list(store.search())
    assert [r.record_id for r in results] == [f"m-{i}" for i in range(5)]


def test_memory_search_respects_limit() -> None:
    from ai_agents.memory import ResearchMemoryStore

    store = ResearchMemoryStore()
    for i in range(10):
        store.add(_rec(f"m-{i}"))
    assert len(list(store.search(limit=3))) == 3
