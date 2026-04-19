"""Research-memory abstraction.

MemPalace-inspired store over `shared_lib.contracts.ResearchMemoryRecord`.
Agents can store and retrieve past findings so researchers don't
re-propose rejected ideas and reviewers can cite prior dataset caveats.

Phase 5 ships an in-memory reference implementation. Phase 9 swaps in
a vector or SQL backend without changing the public surface.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from shared_lib.contracts import ResearchMemoryRecord

__all__ = ["ResearchMemoryStore"]


@dataclass
class ResearchMemoryStore:
    _records: dict[str, ResearchMemoryRecord] = field(
        default_factory=dict, init=False, repr=False
    )

    def add(self, record: ResearchMemoryRecord) -> None:
        if record.record_id in self._records:
            raise ValueError(f"record {record.record_id!r} already stored")
        self._records[record.record_id] = record

    def get(self, record_id: str) -> ResearchMemoryRecord:
        try:
            return self._records[record_id]
        except KeyError as exc:
            raise LookupError(f"no record for {record_id!r}") from exc

    def search(
        self,
        *,
        kind: str | None = None,
        tags: tuple[str, ...] = (),
        limit: int | None = None,
    ) -> Iterator[ResearchMemoryRecord]:
        tag_set = set(tags)
        emitted = 0
        for rec in self._records.values():
            if kind is not None and rec.kind != kind:
                continue
            if tag_set and not (set(rec.tags) & tag_set):
                continue
            yield rec
            emitted += 1
            if limit is not None and emitted >= limit:
                break
