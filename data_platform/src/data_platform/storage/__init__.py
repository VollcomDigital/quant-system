"""Immutable dataset storage primitives.

Phase 2 commits to:

- Parquet as the analytical format.
- Content-addressed snapshots per dataset.
- A single `SnapshotIndex` abstraction that works for bars, factors, and
  prediction artifacts so forecasts are versioned the same way factors
  are.

The index is backend-agnostic. A filesystem-backed reference
implementation is provided; production deployments can swap in a
catalog service (e.g. AWS Glue, Unity Catalog) by implementing the
same surface.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Snapshot",
    "SnapshotIndex",
    "parquet_path",
    "snapshot_id",
]


def snapshot_id(payload: bytes) -> str:
    """Return a 32-hex-char content-addressed snapshot id."""
    return hashlib.sha256(payload).hexdigest()[:32]


_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


def _sanitize(component: str) -> str:
    return _SAFE.sub("_", component)


def parquet_path(root: Path, *, dataset_id: str, snapshot_id: str) -> Path:
    """Return the on-disk Parquet path for `(dataset_id, snapshot_id)`."""
    return Path(root) / _sanitize(dataset_id) / f"{_sanitize(snapshot_id)}.parquet"


@dataclass(frozen=True, slots=True)
class Snapshot:
    dataset_id: str
    snapshot_id: str
    path: Path
    row_count: int


@dataclass
class SnapshotIndex:
    root: Path
    _entries: list[Snapshot] = field(default_factory=list, init=False, repr=False)
    _paths: set[Path] = field(default_factory=set, init=False, repr=False)

    def register(
        self,
        *,
        dataset_id: str,
        snapshot_id: str,
        path: Path,
        row_count: int,
    ) -> Snapshot:
        """Register a new immutable snapshot.

        Raises:
            ValueError: if `(dataset_id, snapshot_id)` already exists.
            ValueError: if `path` is already associated with another snapshot.
        """
        for existing in self._entries:
            if existing.dataset_id == dataset_id and existing.snapshot_id == snapshot_id:
                raise ValueError(
                    f"snapshot ({dataset_id!r}, {snapshot_id!r}) already exists"
                )
        if Path(path) in self._paths:
            raise ValueError(
                f"path already in use by another snapshot: {path}"
            )
        entry = Snapshot(
            dataset_id=dataset_id,
            snapshot_id=snapshot_id,
            path=Path(path),
            row_count=row_count,
        )
        self._entries.append(entry)
        self._paths.add(entry.path)
        return entry

    def list_snapshots(self, dataset_id: str) -> Iterator[Snapshot]:
        for e in self._entries:
            if e.dataset_id == dataset_id:
                yield e

    def latest(self, dataset_id: str) -> Snapshot:
        for e in reversed(self._entries):
            if e.dataset_id == dataset_id:
                return e
        raise LookupError(f"no snapshots for dataset {dataset_id!r}")
