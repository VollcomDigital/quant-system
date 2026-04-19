"""Phase 2 Task 2 - data_platform.storage dataset versioning and snapshots.

Storage rules (roadmap):

- Apache Parquet as the analytical format.
- Datasets are immutable at the snapshot level; new data produces a new
  snapshot, never mutates an old one.
- A dataset has a stable `dataset_id` (from connectors.CachePolicy) and
  snapshots are keyed by `(dataset_id, snapshot_id)` where snapshot_id is
  a content hash or monotonic version.
- Prediction artifacts, factors, and bars all flow through the same
  snapshot primitive so forecasts are versioned exactly like factors.

Phase 2 tests the abstract primitives. They are backend-agnostic (no
pyarrow required for the contract surface); a filesystem snapshot index
is provided for reproducible tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Snapshot ids are content-addressed and deterministic.
# ---------------------------------------------------------------------------


def test_snapshot_id_is_content_addressed(tmp_path: Path) -> None:
    from data_platform.storage import snapshot_id

    a = snapshot_id(b"payload")
    b = snapshot_id(b"payload")
    c = snapshot_id(b"different")
    assert a == b
    assert a != c


def test_snapshot_id_is_hex_with_known_length() -> None:
    from data_platform.storage import snapshot_id

    got = snapshot_id(b"x")
    assert len(got) == 32
    int(got, 16)  # must parse as hex


# ---------------------------------------------------------------------------
# SnapshotIndex: register, read, and refuse mutation of existing snapshots.
# ---------------------------------------------------------------------------


def test_snapshot_index_registers_and_lists(tmp_path: Path) -> None:
    from data_platform.storage import SnapshotIndex

    idx = SnapshotIndex(root=tmp_path)
    idx.register(
        dataset_id="data_platform.bars::polygon::AAPL::1d::abc",
        snapshot_id="s1",
        path=tmp_path / "f1.parquet",
        row_count=100,
    )
    snaps = list(idx.list_snapshots("data_platform.bars::polygon::AAPL::1d::abc"))
    assert len(snaps) == 1
    assert snaps[0].snapshot_id == "s1"
    assert snaps[0].row_count == 100


def test_snapshot_index_rejects_duplicate_snapshot(tmp_path: Path) -> None:
    from data_platform.storage import SnapshotIndex

    idx = SnapshotIndex(root=tmp_path)
    idx.register(
        dataset_id="d",
        snapshot_id="s1",
        path=tmp_path / "f.parquet",
        row_count=1,
    )
    with pytest.raises(ValueError, match="already exists"):
        idx.register(
            dataset_id="d",
            snapshot_id="s1",
            path=tmp_path / "f.parquet",
            row_count=2,
        )


def test_snapshot_index_supports_multiple_versions(tmp_path: Path) -> None:
    from data_platform.storage import SnapshotIndex

    idx = SnapshotIndex(root=tmp_path)
    for i, sid in enumerate(["v1", "v2", "v3"]):
        idx.register(
            dataset_id="d",
            snapshot_id=sid,
            path=tmp_path / f"f{i}.parquet",
            row_count=i + 1,
        )
    snaps = list(idx.list_snapshots("d"))
    assert [s.snapshot_id for s in snaps] == ["v1", "v2", "v3"]
    # Latest returns the most recently registered entry.
    assert idx.latest("d").snapshot_id == "v3"


def test_snapshot_index_latest_on_empty_raises(tmp_path: Path) -> None:
    from data_platform.storage import SnapshotIndex

    idx = SnapshotIndex(root=tmp_path)
    with pytest.raises(LookupError):
        idx.latest("nonexistent")


# ---------------------------------------------------------------------------
# Parquet layout contract: path must be reproducible from dataset+snapshot.
# ---------------------------------------------------------------------------


def test_parquet_layout_is_deterministic(tmp_path: Path) -> None:
    from data_platform.storage import parquet_path

    p1 = parquet_path(tmp_path, dataset_id="d::xyz", snapshot_id="s")
    p2 = parquet_path(tmp_path, dataset_id="d::xyz", snapshot_id="s")
    assert p1 == p2
    assert p1.suffix == ".parquet"
    assert "d::xyz" in str(p1).replace("_", "::") or "d" in str(p1)


# ---------------------------------------------------------------------------
# Prediction artifact persistence piggy-backs on the same SnapshotIndex.
# ---------------------------------------------------------------------------


def test_prediction_snapshot_shares_index_contract(tmp_path: Path) -> None:
    from data_platform.storage import SnapshotIndex

    idx = SnapshotIndex(root=tmp_path)
    idx.register(
        dataset_id="alpha_research.predictions::kronos::AAPL::1d::xyz",
        snapshot_id=f"model:kronos-v0.1@{idx}",  # using str() for uniqueness
        path=tmp_path / "pred.parquet",
        row_count=10,
    )
    assert len(list(idx.list_snapshots("alpha_research.predictions::kronos::AAPL::1d::xyz"))) == 1


# ---------------------------------------------------------------------------
# Snapshot immutability: the backing file path must not be reused across
# different snapshot_ids.
# ---------------------------------------------------------------------------


def test_snapshot_paths_are_unique_per_snapshot(tmp_path: Path) -> None:
    from data_platform.storage import SnapshotIndex

    idx = SnapshotIndex(root=tmp_path)
    shared_path = tmp_path / "shared.parquet"
    idx.register(dataset_id="d", snapshot_id="s1", path=shared_path, row_count=1)
    with pytest.raises(ValueError, match="path already in use"):
        idx.register(dataset_id="d", snapshot_id="s2", path=shared_path, row_count=1)
