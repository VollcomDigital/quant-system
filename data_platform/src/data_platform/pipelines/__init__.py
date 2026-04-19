"""Orchestrator-agnostic DAG primitives.

Airflow is the default runtime (ADR-0002) but DAG *definitions* live here
and must not import airflow. Translators (Airflow / Prefect / Dagster)
consume these objects in Phase 9.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TypeVar

__all__ = [
    "DAG",
    "RetryPolicy",
    "TaskSpec",
    "backfill_windows",
]


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    upstream: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must be non-empty")


@dataclass(frozen=True, slots=True)
class DAG:
    dag_id: str
    tasks: Sequence[TaskSpec]

    def __post_init__(self) -> None:
        ids = [t.task_id for t in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate task_ids in DAG {self.dag_id!r}")
        valid = set(ids)
        for task in self.tasks:
            unknown = [u for u in task.upstream if u not in valid]
            if unknown:
                raise ValueError(
                    f"task {task.task_id!r} references unknown upstream: {unknown}"
                )
        self.topological_order()

    def topological_order(self) -> list[str]:
        remaining = {t.task_id: set(t.upstream) for t in self.tasks}
        ordered: list[str] = []
        while remaining:
            ready = sorted(tid for tid, deps in remaining.items() if not deps)
            if not ready:
                raise ValueError(f"cycle detected in DAG {self.dag_id!r}")
            for tid in ready:
                ordered.append(tid)
                remaining.pop(tid)
                for deps in remaining.values():
                    deps.discard(tid)
        return ordered


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int
    base_delay: float
    retriable: tuple[type[BaseException], ...]

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay < 0.0:
            raise ValueError("base_delay must be >= 0")

    def run(self, fn: Callable[[], T]) -> T:
        last: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return fn()
            except BaseException as exc:  # noqa: BLE001
                if not isinstance(exc, self.retriable):
                    raise
                last = exc
                if attempt >= self.max_attempts:
                    break
                if self.base_delay > 0.0:
                    time.sleep(self.base_delay * (2 ** (attempt - 1)))
        assert last is not None
        raise last


def backfill_windows(
    *,
    start: datetime,
    end: datetime,
    cadence: timedelta,
) -> Iterator[tuple[datetime, datetime]]:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("backfill_windows requires tz-aware datetimes")
    if end <= start:
        raise ValueError("end must be > start")
    if cadence.total_seconds() <= 0:
        raise ValueError("cadence must be > 0")

    cursor = start
    while cursor < end:
        step = cursor + cadence
        if step > end:
            step = end
        yield (cursor, step)
        cursor = step
