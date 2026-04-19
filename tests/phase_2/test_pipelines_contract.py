"""Phase 2 Task 3 - data_platform.pipelines DAG + retry contracts.

Contract goals:

- DAGs are declarative and orchestrator-agnostic. Airflow is the default
  runtime (ADR-0002) but the pipeline definition must not import airflow
  at the contract level.
- A `TaskSpec` names its upstream dependencies. The DAG must refuse
  cycles and unknown dependencies at validation time.
- A `RetryPolicy` defines max_attempts and backoff. Retries happen only
  on retriable classes; non-retriable errors short-circuit.
- Backfill generation produces a deterministic sequence of
  (window_start, window_end) tuples given a cadence and a window.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

# ---------------------------------------------------------------------------
# TaskSpec + DAG validation
# ---------------------------------------------------------------------------


def test_dag_accepts_valid_linear_chain() -> None:
    from data_platform.pipelines import DAG, TaskSpec

    dag = DAG(
        dag_id="bars_daily",
        tasks=[
            TaskSpec(task_id="ingest", upstream=()),
            TaskSpec(task_id="validate", upstream=("ingest",)),
            TaskSpec(task_id="publish", upstream=("validate",)),
        ],
    )
    assert dag.topological_order() == ["ingest", "validate", "publish"]


def test_dag_rejects_cycle() -> None:
    from data_platform.pipelines import DAG, TaskSpec

    with pytest.raises(ValueError, match="cycle"):
        DAG(
            dag_id="bad",
            tasks=[
                TaskSpec(task_id="a", upstream=("b",)),
                TaskSpec(task_id="b", upstream=("a",)),
            ],
        )


def test_dag_rejects_unknown_upstream() -> None:
    from data_platform.pipelines import DAG, TaskSpec

    with pytest.raises(ValueError, match="unknown"):
        DAG(
            dag_id="bad",
            tasks=[
                TaskSpec(task_id="a", upstream=("ghost",)),
            ],
        )


def test_dag_rejects_duplicate_task_ids() -> None:
    from data_platform.pipelines import DAG, TaskSpec

    with pytest.raises(ValueError, match="duplicate"):
        DAG(
            dag_id="dup",
            tasks=[
                TaskSpec(task_id="x", upstream=()),
                TaskSpec(task_id="x", upstream=()),
            ],
        )


def test_dag_topological_order_handles_diamond() -> None:
    from data_platform.pipelines import DAG, TaskSpec

    dag = DAG(
        dag_id="diamond",
        tasks=[
            TaskSpec(task_id="root", upstream=()),
            TaskSpec(task_id="left", upstream=("root",)),
            TaskSpec(task_id="right", upstream=("root",)),
            TaskSpec(task_id="join", upstream=("left", "right")),
        ],
    )
    order = dag.topological_order()
    assert order.index("root") < order.index("left") < order.index("join")
    assert order.index("root") < order.index("right") < order.index("join")


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------


def test_retry_policy_retries_on_retriable_errors() -> None:
    from data_platform.pipelines import RetryPolicy

    policy = RetryPolicy(max_attempts=3, base_delay=0.0, retriable=(TimeoutError,))
    attempts: list[int] = []

    def _work() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise TimeoutError("flaky")
        return "ok"

    assert policy.run(_work) == "ok"
    assert len(attempts) == 3


def test_retry_policy_does_not_retry_nonretriable() -> None:
    from data_platform.pipelines import RetryPolicy

    policy = RetryPolicy(max_attempts=3, base_delay=0.0, retriable=(TimeoutError,))

    def _work() -> None:
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        policy.run(_work)


def test_retry_policy_gives_up_after_max_attempts() -> None:
    from data_platform.pipelines import RetryPolicy

    policy = RetryPolicy(max_attempts=2, base_delay=0.0, retriable=(TimeoutError,))

    def _work() -> None:
        raise TimeoutError("always")

    with pytest.raises(TimeoutError):
        policy.run(_work)


def test_retry_policy_rejects_nonsensical_max_attempts() -> None:
    from data_platform.pipelines import RetryPolicy

    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0, base_delay=0.0, retriable=())


# ---------------------------------------------------------------------------
# Backfill windows
# ---------------------------------------------------------------------------


def test_backfill_generates_daily_windows() -> None:
    from data_platform.pipelines import backfill_windows

    windows = list(
        backfill_windows(
            start=datetime(2026, 4, 1, tzinfo=UTC),
            end=datetime(2026, 4, 4, tzinfo=UTC),
            cadence=timedelta(days=1),
        )
    )
    assert windows == [
        (datetime(2026, 4, 1, tzinfo=UTC), datetime(2026, 4, 2, tzinfo=UTC)),
        (datetime(2026, 4, 2, tzinfo=UTC), datetime(2026, 4, 3, tzinfo=UTC)),
        (datetime(2026, 4, 3, tzinfo=UTC), datetime(2026, 4, 4, tzinfo=UTC)),
    ]


def test_backfill_rejects_non_monotonic_range() -> None:
    from data_platform.pipelines import backfill_windows

    with pytest.raises(ValueError):
        list(
            backfill_windows(
                start=datetime(2026, 4, 4, tzinfo=UTC),
                end=datetime(2026, 4, 1, tzinfo=UTC),
                cadence=timedelta(days=1),
            )
        )


def test_backfill_rejects_naive_timestamps() -> None:
    from data_platform.pipelines import backfill_windows

    with pytest.raises(ValueError):
        list(
            backfill_windows(
                start=datetime(2026, 4, 1),
                end=datetime(2026, 4, 4),
                cadence=timedelta(days=1),
            )
        )


def test_backfill_rejects_nonpositive_cadence() -> None:
    from data_platform.pipelines import backfill_windows

    with pytest.raises(ValueError):
        list(
            backfill_windows(
                start=datetime(2026, 4, 1, tzinfo=UTC),
                end=datetime(2026, 4, 4, tzinfo=UTC),
                cadence=timedelta(0),
            )
        )
