"""Phase 3 Task 3 - alpha_research.ml_models.rl research lane.

RL is a *research lane*, not the default strategy framework. The Phase 3
roadmap only needs the minimum contract so that:

- Environments are reproducible (fixed seed path).
- Observations are structured (shape + dtype on the contract).
- Actions respect the bounded-output-action-space rule (ADR-0004 Layer 1:
  model guardrails before order formulation).
- Train/Test/Trade flow is modelled explicitly (FinRL pattern) so no
  environment can accidentally leak test data into training.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

# ---------------------------------------------------------------------------
# Environment contract
# ---------------------------------------------------------------------------


def test_rl_environment_returns_contract_metadata() -> None:
    from alpha_research.ml_models.rl import RLEnvironment
    from shared_lib.contracts import RLEnvironmentMetadata

    class _CashEnv(RLEnvironment):
        def __init__(self) -> None:
            super().__init__()
            self._t = 0
            self._max_t = 5

        def metadata(self) -> RLEnvironmentMetadata:
            return RLEnvironmentMetadata(
                env_id="cash-v0",
                observation_space_shape=(2,),
                action_space_kind="continuous",
                action_space_bounds=(Decimal("-1"), Decimal("1")),
                reward_scale=Decimal("1"),
            )

        def reset(self, *, seed: int | None = None):
            self._t = 0
            return (0.0, 0.0)

        def step(self, action):
            self._t += 1
            return (float(self._t), 0.0), 0.0, self._t >= self._max_t

    env = _CashEnv()
    meta = env.metadata()
    assert meta.env_id == "cash-v0"


def test_rl_environment_reset_is_deterministic_with_seed() -> None:
    from alpha_research.ml_models.rl import RLEnvironment
    from shared_lib.contracts import RLEnvironmentMetadata

    class _Seeded(RLEnvironment):
        def metadata(self):
            return RLEnvironmentMetadata(
                env_id="seeded-v0",
                observation_space_shape=(1,),
                action_space_kind="discrete",
                action_space_bounds=None,
                reward_scale=Decimal("1"),
            )

        def reset(self, *, seed=None):
            import random

            r = random.Random(seed)
            return (r.random(),)

        def step(self, action):  # pragma: no cover
            return (0.0,), 0.0, True

    env = _Seeded()
    a = env.reset(seed=42)
    b = env.reset(seed=42)
    assert a == b


# ---------------------------------------------------------------------------
# Bounded action space: every action passed to step() must respect the
# declared bounds. Layer 1 of ADR-0004 kill-switch architecture.
# ---------------------------------------------------------------------------


def test_bounded_action_space_enforces_declared_bounds() -> None:
    from alpha_research.ml_models.rl import bounded_action_space

    action_space = bounded_action_space(low=Decimal("-1"), high=Decimal("1"))
    action_space.validate(Decimal("0.5"))
    with pytest.raises(ValueError):
        action_space.validate(Decimal("1.5"))


def test_bounded_action_space_rejects_inverted_bounds() -> None:
    from alpha_research.ml_models.rl import bounded_action_space

    with pytest.raises(ValueError):
        bounded_action_space(low=Decimal("1"), high=Decimal("-1"))


# ---------------------------------------------------------------------------
# Train/Test/Trade split gates
# ---------------------------------------------------------------------------


def test_train_test_trade_split_rejects_overlap() -> None:
    from datetime import UTC, datetime

    from alpha_research.ml_models.rl import TrainTestTradeSplit

    with pytest.raises(ValueError, match="overlap"):
        TrainTestTradeSplit(
            train=(datetime(2020, 1, 1, tzinfo=UTC), datetime(2022, 1, 1, tzinfo=UTC)),
            test=(datetime(2021, 6, 1, tzinfo=UTC), datetime(2023, 1, 1, tzinfo=UTC)),
            trade=(datetime(2023, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)),
        )


def test_train_test_trade_split_rejects_non_chronological() -> None:
    from datetime import UTC, datetime

    from alpha_research.ml_models.rl import TrainTestTradeSplit

    with pytest.raises(ValueError):
        TrainTestTradeSplit(
            train=(datetime(2023, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)),
            test=(datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)),
            trade=(datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC)),
        )


def test_train_test_trade_split_accepts_valid_layout() -> None:
    from datetime import UTC, datetime

    from alpha_research.ml_models.rl import TrainTestTradeSplit

    split = TrainTestTradeSplit(
        train=(datetime(2020, 1, 1, tzinfo=UTC), datetime(2022, 1, 1, tzinfo=UTC)),
        test=(datetime(2022, 1, 1, tzinfo=UTC), datetime(2023, 1, 1, tzinfo=UTC)),
        trade=(datetime(2023, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)),
    )
    assert split.train[1] == split.test[0]
