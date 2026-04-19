"""Phase 3 Task 5 - walk-forward + CV window generators.

These generate `(train_indices, test_indices)` windows against the length
of a dataset. They are pure Python / NumPy-compatible so they are
reusable by `backtest_engine` in Phase 4 without pulling in scikit-learn.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Walk-forward (expanding anchor)
# ---------------------------------------------------------------------------


def test_walk_forward_expanding_produces_expected_windows() -> None:
    from alpha_research.ml_models.validation import walk_forward_expanding

    windows = list(
        walk_forward_expanding(
            n=10,
            initial_train=4,
            test_size=2,
            step=2,
        )
    )
    # Train anchors at 0; windows advance by step=2.
    assert windows == [
        (list(range(0, 4)), [4, 5]),
        (list(range(0, 6)), [6, 7]),
        (list(range(0, 8)), [8, 9]),
    ]


def test_walk_forward_expanding_rejects_invalid_initial_train() -> None:
    from alpha_research.ml_models.validation import walk_forward_expanding

    with pytest.raises(ValueError):
        list(walk_forward_expanding(n=10, initial_train=0, test_size=2, step=1))


def test_walk_forward_expanding_rejects_n_too_small() -> None:
    from alpha_research.ml_models.validation import walk_forward_expanding

    with pytest.raises(ValueError):
        list(walk_forward_expanding(n=3, initial_train=4, test_size=2, step=1))


# ---------------------------------------------------------------------------
# Walk-forward (rolling window)
# ---------------------------------------------------------------------------


def test_walk_forward_rolling_produces_fixed_train_window() -> None:
    from alpha_research.ml_models.validation import walk_forward_rolling

    windows = list(
        walk_forward_rolling(
            n=10,
            train_size=4,
            test_size=2,
            step=2,
        )
    )
    assert windows == [
        ([0, 1, 2, 3], [4, 5]),
        ([2, 3, 4, 5], [6, 7]),
        ([4, 5, 6, 7], [8, 9]),
    ]


def test_walk_forward_rolling_rejects_overlap_via_negative_step() -> None:
    from alpha_research.ml_models.validation import walk_forward_rolling

    with pytest.raises(ValueError):
        list(walk_forward_rolling(n=10, train_size=4, test_size=2, step=0))


# ---------------------------------------------------------------------------
# K-fold CV - time-series aware (no look-ahead). Tests produce
# non-overlapping test slices that always come AFTER the train slice.
# ---------------------------------------------------------------------------


def test_time_series_kfold_produces_non_overlapping_tests() -> None:
    from alpha_research.ml_models.validation import time_series_kfold

    folds = list(time_series_kfold(n=12, n_splits=3))
    # Every test slice must be strictly after its train slice.
    for train, test in folds:
        assert train
        assert max(train) < min(test), "look-ahead bias: train must precede test"
    # Test slices cover disjoint ranges.
    test_positions = [i for _, test in folds for i in test]
    assert len(test_positions) == len(set(test_positions))


def test_time_series_kfold_rejects_bad_split_count() -> None:
    from alpha_research.ml_models.validation import time_series_kfold

    with pytest.raises(ValueError):
        list(time_series_kfold(n=10, n_splits=1))
    with pytest.raises(ValueError):
        list(time_series_kfold(n=10, n_splits=10))


# ---------------------------------------------------------------------------
# Purged / embargoed CV - the classic prevention for feature leakage
# around fold boundaries (Lopez de Prado). Tests that train and test
# never touch at the boundary after an embargo is applied.
# ---------------------------------------------------------------------------


def test_purged_embargoed_cv_respects_embargo() -> None:
    from alpha_research.ml_models.validation import purged_embargoed_cv

    folds = list(
        purged_embargoed_cv(
            n=20,
            n_splits=4,
            embargo=2,
        )
    )
    for train, test in folds:
        for t in test:
            # No training index within `embargo` of any test index.
            assert all(abs(t - tr) > 2 for tr in train) or not train
