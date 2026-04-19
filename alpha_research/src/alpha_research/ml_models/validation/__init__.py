"""Walk-forward and CV window generators.

Pure-Python implementations - no scikit-learn dependency. Returned
windows are `(train_indices, test_indices)` lists of ints so callers
can index into any array-like (Polars, NumPy, pandas, Python lists).
"""

from __future__ import annotations

from collections.abc import Iterator

__all__ = [
    "purged_embargoed_cv",
    "time_series_kfold",
    "walk_forward_expanding",
    "walk_forward_rolling",
]


def walk_forward_expanding(
    *,
    n: int,
    initial_train: int,
    test_size: int,
    step: int,
) -> Iterator[tuple[list[int], list[int]]]:
    if initial_train <= 0 or test_size <= 0 or step <= 0:
        raise ValueError("initial_train, test_size, step must all be > 0")
    if n < initial_train + test_size:
        raise ValueError(
            f"n={n} too small for initial_train={initial_train} + test_size={test_size}"
        )
    train_end = initial_train
    while train_end + test_size <= n:
        train = list(range(0, train_end))
        test = list(range(train_end, train_end + test_size))
        yield (train, test)
        train_end += step


def walk_forward_rolling(
    *,
    n: int,
    train_size: int,
    test_size: int,
    step: int,
) -> Iterator[tuple[list[int], list[int]]]:
    if train_size <= 0 or test_size <= 0 or step <= 0:
        raise ValueError("train_size, test_size, step must all be > 0")
    if n < train_size + test_size:
        raise ValueError(
            f"n={n} too small for train_size={train_size} + test_size={test_size}"
        )
    start = 0
    while start + train_size + test_size <= n:
        train = list(range(start, start + train_size))
        test = list(range(start + train_size, start + train_size + test_size))
        yield (train, test)
        start += step


def time_series_kfold(
    *, n: int, n_splits: int
) -> Iterator[tuple[list[int], list[int]]]:
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n_splits >= n:
        raise ValueError("n_splits must be < n")
    fold_size = n // (n_splits + 1)
    for i in range(1, n_splits + 1):
        train_end = i * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        if test_end <= test_start:
            continue
        yield (list(range(0, train_end)), list(range(test_start, test_end)))


def purged_embargoed_cv(
    *, n: int, n_splits: int, embargo: int
) -> Iterator[tuple[list[int], list[int]]]:
    if embargo < 0:
        raise ValueError("embargo must be >= 0")
    for train, test in time_series_kfold(n=n, n_splits=n_splits):
        test_set = set(test)
        filtered_train = [
            i
            for i in train
            if not any(abs(i - t) <= embargo for t in test_set)
        ]
        yield (filtered_train, test)
