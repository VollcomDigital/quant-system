"""Execution algorithms - TWAP + VWAP slicing.

These helpers compute `(time, quantity)` slice schedules. The EMS
consumes the output to build child `Order`s. They are pure-Python so
they can run in research notebooks and the simulator identically.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

__all__ = ["twap_slice", "vwap_slice"]


def twap_slice(
    *,
    total_quantity: Decimal,
    start: datetime,
    end: datetime,
    num_slices: int,
) -> list[tuple[datetime, Decimal]]:
    if num_slices <= 0:
        raise ValueError("num_slices must be > 0")
    if end <= start:
        raise ValueError("end must be > start")
    if total_quantity <= 0:
        raise ValueError("total_quantity must be > 0")

    gap = (end - start) / num_slices
    base = total_quantity // num_slices
    quantities: list[Decimal] = []
    for i in range(num_slices):
        q = base if i < num_slices - 1 else total_quantity - base * (num_slices - 1)
        quantities.append(q)

    return [
        (start + gap * i, quantities[i]) for i in range(num_slices)
    ]


def vwap_slice(
    *,
    total_quantity: Decimal,
    volume_profile: Sequence[Decimal],
) -> list[Decimal]:
    if not volume_profile:
        raise ValueError("volume_profile must be non-empty")
    total_volume = sum(volume_profile, start=Decimal("0"))
    if total_volume <= 0:
        raise ValueError("volume_profile total must be > 0")
    if total_quantity <= 0:
        raise ValueError("total_quantity must be > 0")

    quantities = [total_quantity * v / total_volume for v in volume_profile]
    # Rounding drift: give the last bucket whatever residual is left so
    # the slices sum to total_quantity exactly.
    drift = total_quantity - sum(quantities[:-1], start=Decimal("0")) - quantities[-1]
    quantities[-1] = quantities[-1] + drift
    return quantities
