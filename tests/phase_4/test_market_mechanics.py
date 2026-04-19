"""Phase 4 Task 3 - Market mechanics: slippage / fee / spread / impact / latency.

Each mechanic is a small, composable unit. The backtest engine
composes them to turn an idealized signal into a realistic fill.

All models live in `backtest_engine.market_mechanics` and take Decimals
(not floats).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

# ---------------------------------------------------------------------------
# Slippage
# ---------------------------------------------------------------------------


def test_fixed_bps_slippage_buy_moves_price_up() -> None:
    from backtest_engine.market_mechanics import FixedBpsSlippage

    m = FixedBpsSlippage(bps=Decimal("5"))  # 5 bps = 0.05%
    got = m.apply(side="buy", mid=Decimal("100"))
    assert got == Decimal("100.05")


def test_fixed_bps_slippage_sell_moves_price_down() -> None:
    from backtest_engine.market_mechanics import FixedBpsSlippage

    m = FixedBpsSlippage(bps=Decimal("10"))
    got = m.apply(side="sell", mid=Decimal("100"))
    assert got == Decimal("99.9")


def test_fixed_bps_slippage_rejects_negative_bps() -> None:
    from backtest_engine.market_mechanics import FixedBpsSlippage

    with pytest.raises(ValueError):
        FixedBpsSlippage(bps=Decimal("-1"))


# ---------------------------------------------------------------------------
# Fee model
# ---------------------------------------------------------------------------


def test_percentage_fee_charges_notional() -> None:
    from backtest_engine.market_mechanics import PercentageFee

    fee = PercentageFee(rate=Decimal("0.001"))  # 10 bps
    charged = fee.compute(notional=Decimal("1000"))
    assert charged == Decimal("1.0")


def test_percentage_fee_minimum_floor_applied() -> None:
    from backtest_engine.market_mechanics import PercentageFee

    fee = PercentageFee(rate=Decimal("0.0001"), minimum=Decimal("0.50"))
    charged = fee.compute(notional=Decimal("100"))  # 0.01, floored to 0.50
    assert charged == Decimal("0.50")


def test_per_share_fee_is_linear() -> None:
    from backtest_engine.market_mechanics import PerShareFee

    fee = PerShareFee(per_share=Decimal("0.005"))
    assert fee.compute_from_quantity(quantity=Decimal("100")) == Decimal("0.5")


# ---------------------------------------------------------------------------
# Spread
# ---------------------------------------------------------------------------


def test_half_spread_adjusts_buy_and_sell_symmetrically() -> None:
    from backtest_engine.market_mechanics import HalfSpreadModel

    m = HalfSpreadModel(spread_bps=Decimal("20"))  # 20 bps spread
    # buy pays mid + half = 100 + 0.1 = 100.1
    assert m.apply(side="buy", mid=Decimal("100")) == Decimal("100.1")
    assert m.apply(side="sell", mid=Decimal("100")) == Decimal("99.9")


# ---------------------------------------------------------------------------
# Market impact (square-root rule)
# ---------------------------------------------------------------------------


def test_square_root_impact_scales_with_size() -> None:
    from backtest_engine.market_mechanics import SquareRootImpact

    m = SquareRootImpact(k=Decimal("10"))
    # impact bps ~ k * sqrt(participation) where participation = qty / volume
    small = m.impact_bps(quantity=Decimal("100"), volume=Decimal("1000000"))
    large = m.impact_bps(quantity=Decimal("10000"), volume=Decimal("1000000"))
    assert large > small


def test_square_root_impact_zero_for_tiny_participation() -> None:
    from backtest_engine.market_mechanics import SquareRootImpact

    m = SquareRootImpact(k=Decimal("10"))
    assert m.impact_bps(quantity=Decimal("0"), volume=Decimal("1000")) == Decimal("0")


def test_square_root_impact_rejects_zero_volume() -> None:
    from backtest_engine.market_mechanics import SquareRootImpact

    m = SquareRootImpact(k=Decimal("10"))
    with pytest.raises(ValueError):
        m.impact_bps(quantity=Decimal("1"), volume=Decimal("0"))


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


def test_latency_model_shifts_fill_timestamp() -> None:
    from backtest_engine.market_mechanics import FixedLatency

    m = FixedLatency(delay=timedelta(milliseconds=250))
    ts = datetime(2026, 4, 1, tzinfo=UTC)
    assert m.delay_fill(ts) == ts + timedelta(milliseconds=250)


def test_latency_model_rejects_negative_delay() -> None:
    from backtest_engine.market_mechanics import FixedLatency

    with pytest.raises(ValueError):
        FixedLatency(delay=timedelta(milliseconds=-1))


# ---------------------------------------------------------------------------
# Composite: apply a full chain (spread -> slippage -> impact) and verify
# ordering doesn't swallow a piece of the cost.
# ---------------------------------------------------------------------------


def test_mechanics_compose_additively() -> None:
    from backtest_engine.market_mechanics import (
        FixedBpsSlippage,
        HalfSpreadModel,
        SquareRootImpact,
    )

    spread = HalfSpreadModel(spread_bps=Decimal("10"))  # ±0.05
    slip = FixedBpsSlippage(bps=Decimal("5"))  # +0.05
    impact = SquareRootImpact(k=Decimal("10"))  # ~bps-scale

    mid = Decimal("100")
    after_spread = spread.apply(side="buy", mid=mid)
    after_slippage = slip.apply(side="buy", mid=after_spread)
    impact_bps = impact.impact_bps(quantity=Decimal("100"), volume=Decimal("10000"))
    final = after_slippage + (after_slippage * impact_bps / Decimal("10000"))
    assert final > mid
