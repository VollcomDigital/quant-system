"""Risk Management System.

Deterministic pre-trade risk engine. Every `check(order, context)` call
returns a single `shared_lib.contracts.ValidationResult` so the OMS,
the web control plane, and the risk_monitor agent consume results
uniformly.

ADR-0004 Layer 2: pre-trade risk engine. This is where orders die
when guardrails disagree with them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from shared_lib.contracts import Fill, Order, ValidationResult

__all__ = ["RMS", "RiskContext", "RiskLimits"]


@dataclass(frozen=True, slots=True)
class RiskLimits:
    max_notional_per_order: Decimal
    max_position_per_symbol: Decimal
    max_gross_exposure: Decimal
    max_pending_orders: int
    max_volume_per_hour: Decimal
    daily_drawdown_halt_pct: Decimal  # negative number, e.g. -0.05 == -5%
    wash_trading_window_sec: int
    ai_confidence_floor: Decimal

    def __post_init__(self) -> None:
        for name, value in (
            ("max_notional_per_order", self.max_notional_per_order),
            ("max_position_per_symbol", self.max_position_per_symbol),
            ("max_gross_exposure", self.max_gross_exposure),
            ("max_volume_per_hour", self.max_volume_per_hour),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.max_pending_orders < 0:
            raise ValueError("max_pending_orders must be >= 0")
        if self.wash_trading_window_sec < 0:
            raise ValueError("wash_trading_window_sec must be >= 0")
        if not (Decimal("0") <= self.ai_confidence_floor <= Decimal("1")):
            raise ValueError("ai_confidence_floor must be in [0, 1]")
        if self.daily_drawdown_halt_pct > 0:
            raise ValueError("daily_drawdown_halt_pct must be <= 0")


@dataclass(frozen=True, slots=True)
class RiskContext:
    current_positions: dict[str, Decimal]
    gross_exposure: Decimal
    pending_orders: int
    recent_fills: tuple[Fill, ...]
    daily_pnl_pct: Decimal
    ai_confidence: Decimal
    drift_flag: bool
    trading_halted: bool
    now: datetime


@dataclass(frozen=True, slots=True)
class RMS:
    limits: RiskLimits

    def check(self, order: Order, context: RiskContext) -> ValidationResult:
        if context.trading_halted:
            return _vr("rms.trading_halted", order, False, "global TRADING_HALTED flag set")
        if context.drift_flag:
            return _vr("rms.drift", order, False, "drift_flag set; refusing new orders")
        if context.ai_confidence < self.limits.ai_confidence_floor:
            return _vr(
                "rms.ai_confidence",
                order,
                False,
                f"ai_confidence {context.ai_confidence} below floor {self.limits.ai_confidence_floor}",
            )
        if context.daily_pnl_pct <= self.limits.daily_drawdown_halt_pct:
            return _vr(
                "rms.daily_drawdown",
                order,
                False,
                f"daily drawdown {context.daily_pnl_pct} breached halt "
                f"{self.limits.daily_drawdown_halt_pct}",
            )
        if context.pending_orders >= self.limits.max_pending_orders:
            return _vr(
                "rms.pending_orders",
                order,
                False,
                f"pending_orders {context.pending_orders} >= max {self.limits.max_pending_orders}",
            )
        notional = (order.limit_price or Decimal("0")) * order.quantity
        if notional > self.limits.max_notional_per_order:
            return _vr(
                "rms.fat_finger_notional",
                order,
                False,
                f"order notional {notional} > max {self.limits.max_notional_per_order}",
            )
        # Position cap (only for buys; sells reduce a long position or grow
        # a short position within its own cap, but Phase 6 mid-frequency
        # treats the symbol-cap as an absolute long-side ceiling).
        signed_qty = order.quantity if order.side == "buy" else -order.quantity
        projected = context.current_positions.get(order.symbol, Decimal("0")) + signed_qty
        if abs(projected) > self.limits.max_position_per_symbol:
            return _vr(
                "rms.position_cap",
                order,
                False,
                f"projected position {projected} exceeds cap "
                f"{self.limits.max_position_per_symbol}",
            )
        # Wash trading - opposite side within window.
        for fill in context.recent_fills:
            if fill.symbol != order.symbol:
                continue
            if fill.side == order.side:
                continue
            if (context.now - fill.filled_at).total_seconds() <= self.limits.wash_trading_window_sec:
                return _vr(
                    "rms.wash_trading",
                    order,
                    False,
                    f"wash-trading: opposing {fill.side} fill within "
                    f"{self.limits.wash_trading_window_sec}s",
                )
        # Volume per hour.
        one_hour = 3600.0
        volume_last_hour = sum(
            (f.quantity for f in context.recent_fills
             if f.symbol == order.symbol
             and (context.now - f.filled_at).total_seconds() <= one_hour),
            start=Decimal("0"),
        )
        if volume_last_hour + order.quantity > self.limits.max_volume_per_hour:
            return _vr(
                "rms.volume_per_hour",
                order,
                False,
                f"projected hourly volume {volume_last_hour + order.quantity} "
                f"> max {self.limits.max_volume_per_hour}",
            )
        return _vr("rms.ok", order, True, None)


def _vr(check_id: str, order: Order, passed: bool, reason: str | None) -> ValidationResult:
    return ValidationResult(
        check_id=check_id,
        target=f"order:{order.order_id}",
        passed=passed,
        reason=reason,
        evaluated_at=datetime.now(tz=UTC),
    )
