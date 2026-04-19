"""Order Management System.

Mid-frequency OMS. Tracks order state transitions, fill aggregation,
reconciliation vs the authoritative broker snapshot. Built on Phase 1
contracts and Phase 4's `Portfolio` ledger.

State machine::

    new -> acknowledged -> partially_filled -> filled
                        \\-> cancelled
                        \\-> rejected
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from backtest_engine.simulator.portfolio import Portfolio
from shared_lib.contracts import Fill, Order
from shared_lib.math_utils import Money

__all__ = ["OMS", "OrderState", "ReconciliationDiff"]


OrderState = Literal[
    "new",
    "acknowledged",
    "partially_filled",
    "filled",
    "cancelled",
    "rejected",
]


@dataclass(frozen=True, slots=True)
class ReconciliationDiff:
    in_sync: bool
    missing_at_broker: dict[str, Decimal]
    missing_at_local: dict[str, Decimal]
    quantity_mismatches: dict[str, tuple[Decimal, Decimal]]  # symbol -> (local, broker)


_TERMINAL = {"filled", "cancelled", "rejected"}


@dataclass
class _OrderEntry:
    order: Order
    state: OrderState
    filled_quantity: Decimal


@dataclass
class OMS:
    starting_cash: Money
    require_reconciliation: bool = False
    _orders: dict[str, _OrderEntry] = field(default_factory=dict, init=False, repr=False)
    _idempotency: set[str] = field(default_factory=set, init=False, repr=False)
    _portfolio: Portfolio = field(init=False, repr=False)
    _reconciled: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._portfolio = Portfolio(starting_cash=self.starting_cash)

    def submit(self, order: Order) -> None:
        if self.require_reconciliation and not self._reconciled:
            raise RuntimeError(
                "reconciliation with broker positions is required before new orders"
            )
        if order.idempotency_key in self._idempotency:
            raise ValueError(
                f"duplicate idempotency key: {order.idempotency_key!r}"
            )
        if order.order_id in self._orders:
            raise ValueError(f"order {order.order_id!r} already submitted")
        self._idempotency.add(order.idempotency_key)
        self._orders[order.order_id] = _OrderEntry(
            order=order,
            state="acknowledged",
            filled_quantity=Decimal("0"),
        )

    def cancel(self, order_id: str) -> None:
        entry = self._lookup(order_id)
        if entry.state in _TERMINAL:
            raise ValueError(
                f"invalid transition: {entry.state!r} -> cancelled"
            )
        entry.state = "cancelled"

    def apply_fill(self, fill: Fill) -> None:
        entry = self._lookup(fill.order_id)
        new_filled = entry.filled_quantity + fill.quantity
        if new_filled > entry.order.quantity:
            raise ValueError(
                f"fill exceeds order quantity: {new_filled} > {entry.order.quantity}"
            )
        self._portfolio.apply_fill(fill)
        entry.filled_quantity = new_filled
        if new_filled == entry.order.quantity:
            entry.state = "filled"
        else:
            entry.state = "partially_filled"

    def get_state(self, order_id: str) -> OrderState:
        return self._lookup(order_id).state

    def _lookup(self, order_id: str) -> _OrderEntry:
        try:
            return self._orders[order_id]
        except KeyError as exc:
            raise LookupError(f"no order for {order_id!r}") from exc

    def positions(self) -> dict[str, Decimal]:
        return {
            symbol: pos.quantity
            for symbol, pos in self._portfolio._positions.items()
            if pos.quantity != 0
        }

    def reconcile(self, *, broker_positions: dict[str, Decimal]) -> ReconciliationDiff:
        local = self.positions()
        missing_at_broker: dict[str, Decimal] = {}
        missing_at_local: dict[str, Decimal] = {}
        mismatches: dict[str, tuple[Decimal, Decimal]] = {}

        for symbol, qty in local.items():
            if symbol not in broker_positions:
                missing_at_broker[symbol] = qty
            elif broker_positions[symbol] != qty:
                mismatches[symbol] = (qty, broker_positions[symbol])

        for symbol, qty in broker_positions.items():
            if symbol not in local:
                missing_at_local[symbol] = qty

        in_sync = not (missing_at_broker or missing_at_local or mismatches)
        self._reconciled = True
        return ReconciliationDiff(
            in_sync=in_sync,
            missing_at_broker=missing_at_broker,
            missing_at_local=missing_at_local,
            quantity_mismatches=mismatches,
        )
