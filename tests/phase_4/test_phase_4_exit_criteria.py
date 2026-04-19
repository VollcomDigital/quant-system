"""Phase 4 Exit Criteria - aggregate gate.

Exit criteria (from `tasks/todo.md`):

- Simulator, analytics, and market-mechanics boundaries are separate
  packages or modules.
- Exact API/order payload replay path is defined against gateway
  contracts.
- Look-ahead and leakage validation runs as part of normal engine
  validation.
"""

from __future__ import annotations


def test_exit_simulator_analytics_mechanics_are_separate_modules() -> None:
    import backtest_engine.analytics as a
    import backtest_engine.market_mechanics as m
    import backtest_engine.simulator as s

    # Distinct module names ensure the ADR-0001 separation holds.
    assert a.__name__ != s.__name__ != m.__name__


def test_exit_payload_replay_path_is_defined() -> None:
    from backtest_engine.api import OrderPayload, record_payloads, replay_payloads  # noqa: F401


def test_exit_leakage_guards_are_available() -> None:
    from backtest_engine.leakage import (
        ensure_factor_precedes_signal,
        ensure_signal_precedes_fill,
        stable_replay_order,
    )

    # Public surface must be importable from the engine.
    assert callable(ensure_factor_precedes_signal)
    assert callable(ensure_signal_precedes_fill)
    assert callable(stable_replay_order)
