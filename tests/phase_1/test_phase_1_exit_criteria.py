"""Phase 1 Exit Criteria - aggregate gate.

Phase 1 exit criteria (from `tasks/todo.md`):

- Shared schemas are importable without `src/*` coupling.
- Telemetry baseline is reusable across batch jobs, APIs, agents, and
  execution paths.
- Decimal-safe money primitives exist before OMS work begins.
"""

from __future__ import annotations


def test_exit_shared_contracts_import_cleanly() -> None:
    # Importing any contract must never trigger a src.* import.
    import sys

    # shared_lib.contracts and its submodules must load without pulling
    # anything under the legacy src package.
    import shared_lib.contracts  # noqa: F401

    loaded_src = {name for name in sys.modules if name == "src" or name.startswith("src.")}
    assert loaded_src == set(), f"shared_lib.contracts pulled in legacy src modules: {loaded_src}"


def test_exit_telemetry_supports_every_profile() -> None:
    from shared_lib.telemetry import SUPPORTED_PROFILES, bootstrap

    assert SUPPORTED_PROFILES == frozenset({"batch", "api", "agent", "live"})
    for profile in SUPPORTED_PROFILES:
        assert bootstrap(service=f"exit-{profile}", profile=profile).profile == profile


def test_exit_decimal_money_primitive_exists() -> None:
    from shared_lib.math_utils import Money

    # Used directly to confirm the public surface is stable before OMS work.
    assert Money("1.00", "USD") + Money("2.00", "USD") == Money("3.00", "USD")


def test_exit_transport_envelopes_available() -> None:
    from shared_lib.transport import EventEnvelope, RpcEnvelope, dlq_topic  # noqa: F401

    assert dlq_topic("trading_system.fills.v1") == "trading_system.fills.v1.dlq.v1"
