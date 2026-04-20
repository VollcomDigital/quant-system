"""Phase 8 Exit Criteria.

- Python is excluded from the HFT critical path by design.
- Benchmark + replay harnesses exist before any live-path consideration.
- HFT runtime contracts integrate with shared risk and gateway
  boundaries without bypassing them.
"""

from __future__ import annotations

from decimal import Decimal


def test_exit_python_exclusion_enforced() -> None:
    import re
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    native = repo_root / "trading_system" / "native"
    for py in native.rglob("*.py"):
        raise AssertionError(f"Python file under native/: {py}")

    pattern = re.compile(
        r"^\s*(?:from|import)\s+trading_system\.native(?:\.|\s)",
        re.MULTILINE,
    )
    for domain in (
        "shared_lib",
        "data_platform",
        "backtest_engine",
        "alpha_research",
        "ai_agents",
        "trading_system",
    ):
        root = repo_root / domain / "src"
        if not root.is_dir():
            continue
        for py in root.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            assert not pattern.search(text), f"domain import from native/: {py}"


def test_exit_benchmark_and_replay_exist() -> None:
    from trading_system.hft_engine.benchmark import (  # noqa: F401
        LatencyBudget,
        LatencyReport,
        enforce_budget,
        summarise_latency,
    )
    from trading_system.shared_gateways.replay import (  # noqa: F401
        HeartbeatTracker,
        detect_gaps,
        replay_sequenced,
    )


def test_exit_hft_model_card_integrates_with_rms_validation() -> None:
    from shared_lib.contracts import ValidationResult
    from trading_system.hft_engine.model_card import HFTModelCard, is_live_eligible

    card = HFTModelCard(
        model_id="m",
        version="v1",
        compiled_target="onnx",
        p99_inference_budget_us=Decimal("10"),
        input_shape=(1, 64),
        output_shape=(1, 3),
        training_data_snapshot_id="snap-exit",
    )
    result = is_live_eligible(card, measured_p99_us=Decimal("7"))
    # ValidationResult is the shared contract with Phase 6 RMS + Phase 5
    # risk monitor agent; the HFT engine reuses it rather than inventing
    # its own rejection shape.
    assert isinstance(result, ValidationResult)
