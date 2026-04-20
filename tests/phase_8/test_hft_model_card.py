"""Phase 8 Task 2 - HFT model-card contract.

A research model is *not* live-eligible on the HFT path until it has a
`HFTModelCard` that:

- Names a compiled-artifact target in {`onnx`, `cpp_kernel`, `fpga`}.
- Declares a measured p99 inference budget in microseconds.
- Enumerates the input + output schema shapes so the Phase 4 simulator
  replay can drive it.
- Records the training-data snapshot id so weights trace back to a
  reproducible dataset.
"""

from __future__ import annotations

from decimal import Decimal

import pytest


def _card(**kw):
    from trading_system.hft_engine.model_card import HFTModelCard

    defaults = dict(
        model_id="m",
        version="v1",
        compiled_target="onnx",
        p99_inference_budget_us=Decimal("10"),
        input_shape=(1, 64),
        output_shape=(1, 3),
        training_data_snapshot_id="snap-1",
    )
    defaults.update(kw)
    return HFTModelCard(**defaults)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_model_card_accepts_valid_onnx_target() -> None:
    card = _card(compiled_target="onnx")
    assert card.compiled_target == "onnx"


def test_model_card_accepts_valid_cpp_target() -> None:
    card = _card(compiled_target="cpp_kernel")
    assert card.compiled_target == "cpp_kernel"


def test_model_card_accepts_valid_fpga_target() -> None:
    card = _card(compiled_target="fpga")
    assert card.compiled_target == "fpga"


def test_model_card_rejects_uncompiled_target() -> None:
    with pytest.raises(ValueError, match="compiled_target"):
        _card(compiled_target="pytorch")  # live HFT refuses raw PyTorch


def test_model_card_rejects_non_positive_budget() -> None:
    with pytest.raises(ValueError):
        _card(p99_inference_budget_us=Decimal("0"))
    with pytest.raises(ValueError):
        _card(p99_inference_budget_us=Decimal("-1"))


def test_model_card_requires_training_snapshot() -> None:
    with pytest.raises(ValueError):
        _card(training_data_snapshot_id="")


def test_model_card_rejects_empty_input_shape() -> None:
    with pytest.raises(ValueError):
        _card(input_shape=())


def test_model_card_rejects_non_positive_dim() -> None:
    with pytest.raises(ValueError):
        _card(input_shape=(1, 0))


# ---------------------------------------------------------------------------
# Live-eligibility gate
# ---------------------------------------------------------------------------


def test_live_eligibility_rejects_budget_below_measured() -> None:
    """`is_live_eligible` checks that a measured p99 is within the declared
    budget."""
    from trading_system.hft_engine.model_card import is_live_eligible

    card = _card(p99_inference_budget_us=Decimal("5"))
    # Measured p99 of 8 us > declared 5 us -> ineligible.
    result = is_live_eligible(card, measured_p99_us=Decimal("8"))
    assert result.passed is False
    assert "budget" in (result.reason or "")


def test_live_eligibility_happy_path() -> None:
    from trading_system.hft_engine.model_card import is_live_eligible

    card = _card(p99_inference_budget_us=Decimal("10"))
    result = is_live_eligible(card, measured_p99_us=Decimal("7"))
    assert result.passed is True


def test_live_eligibility_refuses_negative_measurement() -> None:
    from trading_system.hft_engine.model_card import is_live_eligible

    with pytest.raises(ValueError):
        is_live_eligible(_card(), measured_p99_us=Decimal("-1"))
