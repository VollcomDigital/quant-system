"""HFT model-card contract.

A research model is **not** live-eligible on the HFT path until it has
a `HFTModelCard` that names a compiled-artifact target, declares a p99
inference budget, and records the training-data snapshot id.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

from shared_lib.contracts import ValidationResult

__all__ = ["CompiledTarget", "HFTModelCard", "is_live_eligible"]


CompiledTarget = Literal["onnx", "cpp_kernel", "fpga"]
_ALLOWED_TARGETS = {"onnx", "cpp_kernel", "fpga"}


@dataclass(frozen=True, slots=True)
class HFTModelCard:
    model_id: str
    version: str
    compiled_target: CompiledTarget
    p99_inference_budget_us: Decimal
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    training_data_snapshot_id: str

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if not self.version:
            raise ValueError("version must be non-empty")
        if self.compiled_target not in _ALLOWED_TARGETS:
            raise ValueError(
                f"compiled_target {self.compiled_target!r} not in "
                f"{sorted(_ALLOWED_TARGETS)}; raw PyTorch / sklearn weights "
                "are forbidden on the HFT path"
            )
        if self.p99_inference_budget_us <= Decimal("0"):
            raise ValueError("p99_inference_budget_us must be > 0")
        if not self.input_shape:
            raise ValueError("input_shape must not be empty")
        if not self.output_shape:
            raise ValueError("output_shape must not be empty")
        if any(d <= 0 for d in self.input_shape):
            raise ValueError("input_shape dims must all be > 0")
        if any(d <= 0 for d in self.output_shape):
            raise ValueError("output_shape dims must all be > 0")
        if not self.training_data_snapshot_id:
            raise ValueError(
                "training_data_snapshot_id must be non-empty so every "
                "HFT model binds to a reproducible dataset"
            )


def is_live_eligible(
    card: HFTModelCard,
    *,
    measured_p99_us: Decimal,
) -> ValidationResult:
    """Gate an HFT model's live eligibility against its measured p99."""
    if measured_p99_us < Decimal("0"):
        raise ValueError("measured_p99_us must be >= 0")
    now = datetime.now(tz=UTC)
    target = f"hft_model:{card.model_id}@{card.version}"
    if measured_p99_us > card.p99_inference_budget_us:
        return ValidationResult(
            check_id="hft.model_card.p99_budget",
            target=target,
            passed=False,
            reason=(
                f"measured p99 {measured_p99_us} us exceeds declared "
                f"budget {card.p99_inference_budget_us} us"
            ),
            evaluated_at=now,
        )
    return ValidationResult(
        check_id="hft.model_card.p99_budget",
        target=target,
        passed=True,
        reason=None,
        evaluated_at=now,
    )
