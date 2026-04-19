"""Production factor library.

A `Factor` is the production-promoted successor to a notebook draft.
It owns its metadata and a `compute(bars)` method that yields
`shared_lib.contracts.FactorRecord` instances. The `FactorLibrary`
wraps a registry + a compile-to-feature-store-definition helper so
Phase 2's `data_platform.feature_store.FactorRegistry` is always the
authoritative definition store at run time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import ClassVar, Literal

from shared_lib.contracts import Bar, FactorRecord

__all__ = [
    "Factor",
    "FactorLibrary",
    "FactorMetadata",
    "ValidationStatus",
]


ValidationStatus = Literal["candidate", "validated", "promoted", "retired"]
_VALID_STATUSES = {"candidate", "validated", "promoted", "retired"}


@dataclass(frozen=True, slots=True)
class FactorMetadata:
    factor_id: str
    version: str
    description: str
    source_dependencies: tuple[str, ...]
    stationarity_assumption: str
    universe_coverage: str
    leakage_review: str
    validation_status: ValidationStatus

    def __post_init__(self) -> None:
        if not self.factor_id:
            raise ValueError("factor_id must be non-empty")
        if not self.version:
            raise ValueError("version must be non-empty")
        if not self.description:
            raise ValueError("description must be non-empty")
        if not self.leakage_review:
            raise ValueError(
                "leakage_review must be non-empty; every promoted factor "
                "requires an explicit review note"
            )
        if self.validation_status not in _VALID_STATUSES:
            raise ValueError(
                f"invalid validation_status: {self.validation_status!r}"
            )


class Factor(ABC):
    """Base class for all production-promoted factors."""

    metadata: ClassVar[FactorMetadata]

    @abstractmethod
    def compute(self, bars: Iterable[Bar]) -> Iterator[FactorRecord]:
        ...


class FactorLibrary:
    """Registry of Factor subclasses.

    Every registered factor can be projected onto a
    `data_platform.feature_store.FactorDefinition` via `export_to`.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], Factor] = {}

    def register(self, factor: Factor) -> None:
        key = (factor.metadata.factor_id, factor.metadata.version)
        if key in self._entries:
            raise ValueError(f"factor {key} already registered")
        self._entries[key] = factor

    def get(self, factor_id: str, version: str) -> Factor:
        try:
            return self._entries[(factor_id, version)]
        except KeyError as exc:
            raise LookupError(f"no factor for {(factor_id, version)!r}") from exc

    def __iter__(self) -> Iterator[Factor]:
        return iter(self._entries.values())

    def export_to(self, registry: object) -> None:
        """Project every factor metadata into a
        `data_platform.feature_store.FactorRegistry`.
        """
        from data_platform.feature_store import FactorDefinition

        for factor in self._entries.values():
            m = factor.metadata
            registry.register(  # type: ignore[attr-defined]
                FactorDefinition(
                    factor_id=m.factor_id,
                    version=m.version,
                    description=m.description,
                    source_dependencies=m.source_dependencies,
                    universe=m.universe_coverage,
                    validation_status=m.validation_status,
                )
            )
