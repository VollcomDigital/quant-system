"""Feature store contracts.

One in-memory reference implementation is provided; production swaps a
Parquet / Iceberg / Delta backend while keeping the same surface.

Contract rules:

- Factor definitions are immutable per `(factor_id, version)`.
- Every `FactorRecord` written must reference an existing definition.
- Reads return records filtered by factor_id, version, symbol, window.
- If multiple versions exist for a factor_id, reads must specify one.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from shared_lib.contracts import FactorRecord

__all__ = [
    "FactorDefinition",
    "FactorRegistry",
    "FeatureStore",
    "ValidationStatus",
]


ValidationStatus = Literal["candidate", "validated", "promoted", "retired"]


@dataclass(frozen=True, slots=True)
class FactorDefinition:
    factor_id: str
    version: str
    description: str
    source_dependencies: tuple[str, ...]
    universe: str
    validation_status: ValidationStatus

    def __post_init__(self) -> None:
        if not self.factor_id:
            raise ValueError("factor_id must be non-empty")
        if not self.version:
            raise ValueError("version must be non-empty")
        if self.validation_status not in {
            "candidate",
            "validated",
            "promoted",
            "retired",
        }:
            raise ValueError(
                f"invalid validation_status: {self.validation_status!r}"
            )


@dataclass
class FactorRegistry:
    _defns: dict[tuple[str, str], FactorDefinition] = field(
        default_factory=dict, init=False, repr=False
    )

    def register(self, defn: FactorDefinition) -> None:
        key = (defn.factor_id, defn.version)
        if key in self._defns:
            raise ValueError(f"factor {key} already registered")
        self._defns[key] = defn

    def get(self, factor_id: str, version: str) -> FactorDefinition:
        try:
            return self._defns[(factor_id, version)]
        except KeyError as exc:
            raise LookupError(f"unknown factor {(factor_id, version)!r}") from exc

    def list(self, factor_id: str) -> list[FactorDefinition]:
        return [d for (fid, _), d in self._defns.items() if fid == factor_id]

    def has(self, factor_id: str, version: str) -> bool:
        return (factor_id, version) in self._defns

    def has_factor(self, factor_id: str) -> bool:
        return any(fid == factor_id for (fid, _) in self._defns)

    def versions(self, factor_id: str) -> list[str]:
        return [v for (fid, v) in self._defns if fid == factor_id]


@dataclass
class FeatureStore:
    registry: FactorRegistry
    _rows: list[FactorRecord] = field(default_factory=list, init=False, repr=False)

    def write(self, records: Iterable[FactorRecord]) -> None:
        rows = list(records)
        for r in rows:
            if self.registry.has(r.factor_id, r.version):
                continue
            if self.registry.has_factor(r.factor_id):
                raise LookupError(
                    f"unknown version {r.version!r} for factor {r.factor_id!r}"
                )
            raise LookupError(f"unknown factor {r.factor_id!r}")
        self._rows.extend(rows)

    def read(
        self,
        *,
        factor_id: str,
        version: str | None = None,
        symbol: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Iterator[FactorRecord]:
        if version is None and len(self.registry.versions(factor_id)) > 1:
            raise ValueError(
                f"factor {factor_id!r} has multiple versions; specify one"
            )
        for r in self._rows:
            if r.factor_id != factor_id:
                continue
            if version is not None and r.version != version:
                continue
            if symbol is not None and r.symbol != symbol:
                continue
            if start is not None and r.as_of < start:
                continue
            if end is not None and r.as_of >= end:
                continue
            yield r
