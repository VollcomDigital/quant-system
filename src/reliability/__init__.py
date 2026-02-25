"""Reliability metadata and data-quality gating for collections (VD-4344)."""

from .continuity import compute_continuity_score
from .schema import CollectionReliability, ReliabilityThresholds, SymbolContinuityReport

__all__ = [
    "CollectionReliability",
    "ReliabilityThresholds",
    "SymbolContinuityReport",
    "compute_continuity_score",
]
