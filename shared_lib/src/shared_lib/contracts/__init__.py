"""Shared cross-package domain schemas.

Every cross-service payload in the monorepo is typed through one of these
pydantic v2 models. Breaking changes require a new ADR and a new contract
version.
"""

from __future__ import annotations

from shared_lib.contracts.approvals import ApprovalDecision, ApprovalRequest
from shared_lib.contracts.audit import AuditEvent
from shared_lib.contracts.factors import FactorRecord
from shared_lib.contracts.market_data import Bar
from shared_lib.contracts.memory import ResearchMemoryRecord
from shared_lib.contracts.optimization import OptimizerRequest, OptimizerResponse
from shared_lib.contracts.orders import Fill, Order, Position
from shared_lib.contracts.predictions import PredictionArtifact
from shared_lib.contracts.rl import RLEnvironmentMetadata
from shared_lib.contracts.runs import JobStatus, RunMetadata
from shared_lib.contracts.signals import TradeSignal
from shared_lib.contracts.status import ExecutionStatus, HealthStatus
from shared_lib.contracts.validation import AnomalyEvent, ValidationResult

__all__ = [
    "AnomalyEvent",
    "ApprovalDecision",
    "ApprovalRequest",
    "AuditEvent",
    "Bar",
    "ExecutionStatus",
    "FactorRecord",
    "Fill",
    "HealthStatus",
    "JobStatus",
    "OptimizerRequest",
    "OptimizerResponse",
    "Order",
    "Position",
    "PredictionArtifact",
    "RLEnvironmentMetadata",
    "ResearchMemoryRecord",
    "RunMetadata",
    "TradeSignal",
    "ValidationResult",
]
