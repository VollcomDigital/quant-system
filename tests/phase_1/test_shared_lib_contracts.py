"""Phase 1 Task 4 - shared_lib.contracts domain schemas.

Every cross-package payload in the monorepo has a pydantic v2 contract.
These tests assert the minimum invariants for each schema category:
construction, field validation, currency/Decimal typing, enum closure, and
timestamp semantics.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

# ---------------------------------------------------------------------------
# Market data bars
# ---------------------------------------------------------------------------


def test_bar_round_trip() -> None:
    from shared_lib.contracts import Bar

    bar = Bar(
        symbol="AAPL",
        interval="1d",
        timestamp=datetime(2026, 4, 19, tzinfo=UTC),
        open=Decimal("170.10"),
        high=Decimal("172.00"),
        low=Decimal("169.50"),
        close=Decimal("171.25"),
        volume=Decimal("12345678"),
    )
    as_json = bar.model_dump_json()
    rebuilt = Bar.model_validate_json(as_json)
    assert rebuilt == bar


def test_bar_rejects_naive_timestamp() -> None:
    from shared_lib.contracts import Bar

    with pytest.raises(ValueError):
        Bar(
            symbol="AAPL",
            interval="1d",
            timestamp=datetime(2026, 4, 19),  # naive -> rejected
            open=Decimal("1"),
            high=Decimal("1"),
            low=Decimal("1"),
            close=Decimal("1"),
            volume=Decimal("0"),
        )


def test_bar_high_must_not_be_below_low() -> None:
    from shared_lib.contracts import Bar

    with pytest.raises(ValueError, match="high"):
        Bar(
            symbol="AAPL",
            interval="1d",
            timestamp=datetime(2026, 4, 19, tzinfo=UTC),
            open=Decimal("1"),
            high=Decimal("1"),
            low=Decimal("2"),
            close=Decimal("1"),
            volume=Decimal("0"),
        )


# ---------------------------------------------------------------------------
# Factor frames
# ---------------------------------------------------------------------------


def test_factor_record_basic() -> None:
    from shared_lib.contracts import FactorRecord

    rec = FactorRecord(
        factor_id="mom_12_1",
        as_of=datetime(2026, 4, 19, tzinfo=UTC),
        symbol="AAPL",
        value=Decimal("0.1234"),
        version="v1",
    )
    assert rec.factor_id == "mom_12_1"


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


def test_prediction_artifact_requires_model_metadata() -> None:
    from shared_lib.contracts import PredictionArtifact

    art = PredictionArtifact(
        model_id="kronos-v0.1",
        symbol="AAPL",
        horizon="1d",
        generated_at=datetime(2026, 4, 19, tzinfo=UTC),
        value=Decimal("0.005"),
        confidence=Decimal("0.7"),
    )
    assert art.model_id == "kronos-v0.1"


def test_prediction_confidence_must_be_between_0_and_1() -> None:
    from shared_lib.contracts import PredictionArtifact

    with pytest.raises(ValueError):
        PredictionArtifact(
            model_id="x",
            symbol="AAPL",
            horizon="1d",
            generated_at=datetime(2026, 4, 19, tzinfo=UTC),
            value=Decimal("0"),
            confidence=Decimal("1.1"),
        )


# ---------------------------------------------------------------------------
# Portfolio optimizer
# ---------------------------------------------------------------------------


def test_optimizer_request_rejects_weights_over_1() -> None:
    from shared_lib.contracts import OptimizerRequest

    with pytest.raises(ValueError):
        OptimizerRequest(
            request_id="r1",
            universe=["AAPL", "MSFT"],
            objective="mean_cvar",
            gross_leverage=Decimal("1.0"),
            bounds={"AAPL": (Decimal("0"), Decimal("1.5"))},  # upper>1 rejected
            risk_aversion=Decimal("1"),
        )


def test_optimizer_response_weights_sum_to_1() -> None:
    from shared_lib.contracts import OptimizerResponse

    resp = OptimizerResponse(
        request_id="r1",
        weights={"AAPL": Decimal("0.6"), "MSFT": Decimal("0.4")},
        objective_value=Decimal("0.01"),
    )
    assert resp.weights["AAPL"] + resp.weights["MSFT"] == Decimal("1.0")


def test_optimizer_response_rejects_weights_not_summing_to_1() -> None:
    from shared_lib.contracts import OptimizerResponse

    with pytest.raises(ValueError, match="sum"):
        OptimizerResponse(
            request_id="r1",
            weights={"AAPL": Decimal("0.7"), "MSFT": Decimal("0.4")},
            objective_value=Decimal("0"),
        )


# ---------------------------------------------------------------------------
# Research memory
# ---------------------------------------------------------------------------


def test_research_memory_record_roundtrip() -> None:
    from shared_lib.contracts import ResearchMemoryRecord

    rec = ResearchMemoryRecord(
        record_id="mem-1",
        kind="factor_hypothesis",
        title="Momentum in small caps",
        body="...",
        tags=("momentum", "small_cap"),
        created_at=datetime(2026, 4, 19, tzinfo=UTC),
    )
    assert rec.kind == "factor_hypothesis"


def test_research_memory_rejects_unknown_kind() -> None:
    from shared_lib.contracts import ResearchMemoryRecord

    with pytest.raises(ValueError):
        ResearchMemoryRecord(
            record_id="mem-1",
            kind="bogus",  # not in enum
            title="x",
            body="x",
            tags=(),
            created_at=datetime(2026, 4, 19, tzinfo=UTC),
        )


# ---------------------------------------------------------------------------
# RL metadata
# ---------------------------------------------------------------------------


def test_rl_environment_metadata_basic() -> None:
    from shared_lib.contracts import RLEnvironmentMetadata

    meta = RLEnvironmentMetadata(
        env_id="tradig-v0",
        observation_space_shape=(4,),
        action_space_kind="continuous",
        action_space_bounds=(Decimal("-1"), Decimal("1")),
        reward_scale=Decimal("1"),
    )
    assert meta.action_space_kind == "continuous"


def test_rl_environment_rejects_zero_obs_dim() -> None:
    from shared_lib.contracts import RLEnvironmentMetadata

    with pytest.raises(ValueError):
        RLEnvironmentMetadata(
            env_id="x",
            observation_space_shape=(0,),
            action_space_kind="discrete",
            action_space_bounds=None,
            reward_scale=Decimal("1"),
        )


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------


def test_run_metadata_and_job_status() -> None:
    from shared_lib.contracts import JobStatus, RunMetadata

    run = RunMetadata(
        run_id="run-1",
        kind="backtest",
        started_at=datetime(2026, 4, 19, tzinfo=UTC),
        git_sha="deadbeef",
    )
    status = JobStatus(run_id="run-1", state="succeeded", progress=Decimal("1.0"))
    assert run.kind == "backtest"
    assert status.state == "succeeded"


def test_job_status_rejects_bad_state() -> None:
    from shared_lib.contracts import JobStatus

    with pytest.raises(ValueError):
        JobStatus(run_id="r", state="melted", progress=Decimal("0"))


# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------


def test_approval_request_and_decision() -> None:
    from shared_lib.contracts import ApprovalDecision, ApprovalRequest

    req = ApprovalRequest(
        approval_id="app-1",
        subject="factor_promotion",
        target_id="mom_12_1",
        requested_by="alice",
        requested_at=datetime(2026, 4, 19, tzinfo=UTC),
        context={"coverage_pct": "0.85"},
    )
    dec = ApprovalDecision(
        approval_id="app-1",
        decision="approved",
        decided_by="bob",
        decided_at=datetime(2026, 4, 19, tzinfo=UTC),
        notes="LGTM",
    )
    assert req.subject == "factor_promotion"
    assert dec.decision == "approved"


def test_approval_decision_rejects_unknown_outcome() -> None:
    from shared_lib.contracts import ApprovalDecision

    with pytest.raises(ValueError):
        ApprovalDecision(
            approval_id="x",
            decision="kinda",
            decided_by="bob",
            decided_at=datetime(2026, 4, 19, tzinfo=UTC),
            notes="",
        )


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def test_audit_event_operator_action() -> None:
    from shared_lib.contracts import AuditEvent

    ev = AuditEvent(
        event_id="aud-1",
        actor="operator:alice",
        action="halt_strategy",
        target="strategy:mom_12_1",
        at=datetime(2026, 4, 19, tzinfo=UTC),
        trace_id="t1",
        details={"reason": "drawdown_breach"},
    )
    assert ev.action == "halt_strategy"


# ---------------------------------------------------------------------------
# Status payloads
# ---------------------------------------------------------------------------


def test_execution_status() -> None:
    from shared_lib.contracts import ExecutionStatus

    st = ExecutionStatus(
        service="trading_system.oms",
        state="running",
        last_heartbeat=datetime(2026, 4, 19, tzinfo=UTC),
        open_orders=3,
        pending_fills=1,
    )
    assert st.state == "running"


def test_health_status_degrades_when_any_subsystem_fails() -> None:
    from shared_lib.contracts import HealthStatus

    st = HealthStatus(
        service="data_platform",
        ok=False,
        checks={"polygon": True, "tiingo": False},
    )
    # Contract: `ok` must reflect `all(checks.values())`.
    assert st.ok is False


def test_health_status_ok_must_match_checks() -> None:
    from shared_lib.contracts import HealthStatus

    with pytest.raises(ValueError, match="ok"):
        HealthStatus(
            service="data_platform",
            ok=True,
            checks={"polygon": True, "tiingo": False},
        )


# ---------------------------------------------------------------------------
# Trade signals, orders, fills, positions
# ---------------------------------------------------------------------------


def test_trade_signal_defaults() -> None:
    from shared_lib.contracts import TradeSignal

    sig = TradeSignal(
        signal_id="sig-1",
        strategy_id="mom_12_1",
        symbol="AAPL",
        direction="long",
        strength=Decimal("0.5"),
        generated_at=datetime(2026, 4, 19, tzinfo=UTC),
    )
    assert sig.direction == "long"


def test_trade_signal_direction_enum_is_closed() -> None:
    from shared_lib.contracts import TradeSignal

    with pytest.raises(ValueError):
        TradeSignal(
            signal_id="x",
            strategy_id="s",
            symbol="AAPL",
            direction="kinda-long",
            strength=Decimal("0.1"),
            generated_at=datetime(2026, 4, 19, tzinfo=UTC),
        )


def test_order_fill_position_roundtrip() -> None:
    from shared_lib.contracts import Fill, Order, Position

    order = Order(
        order_id="o-1",
        idempotency_key="idem-1",
        symbol="AAPL",
        side="buy",
        quantity=Decimal("100"),
        limit_price=Decimal("170.50"),
        time_in_force="day",
        placed_at=datetime(2026, 4, 19, tzinfo=UTC),
    )
    fill = Fill(
        fill_id="f-1",
        order_id="o-1",
        symbol="AAPL",
        side="buy",
        quantity=Decimal("100"),
        price=Decimal("170.40"),
        fee=Decimal("0.50"),
        currency="USD",
        filled_at=datetime(2026, 4, 19, 14, tzinfo=UTC),
    )
    pos = Position(
        symbol="AAPL",
        quantity=Decimal("100"),
        avg_price=Decimal("170.40"),
        currency="USD",
        as_of=datetime(2026, 4, 19, 14, tzinfo=UTC),
    )
    assert order.side == "buy"
    assert fill.price == Decimal("170.40")
    assert pos.quantity == Decimal("100")


def test_order_rejects_zero_quantity() -> None:
    from shared_lib.contracts import Order

    with pytest.raises(ValueError):
        Order(
            order_id="x",
            idempotency_key="y",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("0"),
            limit_price=None,
            time_in_force="day",
            placed_at=datetime(2026, 4, 19, tzinfo=UTC),
        )


# ---------------------------------------------------------------------------
# Validation + anomaly events
# ---------------------------------------------------------------------------


def test_validation_result_fail_requires_reason() -> None:
    from shared_lib.contracts import ValidationResult

    with pytest.raises(ValueError, match="reason"):
        ValidationResult(
            check_id="leak_check",
            target="factor:mom_12_1",
            passed=False,
            reason=None,  # fail must carry reason
            evaluated_at=datetime(2026, 4, 19, tzinfo=UTC),
        )


def test_anomaly_event_basic() -> None:
    from shared_lib.contracts import AnomalyEvent

    ev = AnomalyEvent(
        anomaly_id="an-1",
        source="rms",
        severity="high",
        summary="Drawdown > 5% in 1 hour",
        detected_at=datetime(2026, 4, 19, tzinfo=UTC),
        details={"drawdown": "0.053"},
    )
    assert ev.severity == "high"


def test_anomaly_severity_enum_closed() -> None:
    from shared_lib.contracts import AnomalyEvent

    with pytest.raises(ValueError):
        AnomalyEvent(
            anomaly_id="x",
            source="rms",
            severity="catastrophic",
            summary="",
            detected_at=datetime(2026, 4, 19, tzinfo=UTC),
            details={},
        )
