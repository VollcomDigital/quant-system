"""Phase 9 Exit Criteria.

- Deployment patterns exist for research, data, agents, model serving,
  and trading control-plane services.
- Signing/custody model is documented in deployable infrastructure
  terms.
- Out-of-band hard-kill path is documented and testable.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_exit_deployment_patterns_cover_all_workloads() -> None:
    overlays = REPO_ROOT / "infrastructure" / "kubernetes" / "overlays"
    for overlay in (
        "research",
        "ingestion",
        "api_model_serving",
        "agent_runners",
        "web_control_plane",
    ):
        kust = overlays / overlay / "kustomization.yaml"
        assert kust.is_file(), f"missing overlay {overlay}"


def test_exit_custody_doc_is_deployable_infrastructure() -> None:
    doc = REPO_ROOT / "docs" / "architecture" / "custody-treasury-phase-9.md"
    text = doc.read_text(encoding="utf-8")
    # Signing / custody + a concrete Terraform module reference so
    # "deployable" means something.
    assert "kms_signing" in text or "KMS" in text
    assert "Terraform" in text or "terraform" in text


def test_exit_hard_kill_path_is_testable() -> None:
    from trading_system.hard_kill import (  # noqa: F401
        HardKillRequest,
        execute_hard_kill,
    )
