"""Phase 9 Task 2 - Kubernetes overlay invariants.

- Overlays exist per workload: research, ingestion, api_model_serving,
  agent_runners, web_control_plane.
- Each overlay has a `kustomization.yaml` + at least one
  `deployment.yaml`.
- HFT engine must NOT appear as a Kubernetes workload
  (hft-colocation-phase-8.md rule).
- Deployments carry `securityContext.runAsNonRoot: true`.
"""

from __future__ import annotations

from pathlib import Path

K8S_ROOT = Path("infrastructure") / "kubernetes"


REQUIRED_OVERLAYS = (
    "research",
    "ingestion",
    "api_model_serving",
    "agent_runners",
    "web_control_plane",
)


def _read(repo_root: Path, relative: Path) -> str:
    path = repo_root / relative
    assert path.is_file(), f"missing {relative}"
    return path.read_text(encoding="utf-8")


def test_every_overlay_has_kustomization_and_deployment(repo_root: Path) -> None:
    for overlay in REQUIRED_OVERLAYS:
        overlay_root = repo_root / K8S_ROOT / "overlays" / overlay
        kust = overlay_root / "kustomization.yaml"
        assert kust.is_file(), f"missing {kust.relative_to(repo_root)}"
        deploys = list(overlay_root.glob("deployment*.yaml"))
        assert deploys, f"overlay {overlay} needs at least one deployment.yaml"


def test_hft_engine_is_not_a_kubernetes_overlay(repo_root: Path) -> None:
    overlays_root = repo_root / K8S_ROOT / "overlays"
    if not overlays_root.is_dir():
        return
    for item in overlays_root.iterdir():
        if not item.is_dir():
            continue
        assert "hft" not in item.name.lower(), (
            f"HFT engine must not appear as a Kubernetes overlay: {item}"
        )


def test_deployments_declare_run_as_non_root(repo_root: Path) -> None:
    for overlay in REQUIRED_OVERLAYS:
        overlay_root = repo_root / K8S_ROOT / "overlays" / overlay
        for deploy in overlay_root.glob("deployment*.yaml"):
            text = deploy.read_text(encoding="utf-8")
            assert "runAsNonRoot: true" in text, (
                f"{deploy.relative_to(repo_root)} must set runAsNonRoot: true"
            )


def test_deployments_set_resource_limits(repo_root: Path) -> None:
    for overlay in REQUIRED_OVERLAYS:
        overlay_root = repo_root / K8S_ROOT / "overlays" / overlay
        for deploy in overlay_root.glob("deployment*.yaml"):
            text = deploy.read_text(encoding="utf-8")
            assert "limits:" in text, (
                f"{deploy.relative_to(repo_root)} must declare resource limits"
            )


def test_web_control_plane_deployment_declares_auth_session_env(repo_root: Path) -> None:
    text = _read(repo_root, K8S_ROOT / "overlays" / "web_control_plane" / "deployment.yaml")
    assert "SESSION_SECRET" in text or "AUTH_SESSION_SECRET" in text
