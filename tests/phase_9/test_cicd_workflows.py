"""Phase 9 Task 3 - CI/CD workflow invariants.

- `.github/workflows/ci.yml` runs lint + tests + coverage.
- `.github/workflows/container-scan.yml` runs image scanning.
- `.github/workflows/iac-scan.yml` runs Terraform / Kubernetes scanning.
- `.github/workflows/factor-promotion.yml` runs the Phase 3 promotion
  gate + the Phase 5 code_reviewer agent on every PR touching factors.
- `.github/workflows/model-deployment.yml` gates model promotion with
  MLflow-style registry metadata.
- `.github/workflows/staged-deploy.yml` promotes dev -> paper ->
  production with explicit approval environments.
- Every workflow pins runner images and declares explicit `permissions`.
"""

from __future__ import annotations

from pathlib import Path

WORKFLOWS_ROOT = Path(".github") / "workflows"


REQUIRED_WORKFLOWS = (
    "ci.yml",
    "container-scan.yml",
    "iac-scan.yml",
    "factor-promotion.yml",
    "model-deployment.yml",
    "staged-deploy.yml",
)


def _read(repo_root: Path, filename: str) -> str:
    path = repo_root / WORKFLOWS_ROOT / filename
    assert path.is_file(), f"missing workflow {filename}"
    return path.read_text(encoding="utf-8")


def test_every_required_workflow_exists(repo_root: Path) -> None:
    for filename in REQUIRED_WORKFLOWS:
        text = _read(repo_root, filename)
        assert "name:" in text
        assert "on:" in text
        assert "jobs:" in text


def test_ci_workflow_runs_lint_tests_and_coverage(repo_root: Path) -> None:
    text = _read(repo_root, "ci.yml")
    assert "ruff" in text.lower()
    assert "pytest" in text.lower()
    assert "--cov" in text


def test_container_scan_uses_trivy_or_equivalent(repo_root: Path) -> None:
    text = _read(repo_root, "container-scan.yml").lower()
    assert "trivy" in text or "grype" in text or "snyk" in text


def test_iac_scan_covers_terraform_and_kubernetes(repo_root: Path) -> None:
    text = _read(repo_root, "iac-scan.yml").lower()
    assert "terraform" in text or "tflint" in text or "tfsec" in text or "checkov" in text
    assert "kubernetes" in text or "kubesec" in text or "kubescape" in text or "kube" in text


def test_factor_promotion_runs_code_reviewer_and_promotion_gate(repo_root: Path) -> None:
    text = _read(repo_root, "factor-promotion.yml").lower()
    assert "code_reviewer" in text or "code-reviewer" in text
    assert "promote_factor" in text or "promotion" in text


def test_model_deployment_references_registry_metadata(repo_root: Path) -> None:
    text = _read(repo_root, "model-deployment.yml").lower()
    assert "modelregistry" in text or "model_registry" in text or "mlflow" in text


def test_staged_deploy_uses_environments(repo_root: Path) -> None:
    text = _read(repo_root, "staged-deploy.yml")
    # GitHub Actions "environment:" gate per env.
    for env in ("dev", "paper", "production"):
        assert f"environment: {env}" in text or f"name: {env}" in text


def test_every_workflow_declares_explicit_permissions(repo_root: Path) -> None:
    for filename in REQUIRED_WORKFLOWS:
        text = _read(repo_root, filename)
        assert "permissions:" in text, f"{filename} must declare explicit permissions"
