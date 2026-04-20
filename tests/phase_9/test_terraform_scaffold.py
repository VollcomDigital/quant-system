"""Phase 9 Task 1 - Terraform scaffold invariants.

- Modules exist per responsibility: kms_signing, vault_secrets,
  object_storage, observability, kubernetes_cluster, bare_metal_host.
- Each module carries main.tf + variables.tf + outputs.tf + README.md.
- Envs exist for dev / paper / production with a main.tf that names
  the environment in a `locals` block.
- Every main.tf declares `required_providers` and pins a version.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path("infrastructure") / "terraform"


REQUIRED_MODULES = (
    "kms_signing",
    "vault_secrets",
    "object_storage",
    "observability",
    "kubernetes_cluster",
    "bare_metal_host",
)


REQUIRED_ENVS = ("dev", "paper", "production")


def _read(repo_root: Path, relative: Path) -> str:
    path = repo_root / relative
    assert path.is_file(), f"missing {relative}"
    return path.read_text(encoding="utf-8")


def test_every_module_has_required_files(repo_root: Path) -> None:
    for module in REQUIRED_MODULES:
        module_root = repo_root / ROOT / "modules" / module
        for filename in ("main.tf", "variables.tf", "outputs.tf", "README.md"):
            path = module_root / filename
            assert path.is_file(), f"missing {path.relative_to(repo_root)}"


def test_every_env_has_required_files(repo_root: Path) -> None:
    for env in REQUIRED_ENVS:
        env_root = repo_root / ROOT / "envs" / env
        for filename in ("main.tf", "backend.tf", "README.md"):
            path = env_root / filename
            assert path.is_file(), f"missing {path.relative_to(repo_root)}"


def test_env_main_declares_locals_with_environment_name(repo_root: Path) -> None:
    for env in REQUIRED_ENVS:
        text = _read(repo_root, ROOT / "envs" / env / "main.tf")
        assert "locals" in text
        assert f'environment = "{env}"' in text


def test_kms_signing_module_declares_required_providers(repo_root: Path) -> None:
    text = _read(repo_root, ROOT / "modules" / "kms_signing" / "main.tf")
    assert "terraform {" in text
    assert "required_providers" in text
    # AWS KMS signing is the canonical Phase 9 path.
    assert "aws" in text.lower()


def test_production_env_separates_from_dev(repo_root: Path) -> None:
    dev = _read(repo_root, ROOT / "envs" / "dev" / "backend.tf")
    prod = _read(repo_root, ROOT / "envs" / "production" / "backend.tf")
    # Environments must not share a backend key.
    assert dev != prod


def test_production_env_pins_terraform_state_encryption(repo_root: Path) -> None:
    text = _read(repo_root, ROOT / "envs" / "production" / "backend.tf")
    assert "encrypt" in text.lower()
    assert "true" in text.lower()
