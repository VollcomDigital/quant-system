"""Phase 8 Task 1 - native HFT scaffold invariants.

The Phase 0 scaffold test guarded the directory shape; Phase 8 locks
in the content shape:

- A workspace `Cargo.toml` at `trading_system/native/Cargo.toml` lists
  the HFT member crates.
- Each crate has its own `Cargo.toml` and at least one source file so
  cargo can resolve the workspace.
- No `.py` files exist under `trading_system/native/` (Python exclusion
  on the HFT critical path, ADR-0003).
- The fpga/ directory exists as a reserved subtree (software baselines
  come first; FPGA is the last leg).
"""

from __future__ import annotations

from pathlib import Path

NATIVE_ROOT = Path("trading_system") / "native"
HFT_ROOT = NATIVE_ROOT / "hft_engine"


REQUIRED_CRATES = (
    "core",
    "network",
    "fast_inference",
)


def _read(repo_root: Path, relative: Path) -> str:
    path = repo_root / relative
    assert path.is_file(), f"missing {relative}"
    return path.read_text(encoding="utf-8")


def test_workspace_cargo_toml_lists_required_crates(repo_root: Path) -> None:
    text = _read(repo_root, NATIVE_ROOT / "Cargo.toml")
    for crate in REQUIRED_CRATES:
        assert crate in text, f"workspace Cargo.toml missing crate {crate!r}"
    assert "[workspace]" in text


def test_each_crate_has_cargo_toml_and_src(repo_root: Path) -> None:
    for crate in REQUIRED_CRATES:
        cargo = repo_root / HFT_ROOT / crate / "Cargo.toml"
        assert cargo.is_file(), f"missing Cargo.toml for {crate}"
        text = cargo.read_text(encoding="utf-8")
        assert f'name = "{crate}"' in text, f"Cargo.toml for {crate} must declare its name"

        src_dir = repo_root / HFT_ROOT / crate / "src"
        assert src_dir.is_dir(), f"missing src dir for {crate}"
        lib = src_dir / "lib.rs"
        assert lib.is_file(), f"missing src/lib.rs for {crate}"


def test_fpga_directory_is_reserved_but_empty_of_python(repo_root: Path) -> None:
    fpga = repo_root / HFT_ROOT / "fpga"
    assert fpga.is_dir(), "fpga/ must exist as a reserved subtree"


def test_no_python_files_anywhere_under_native(repo_root: Path) -> None:
    native = repo_root / NATIVE_ROOT
    offenders: list[str] = []
    for py in native.rglob("*.py"):
        offenders.append(str(py.relative_to(repo_root)))
    assert not offenders, (
        f"Python source is forbidden under trading_system/native/ (ADR-0003); "
        f"offenders: {offenders}"
    )


def test_workspace_declares_strict_release_profile(repo_root: Path) -> None:
    text = _read(repo_root, NATIVE_ROOT / "Cargo.toml")
    assert "[profile.release]" in text
    # Latency-critical release profile expectations.
    assert "lto" in text.lower() or "codegen-units" in text.lower()
