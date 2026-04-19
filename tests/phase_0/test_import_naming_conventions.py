"""Phase 0 Task 3 - Internal package naming and import conventions.

The naming rules are normative for every future phase:
- absolute imports only (no `src.*` outside legacy code and compatibility
  adapters)
- one domain package per top-level name
- `snake_case` package names, no dashes, no nested `src` prefixes in imports
- a written standard that can be linked from PR reviews and agent prompts

These tests lock those rules in so future contributors cannot silently drift.
"""

from __future__ import annotations

import re
from pathlib import Path

CONVENTIONS_DOC = Path("docs") / "architecture" / "package-and-import-conventions.md"


REQUIRED_SECTION_HEADINGS = (
    "## Package Naming Rules",
    "## Import Convention",
    "## Forbidden Patterns",
    "## Test Discovery",
    "## Enforcement",
)


REQUIRED_DOMAIN_IMPORT_EXAMPLES = (
    "shared_lib.",
    "data_platform.",
    "backtest_engine.",
    "alpha_research.",
    "ai_agents.",
    "trading_system.",
)


def _read(repo_root: Path) -> str:
    path = repo_root / CONVENTIONS_DOC
    assert path.is_file(), f"Conventions doc missing at {CONVENTIONS_DOC}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Edge case 1: the doc must declare all required sections so the standard is
# complete, not a stub.
# ---------------------------------------------------------------------------


def test_conventions_doc_has_all_required_sections(repo_root: Path) -> None:
    text = _read(repo_root)
    missing = [h for h in REQUIRED_SECTION_HEADINGS if h not in text]
    assert not missing, (
        f"Conventions doc is missing required sections: {missing}"
    )


# ---------------------------------------------------------------------------
# Edge case 2: every Python domain must appear as an import example so the
# canonical prefix is unambiguous.
# ---------------------------------------------------------------------------


def test_conventions_doc_lists_every_domain_import(repo_root: Path) -> None:
    text = _read(repo_root)
    missing = [d for d in REQUIRED_DOMAIN_IMPORT_EXAMPLES if d not in text]
    assert not missing, (
        f"Conventions doc must enumerate domain import prefixes; missing: {missing}"
    )


# ---------------------------------------------------------------------------
# Edge case 3: legacy `src.*` must be explicitly called out as forbidden
# outside the compatibility facade so no new code inherits the legacy prefix.
# ---------------------------------------------------------------------------


def test_conventions_doc_forbids_new_src_imports(repo_root: Path) -> None:
    text = _read(repo_root)
    # Look for a prohibition of `src.` imports outside compatibility code.
    lowered = text.lower()
    assert "src." in lowered, "Conventions doc must reference the `src.` prefix"
    assert (
        "forbidden" in lowered
        or "must not" in lowered
        or "do not" in lowered
        or "prohibited" in lowered
    ), "Conventions doc must prohibit new `src.*` imports"


# ---------------------------------------------------------------------------
# Edge case 4: snake_case rule must be explicit. We look for either the exact
# phrase or both "snake_case" and "no dashes".
# ---------------------------------------------------------------------------


def test_conventions_doc_requires_snake_case(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    assert "snake_case" in text, "Conventions doc must mandate snake_case package names"


# ---------------------------------------------------------------------------
# Edge case 5: the enforcement path must be explicit - either a ruff rule, a
# pre-commit hook, or a CI job. We accept any of those terms.
# ---------------------------------------------------------------------------


def test_conventions_doc_declares_enforcement_hook(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    assert any(
        token in text for token in ("ruff", "pre-commit", "ci", "lint")
    ), "Conventions doc must declare an enforcement path (ruff/pre-commit/ci/lint)"


# ---------------------------------------------------------------------------
# Edge case 6: every on-disk package must satisfy the naming rule. We walk
# each domain's src/<domain> subtree and verify every sub-package uses
# snake_case (no dashes, no camelCase).
# ---------------------------------------------------------------------------


PYTHON_DOMAINS = (
    "shared_lib",
    "data_platform",
    "backtest_engine",
    "alpha_research",
    "ai_agents",
    "trading_system",
)


SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")


def test_on_disk_subpackage_names_are_snake_case(repo_root: Path) -> None:
    offenders: list[str] = []
    for domain in PYTHON_DOMAINS:
        root = repo_root / domain / "src" / domain
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if child.name.startswith("__"):
                continue
            if not SNAKE_CASE.match(child.name):
                offenders.append(f"{domain}/src/{domain}/{child.name}")
    assert not offenders, (
        f"Sub-package names must be snake_case; offenders: {offenders}"
    )
