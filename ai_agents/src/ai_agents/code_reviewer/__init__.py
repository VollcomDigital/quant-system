"""Code reviewer agent prototype.

Static checks that run in CI. Each check emits a
`shared_lib.contracts.ValidationResult` so the approval queue can
display a uniform list of findings.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from shared_lib.contracts import ValidationResult

__all__ = [
    "review_module_has_tests",
    "review_source",
]


_LOOK_AHEAD_PATTERNS = (
    (
        "look_ahead.shift_negative",
        re.compile(r"\.shift\s*\(\s*-\s*\d+"),
        "uses `.shift(-N)` which exposes future data",
    ),
    (
        "look_ahead.future_column",
        re.compile(r"\bfuture_[a-zA-Z_]+\b"),
        "reads a column named `future_*` directly",
    ),
)


_FORBIDDEN_IMPORT_PATTERN = re.compile(
    r"^\s*(?:from|import)\s+(?P<ns>(?:src|oms|kms|treasury))(?:\.|\s)",
    re.MULTILINE,
)


def _vr(check_id: str, passed: bool, reason: str | None) -> ValidationResult:
    return ValidationResult(
        check_id=check_id,
        target="pr_diff",
        passed=passed,
        reason=reason,
        evaluated_at=datetime.now(tz=UTC),
    )


def review_source(source: str) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    for check_id, pattern, reason in _LOOK_AHEAD_PATTERNS:
        if pattern.search(source):
            results.append(_vr(check_id, False, reason))
        else:
            results.append(_vr(check_id, True, None))
    m = _FORBIDDEN_IMPORT_PATTERN.search(source)
    if m:
        results.append(
            _vr(
                "imports.forbidden_namespace",
                False,
                f"forbidden namespace {m.group('ns')!r} imported in domain code",
            )
        )
    else:
        results.append(_vr("imports.forbidden_namespace", True, None))
    return results


def review_module_has_tests(
    *,
    source_file: Path,
    candidate_test_files: list[Path],
) -> ValidationResult:
    text = source_file.read_text(encoding="utf-8")
    public_fns = re.findall(r"^def\s+([a-zA-Z][a-zA-Z0-9_]*)\s*\(", text, re.MULTILINE)
    public_fns = [f for f in public_fns if not f.startswith("_")]
    if not public_fns:
        return _vr("tests.public_function", True, None)
    if not candidate_test_files:
        return _vr(
            "tests.public_function",
            False,
            f"public functions {public_fns} have no matching test file",
        )
    combined = "\n".join(
        p.read_text(encoding="utf-8") for p in candidate_test_files if p.is_file()
    )
    missing = [fn for fn in public_fns if fn not in combined]
    if missing:
        return _vr(
            "tests.public_function",
            False,
            f"public functions {missing} have no reference in test files",
        )
    return _vr("tests.public_function", True, None)
