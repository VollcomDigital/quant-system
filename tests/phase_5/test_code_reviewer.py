"""Phase 5 Task 7 - code_reviewer agent prototype.

Runs a set of static checks on a PR diff (or a snippet of source code)
and returns `ValidationResult` objects. Heuristics target the highest-
value review findings named in the roadmap:

- look-ahead bias patterns (e.g. `shift(-1)`, accessing `future_*`).
- data leakage patterns (`pd.merge` without `how='left'` asof concerns).
- missing tests (public function with no matching test_ file).
- direct imports of forbidden namespaces (src.*, oms.*, kms.*).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Look-ahead / leakage patterns.
# ---------------------------------------------------------------------------


def test_code_reviewer_flags_shift_negative_one() -> None:
    from ai_agents.code_reviewer import review_source

    diff = """
        df["signal"] = df["price"].shift(-1)
    """
    results = review_source(diff)
    ids = {r.check_id for r in results if not r.passed}
    assert "look_ahead.shift_negative" in ids


def test_code_reviewer_flags_future_column_access() -> None:
    from ai_agents.code_reviewer import review_source

    diff = """
        df["signal"] = df["future_return"]
    """
    results = review_source(diff)
    ids = {r.check_id for r in results if not r.passed}
    assert "look_ahead.future_column" in ids


def test_code_reviewer_flags_forbidden_namespace_import() -> None:
    from ai_agents.code_reviewer import review_source

    diff = "from src.backtest import runner"
    results = review_source(diff)
    ids = {r.check_id for r in results if not r.passed}
    assert "imports.forbidden_namespace" in ids


def test_code_reviewer_flags_kms_import_from_agent() -> None:
    from ai_agents.code_reviewer import review_source

    diff = "from kms import sign_blob"
    results = review_source(diff)
    ids = {r.check_id for r in results if not r.passed}
    assert "imports.forbidden_namespace" in ids


def test_code_reviewer_accepts_clean_diff() -> None:
    from ai_agents.code_reviewer import review_source

    diff = """
        from shared_lib.contracts import Bar

        def compute(bars):
            return [b.close for b in bars]
    """
    results = review_source(diff)
    assert all(r.passed for r in results)


# ---------------------------------------------------------------------------
# Missing-tests heuristic: a public function without a matching test_* file.
# ---------------------------------------------------------------------------


def test_code_reviewer_flags_public_function_without_test(tmp_path) -> None:
    from ai_agents.code_reviewer import review_module_has_tests

    # Simulate a source file with a public function but no matching tests.
    (tmp_path / "src.py").write_text("def compute(x):\n    return x\n")
    result = review_module_has_tests(
        source_file=tmp_path / "src.py",
        candidate_test_files=[],
    )
    assert result.passed is False


def test_code_reviewer_accepts_when_test_file_exists(tmp_path) -> None:
    from ai_agents.code_reviewer import review_module_has_tests

    (tmp_path / "src.py").write_text("def compute(x):\n    return x\n")
    (tmp_path / "test_src.py").write_text("def test_compute():\n    pass\n")
    result = review_module_has_tests(
        source_file=tmp_path / "src.py",
        candidate_test_files=[tmp_path / "test_src.py"],
    )
    assert result.passed is True
