from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _tests_present(root: Path) -> bool:
    tests_dir = root / "tests"
    if not tests_dir.is_dir():
        return False
    return any(tests_dir.glob("*.py"))


def _maybe_arch_prefix(cmd: list[str]) -> list[str]:
    if sys.platform != "darwin":
        return cmd
    try:
        translated = subprocess.check_output(
            ["sysctl", "-n", "sysctl.proc_translated"], text=True
        ).strip()
    except Exception:
        return cmd
    if translated != "1":
        return cmd
    if shutil.which("arch") is None:
        return cmd
    return ["arch", "-arm64", *cmd]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if not _tests_present(root):
        print("No tests found; skipping coverage")
        return 0

    env = os.environ.copy()
    env.setdefault("COVERAGE_FILE", str(Path(tempfile.gettempdir()) / ".coverage"))
    env.setdefault("SKIP_PANDAS_TESTS", "1")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(root)
        if not existing_pythonpath
        else f"{root}{os.pathsep}{existing_pythonpath}"
    )

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-q",
        "--maxfail=1",
        "--disable-warnings",
        "--cov=src.backtest.results_cache",
        "--cov=src.data.cache",
        "--cov=src.utils.http",
        "--cov=src.reporting.html",
        "--cov-report=term-missing",
        "--cov-fail-under=80",
    ]
    cmd = _maybe_arch_prefix(cmd)
    result = subprocess.run(cmd, env=env, cwd=root)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
