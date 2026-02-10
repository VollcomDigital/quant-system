from __future__ import annotations

import json
import os
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

if "src.backtest.runner" not in sys.modules:
    runner_stub = types.ModuleType("src.backtest.runner")

    class _PreImportRunner:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("BacktestRunner stub not patched")

    runner_stub.BacktestRunner = _PreImportRunner
    runner_stub.BestResult = SimpleNamespace
    sys.modules["src.backtest.runner"] = runner_stub

import src.main
from src.main import app

runner = CliRunner()


def _patch_common(monkeypatch: pytest.MonkeyPatch, export_calls: dict[str, int]):
    def factory(label: str):
        class _Exporter:
            def __init__(self, *args, **kwargs):
                pass

            def export(self, *args, **kwargs):
                export_calls[label] = export_calls.get(label, 0) + 1

        return _Exporter

    class _DashboardReporter:
        def __init__(self, *args, **kwargs):
            pass

        def export(self, payload):  # pragma: no cover - simple side effect
            payload["exported"] = True

    class _HealthReporter:
        def __init__(self, *args, **kwargs):
            self.failures = None

        def export(self, failures):
            self.failures = failures

    monkeypatch.setattr("src.main.CSVExporter", factory("csv"))
    monkeypatch.setattr("src.main.AllCSVExporter", factory("all_csv"))
    monkeypatch.setattr("src.main.MarkdownReporter", factory("markdown"))
    monkeypatch.setattr("src.main.TradingViewExporter", factory("tradingview"))
    monkeypatch.setattr("src.main.HTMLReporter", factory("html"))
    monkeypatch.setattr("src.main.DashboardReporter", _DashboardReporter)
    monkeypatch.setattr("src.main.HealthReporter", _HealthReporter)


def _patch_manifest(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "src.main.build_dashboard_payload",
        lambda cache, run_id, results: {
            "run_id": run_id,
            "rows": [],
            "summary": {"counts": {"results": len(results)}},
            "available_metrics": ["sharpe"],
            "highlights": {},
        },
    )
    monkeypatch.setattr(
        "src.main.collect_runs_manifest",
        lambda reports_root, run_id, summary, meta: [{"run_id": "old-run", "summary": summary}],
    )
    monkeypatch.setattr(
        "src.main.refresh_manifest",
        lambda reports_root, base_out, cache, payload: [
            {"run_id": "old-run", "message": "refreshed"}
        ],
    )


@pytest.fixture(autouse=True)
def _stub_ccxt(monkeypatch: pytest.MonkeyPatch):
    if "ccxt" not in sys.modules:
        sys.modules["ccxt"] = types.ModuleType("ccxt")
    yield


def _write_config(tmp_path: Path) -> Path:
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
strategies:
  - name: stratA
    module: dummy
    class: DummyStrategy
    params: {}
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)
    return path


def _run_args(tmp_path: Path) -> list[str]:
    return [
        "run",
        "--config",
        str(_write_config(tmp_path)),
        "--output-dir",
        str(tmp_path / "reports"),
        "--strategies-path",
        str(tmp_path / "strategies"),
    ]


def _make_cache_file(path: Path, age_days: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("cache")
    past = time.time() - (age_days * 86400) - 60
    os.utime(path, (past, past))


def test_run_command_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "strategies").mkdir()
    export_calls: dict[str, int] = {}

    class DummyRunner:
        def __init__(self, cfg, strategies_root, run_id):
            self.cfg = cfg
            self.strategies_root = strategies_root
            self.run_id = run_id
            self.external_index = {"stratA": object()}
            self.results_cache = object()
            self.failures = [
                {
                    "collection": "test",
                    "symbol": "AAPL",
                    "timeframe": "1d",
                    "source": "yfinance",
                    "error": "network",
                }
            ]

        def run_all(self, only_cached: bool = False):
            return [
                SimpleNamespace(
                    collection="test",
                    symbol="AAPL",
                    timeframe="1d",
                    strategy="stratA",
                    params={"x": 1},
                    metric_name="sharpe",
                    metric_value=1.5,
                    stats={"sharpe": 1.5},
                )
            ]

    monkeypatch.setattr("src.main.BacktestRunner", DummyRunner)
    _patch_common(monkeypatch, export_calls)
    _patch_manifest(monkeypatch)
    monkeypatch.setattr(
        "src.main.notify_all",
        lambda results, cfg, run_id: [
            {"channel": "slack", "sent": True, "metric": "sharpe", "value": 1.6},
            {"channel": "slack", "sent": False, "metric": "sortino", "reason": "skipped"},
        ],
    )

    result = runner.invoke(app, _run_args(tmp_path))
    assert result.exit_code == 0

    reports_dir = tmp_path / "reports"
    run_dirs = list(reports_dir.iterdir())
    assert len(run_dirs) == 1
    summary_path = run_dirs[0] / "summary.json"
    summary_data = json.loads(summary_path.read_text())
    assert summary_data["results_count"] == 1
    assert summary_data["manifest_refresh"]
    assert export_calls["csv"] == 1
    assert export_calls["markdown"] == 1


def test_run_command_no_strategies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "strategies").mkdir()

    class EmptyRunner:
        def __init__(self, cfg, strategies_root, run_id):
            self.external_index = {}
            self.results_cache = object()
            self.failures = []

        def run_all(self, only_cached: bool = False):
            return []

    monkeypatch.setattr("src.main.BacktestRunner", EmptyRunner)
    result = runner.invoke(app, _run_args(tmp_path))
    assert result.exit_code == 1
    assert "No strategies discovered" in result.stdout


def test_run_command_no_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "strategies").mkdir()

    class NoResultsRunner:
        def __init__(self, cfg, strategies_root, run_id):
            self.external_index = {"stratA": object()}
            self.results_cache = object()
            self.failures = []

        def run_all(self, only_cached: bool = False):
            return []

    monkeypatch.setattr("src.main.BacktestRunner", NoResultsRunner)
    result = runner.invoke(app, _run_args(tmp_path))
    assert result.exit_code == 2
    assert "No backtest results produced" in result.output


def test_discover_symbols_writes_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "src.main.discover_ccxt_symbols",
        lambda opts: [("ETH/USDT", 1000.0)],
    )

    output_path = tmp_path / "universe.yaml"
    result = runner.invoke(
        app,
        [
            "discover-symbols",
            "--output",
            str(output_path),
            "--exchange",
            "binance",
            "--quote",
            "USDT",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert "ETH/USDT" in output_path.read_text()


def test_discover_symbols_merges_exchanges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fake(opts):
        if opts.exchange == "binance":
            return [("ETH/USDT", 900.0), ("BTC/USDT", 800.0)]
        return [("ETH/USDT", 1200.0), ("XRP/USDT", 500.0)]

    monkeypatch.setattr("src.main.discover_ccxt_symbols", fake)

    result = runner.invoke(
        app,
        [
            "discover-symbols",
            "--exchange",
            "binance",
            "--exchange",
            "bybit",
            "--top-n",
            "2",
            "--annotate",
        ],
    )

    assert result.exit_code == 0
    config = yaml.safe_load(result.stdout)
    symbols = config["collections"][0]["symbols"]
    assert symbols == ["ETH/USDT", "BTC/USDT"]
    liquidity = config["liquidity"]
    assert liquidity[0]["symbol"] == "ETH/USDT"
    assert liquidity[0]["exchange"] == "bybit"
    assert liquidity[0]["volume"] == pytest.approx(1200.0)
    assert liquidity[1]["symbol"] == "BTC/USDT"


def test_discover_symbols_exclusions_and_extras(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "src.main.discover_ccxt_symbols",
        lambda opts: [("AAA/USDT", 100.0), ("BBB/USDT", 90.0), ("CCC/USDT", 80.0)],
    )

    result = runner.invoke(
        app,
        [
            "discover-symbols",
            "--exchange",
            "binance",
            "--top-n",
            "2",
            "--exclude-symbol",
            "BBB/USDT",
            "--exclude-pattern",
            "CCC/*",
            "--extra-symbol",
            "MANUAL/USDT",
        ],
    )

    assert result.exit_code == 0
    config = yaml.safe_load(result.stdout)
    symbols = config["collections"][0]["symbols"]
    assert "BBB/USDT" not in symbols
    assert "CCC/USDT" not in symbols
    assert "AAA/USDT" in symbols
    assert "MANUAL/USDT" in symbols


def test_fundamentals_command_outputs_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class DummySource:
        def __init__(self, cache_dir):
            self.cache_dir = cache_dir

        def fetch_fundamentals(self, symbol):
            return {"symbol": symbol, "info": {"sector": "tech"}}

    monkeypatch.setattr("src.main.YFinanceSource", DummySource)

    output_path = tmp_path / "fundamentals.yaml"
    result = runner.invoke(
        app,
        [
            "fundamentals",
            "AAPL",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--format",
            "yaml",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    data = yaml.safe_load(output_path.read_text())
    assert data["symbol"] == "AAPL"
    assert data["info"]["sector"] == "tech"


def test_ingest_data_invokes_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, str, bool]] = []

    class DummySource:
        def __init__(self, cache_dir):
            self.cache_dir = cache_dir

        def fetch(self, symbol: str, timeframe: str, only_cached: bool = False):
            calls.append((symbol, timeframe, only_cached))

    monkeypatch.setitem(src.main.DATA_SOURCES, "dummy", DummySource)

    result = runner.invoke(
        app,
        [
            "ingest-data",
            "--source",
            "dummy",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--timeframe",
            "1d",
            "--timeframe",
            "4h",
            "AAPL",
            "MSFT",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        ("AAPL", "1d", False),
        ("AAPL", "4h", False),
        ("MSFT", "1d", False),
        ("MSFT", "4h", False),
    ]


def test_list_strategies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "strategies").mkdir()
    monkeypatch.setattr(
        "src.main.discover_external_strategies",
        lambda root: {"alpha": object(), "beta": object()},
    )

    result = runner.invoke(
        app,
        [
            "list-strategies",
            "--strategies-path",
            str(tmp_path / "strategies"),
        ],
    )

    assert result.exit_code == 0
    assert "alpha" in result.stdout
    assert "beta" in result.stdout


def test_package_run_invokes_make_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import shutil

    reports_dir = tmp_path / "reports"
    run_dir = reports_dir / "latest"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}")

    made_archives = {}

    def fake_make_archive(base_name: str, fmt: str, root_dir: Path):
        made_archives["base_name"] = base_name
        made_archives["format"] = fmt
        made_archives["root_dir"] = Path(root_dir)
        return f"{base_name}.{fmt}"

    monkeypatch.setattr(shutil, "make_archive", fake_make_archive)

    result = runner.invoke(
        app,
        [
            "package-run",
            "latest",
            "--reports-dir",
            str(reports_dir),
            "--output",
            str(tmp_path / "archive.zip"),
        ],
    )

    assert result.exit_code == 0
    assert made_archives["root_dir"] == run_dir
    assert "Packaged run" in result.stdout


def test_package_run_rejects_path_traversal(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    (reports_dir / "latest").mkdir(parents=True)

    result = runner.invoke(
        app,
        [
            "package-run",
            "../outside",
            "--reports-dir",
            str(reports_dir),
        ],
    )

    assert result.exit_code != 0
    assert "run_id must be a simple name" in result.stdout or "Invalid run_id" in result.stdout


def test_clean_cache_removes_old_files(tmp_path: Path):
    cache_dir = tmp_path / "cache" / "data"
    results_dir = tmp_path / "cache" / "results"
    old_data = cache_dir / "old.parquet"
    new_data = cache_dir / "new.parquet"
    old_results = results_dir / "metrics.json"

    _make_cache_file(old_data, age_days=10)
    _make_cache_file(new_data, age_days=1)
    _make_cache_file(old_results, age_days=15)

    result = runner.invoke(
        app,
        [
            "clean-cache",
            "--cache-dir",
            str(cache_dir),
            "--results-cache-dir",
            str(results_dir),
            "--max-age-days",
            "7",
        ],
    )

    assert result.exit_code == 0
    assert not old_data.exists()
    assert new_data.exists()
    assert not old_results.exists()


def test_clean_cache_dry_run(tmp_path: Path):
    cache_dir = tmp_path / "cache" / "data"
    target = cache_dir / "stale.parquet"
    _make_cache_file(target, age_days=20)

    result = runner.invoke(
        app,
        [
            "clean-cache",
            "--cache-dir",
            str(cache_dir),
            "--max-age-days",
            "5",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "DRY-RUN" in result.stdout
    assert target.exists()


def test_clean_cache_does_not_follow_symlink_dirs(tmp_path: Path):
    cache_dir = tmp_path / "cache" / "data"
    cache_dir.mkdir(parents=True)

    # File that should be deleted (old + inside cache root).
    stale = cache_dir / "stale.parquet"
    _make_cache_file(stale, age_days=20)

    # File that must NOT be deleted (outside cache root).
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "keep.txt"
    _make_cache_file(outside_file, age_days=20)

    # Symlink inside cache root pointing outside.
    link_dir = cache_dir / "link-outside"
    os.symlink(outside_dir, link_dir)

    result = runner.invoke(
        app,
        [
            "clean-cache",
            "--cache-dir",
            str(cache_dir),
            "--max-age-days",
            "7",
            "--no-include-results",
        ],
    )

    assert result.exit_code == 0
    assert not stale.exists()
    assert outside_file.exists()
