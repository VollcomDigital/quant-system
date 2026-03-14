from __future__ import annotations

from decimal import Decimal
from math import isfinite
from types import SimpleNamespace

import pandas as pd
import pytest

from src.backtest.runner import BacktestRunner, BestResult, GateDecision
from src.config import (
    CollectionConfig,
    Config,
    OptimizationPolicyConfig,
    StrategyConfig,
    ValidationConfig,
    ValidationDataQualityConfig,
)
from src.strategies.base import BaseStrategy


@pytest.fixture(autouse=True)
def _skip_if_numpy_reload_detected():
    try:
        pd.Series([1.0, 2.0, 3.0]).mean()
    except TypeError:
        pytest.skip(
            "NumPy reload detected under coverage; skipping pandas-heavy tests",
            allow_module_level=False,
        )


class _StubResultsCache:
    def __init__(self):
        self.saved = []
        self.retrieved = []

    def get(
        self,
        *,
        collection: str,
        symbol: str,
        timeframe: str,
        strategy: str,
        params: dict,
        data_fingerprint: str,
        fees: float,
        slippage: float,
        metric_name: str,
    ):
        self.retrieved.append(
            {
                "collection": collection,
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy": strategy,
                "params": params,
                "fingerprint": data_fingerprint,
                "fees": fees,
                "slippage": slippage,
                "metric": metric_name,
            }
        )
        return None

    def set(self, **kwargs):
        self.saved.append(kwargs)


class _StubEvaluationCache:
    def __init__(self):
        self.saved = []
        self.retrieved = []

    def hash_mode_config(self, mode_config):
        payload = mode_config.payload if hasattr(mode_config, "payload") else {}
        return f"{mode_config.mode}:{sorted(payload.items())}"

    def get(self, **kwargs):
        self.retrieved.append(kwargs)
        return None

    def set(self, **kwargs):
        self.saved.append(kwargs)


class _DummyStrategy(BaseStrategy):
    name = "dummy"

    def param_grid(self) -> dict[str, list[int]]:
        return {"window": [1, 2]}

    def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
        entries = pd.Series([True] + [False] * (len(df.index) - 1), index=df.index)
        exits = pd.Series([False] * (len(df.index) - 1) + [True], index=df.index)
        return entries, exits


def _make_runner(
    tmp_path,
    monkeypatch,
    collections=None,
    metric="sharpe",
    patch_pybroker=True,
    patch_sim=True,
    patch_source=True,
):
    cfg = Config(
        collections=collections
        or [
            CollectionConfig(
                name="demo",
                source="custom",
                symbols=["AAPL"],
                fees=0.0004,
                slippage=0.0003,
            )
        ],
        timeframes=["1d"],
        metric=metric,
        strategies=[
            StrategyConfig(
                name="dummy",
                module=None,
                cls=None,
                params={"grid": {"window": [3, 4]}, "bias": 0.1},
            )
        ],
        engine="pybroker",
        param_search="grid",
        param_trials=3,
        max_workers=1,
        asset_workers=1,
        param_workers=1,
        max_fetch_concurrency=1,
        fees=0.0,
        slippage=0.0,
        risk_free_rate=0.01,
        cache_dir=str(tmp_path / "cache"),
        notifications=None,
    )

    monkeypatch.setattr(
        "src.backtest.runner.discover_external_strategies",
        lambda root: {"dummy": _DummyStrategy},
    )

    runner = BacktestRunner(cfg, strategies_root=tmp_path, run_id="test-run")

    class _StubSource:
        def fetch(self, symbol, timeframe, only_cached=False):
            dates = pd.date_range("2024-01-01", periods=5, freq="D")
            data = pd.DataFrame(
                {
                    "Open": [10, 11, 12, 13, 14],
                    "High": [11, 12, 13, 14, 15],
                    "Low": [9, 10, 11, 12, 13],
                    "Close": [10.5, 11.5, 12.5, 13.5, 14.5],
                    "Volume": [100, 110, 120, 130, 140],
                },
                index=dates,
            )
            return data

    if patch_source:
        monkeypatch.setattr(BacktestRunner, "_make_source", lambda self, col: _StubSource())

    class _Val:
        def __init__(self, value: str):
            self.value = value

    class _Enum:
        SYMBOL = _Val("symbol")
        DATE = _Val("date")
        OPEN = _Val("open")
        HIGH = _Val("high")
        LOW = _Val("low")
        CLOSE = _Val("close")
        VOLUME = _Val("volume")

    if patch_pybroker:

        def _fake_ensure(self):
            StrategyCls = SimpleNamespace

            class _Config:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

            class _FeeMode:
                ORDER_PERCENT = "order_percent"

            class _PriceType:
                CLOSE = "close"

            return StrategyCls, _Config, _FeeMode, _PriceType, _Enum

        monkeypatch.setattr(BacktestRunner, "_ensure_pybroker", _fake_ensure)

    returns = pd.Series(
        [0.01, -0.005, 0.02, 0.015, -0.01], index=pd.date_range("2024-01-01", periods=5, freq="D")
    )
    equity = (1 + returns.fillna(0.0)).cumprod()
    stats = {
        "sharpe": 1.0,
        "sortino": 0.8,
        "omega": 1.2,
        "tail_ratio": 1.1,
        "profit": 0.1,
        "pain_index": 0.02,
        "trades": 2,
        "max_drawdown": -0.05,
        "cagr": 0.12,
        "calmar": -2.4,
        "equity_curve": [],
        "drawdown_curve": [],
        "trades_log": [],
    }

    if patch_sim:

        def _fake_sim(self, *args, **kwargs):
            return returns, equity, stats

        monkeypatch.setattr(BacktestRunner, "_run_pybroker_simulation", _fake_sim)

    runner.results_cache = _StubResultsCache()
    runner.evaluation_cache = _StubEvaluationCache()
    runner.mode_config_hash = runner.evaluation_cache.hash_mode_config(runner.mode_config)
    return runner


def test_bars_per_year_various_units():
    assert BacktestRunner._bars_per_year("1d") == 252
    assert BacktestRunner._bars_per_year("2w") == 26
    assert BacktestRunner._bars_per_year("1wk") == 52
    assert BacktestRunner._bars_per_year("1mo") == 12
    assert BacktestRunner._bars_per_year("4h") == pytest.approx(int(round(24 * 365 / 4)))
    assert BacktestRunner._bars_per_year("15m") == pytest.approx(int(round(60 * 24 * 365 / 15)))
    assert BacktestRunner._bars_per_year("10s") == pytest.approx(
        int(round(60 * 60 * 24 * 365 / 10))
    )


def test_timeframe_to_timedelta_supports_common_aliases():
    assert BacktestRunner._timeframe_to_timedelta("1wk") == pd.Timedelta(weeks=1)
    assert BacktestRunner._timeframe_to_timedelta("1mo") == pd.Timedelta(days=30)


@pytest.mark.parametrize(
    ("idx", "values", "expected"),
    [
        (
            pd.date_range("2024-01-01", periods=5, freq="D"),
            [1, 2, 3, 4, 5],
            {
                "score": pytest.approx(1.0),
                "coverage_ratio": pytest.approx(1.0),
                "missing_bars": 0,
                "largest_gap_bars": 0,
                "expected_bars": 5,
                "actual_bars": 5,
                "unique_bars": 5,
                "duplicate_bars": 0,
            },
        ),
        (
            pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"]),
            [1, 2, 4, 5],
            {
                "missing_bars": 1,
                "largest_gap_bars": 1,
                "expected_bars": 5,
                "actual_bars": 4,
            },
        ),
        (
            pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"]),
            [1, 2, 2, 3],
            {
                "missing_bars": 0,
                "actual_bars": 4,
                "unique_bars": 3,
                "duplicate_bars": 1,
            },
        ),
    ],
    ids=["complete_series", "missing_internal_bars", "deduplicated_index"],
)
def test_compute_continuity_score_scenarios(idx, values, expected):
    df = pd.DataFrame({"Close": values}, index=idx)
    continuity = BacktestRunner.compute_continuity_score(df, "1d")

    for key, value in expected.items():
        assert continuity[key] == value

    if expected["missing_bars"] > 0 or expected.get("duplicate_bars", 0) > 0:
        assert continuity["score"] < 1.0


@pytest.mark.parametrize(
    ("df", "timeframe", "error_match"),
    [
        (pd.DataFrame(columns=["Close"]), "1d", "insufficient_bars_for_continuity"),
        (
            pd.DataFrame({"Close": [1]}, index=pd.to_datetime(["2024-01-01"])),
            "1d",
            "insufficient_bars_for_continuity",
        ),
        (
            pd.DataFrame({"Close": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3, freq="D")),
            "1quarter",
            "unsupported_timeframe_for_continuity",
        ),
    ],
    ids=["empty_frame", "single_bar", "unsupported_timeframe"],
)
def test_compute_continuity_score_invalid_inputs(df, timeframe, error_match):
    with pytest.raises(ValueError, match=error_match):
        BacktestRunner.compute_continuity_score(df, timeframe)


def test_compute_continuity_score_weekend_gap_not_missing_for_weekday_calendar():
    idx = pd.to_datetime(["2024-01-05", "2024-01-08"])  # Friday -> Monday
    df = pd.DataFrame({"Close": [100.0, 101.0]}, index=idx)
    continuity = BacktestRunner.compute_continuity_score(df, "1d", calendar_kind="weekday")
    assert continuity["missing_bars"] == 0
    assert continuity["score"] == pytest.approx(1.0)


def test_compute_continuity_score_weekend_gap_missing_for_24_7_calendar():
    idx = pd.to_datetime(["2024-01-05", "2024-01-08"])  # Friday -> Monday
    df = pd.DataFrame({"Close": [100.0, 101.0]}, index=idx)
    continuity = BacktestRunner.compute_continuity_score(df, "1d", calendar_kind="crypto_24_7")
    assert continuity["missing_bars"] == 2
    assert continuity["score"] < 1.0


@pytest.mark.parametrize(
    ("timeframe", "idx"),
    [
        ("1wk", pd.to_datetime(["2024-01-06", "2024-01-20", "2024-01-27"])),
        ("1mo", pd.to_datetime(["2024-01-02", "2024-02-01", "2024-04-01"])),
    ],
    ids=["weekly_weekend_anchors", "monthly_30d_anchors"],
)
def test_compute_continuity_score_weekday_calendar_non_daily_uses_fixed_delta(timeframe, idx):
    df = pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=idx)
    continuity = BacktestRunner.compute_continuity_score(df, timeframe, calendar_kind="weekday")
    assert continuity["expected_bars"] == 4
    assert continuity["missing_bars"] == 1
    assert continuity["largest_gap_bars"] == 1
    assert continuity["score"] < 1.0


def test_compute_continuity_score_exchange_calendar_ignores_market_holiday():
    pytest.importorskip("exchange_calendars")
    idx = pd.to_datetime(["2023-12-22", "2023-12-26"])  # Dec 25th is NYSE holiday
    df = pd.DataFrame({"Close": [100.0, 101.0]}, index=idx)
    weekday_continuity = BacktestRunner.compute_continuity_score(df, "1d", calendar_kind="weekday")
    exchange_continuity = BacktestRunner.compute_continuity_score(
        df, "1d", calendar_kind="exchange", exchange_calendar="XNYS"
    )
    assert weekday_continuity["missing_bars"] == 1
    assert exchange_continuity["missing_bars"] == 0
    assert weekday_continuity["score"] < 1.0
    assert exchange_continuity["score"] == pytest.approx(1.0)


def test_sample_series_preserves_last_point():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    series = pd.Series([1.0, 2.0, 3.0], index=idx)
    sampled = BacktestRunner._sample_series(series, max_points=2)
    assert sampled[-1]["ts"].startswith("2024-01-03")
    assert sampled[-1]["value"] == pytest.approx(3.0)


def test_fractional_enabled_rules():
    cfg = CollectionConfig(name="c", source="binance", symbols=["ETH/USDT"])
    assert BacktestRunner._fractional_enabled(cfg, "ETH/USDT")
    cfg2 = CollectionConfig(name="c2", source="yfinance", symbols=["AAPL"])
    assert BacktestRunner._fractional_enabled(cfg2, "AAPLUSDT")
    assert not BacktestRunner._fractional_enabled(cfg2, "AAPL")


@pytest.mark.parametrize(
    ("source", "cls_name", "exchange"),
    [
        ("yfinance", "YFinanceSource", None),
        ("polygon", "PolygonSource", None),
        ("tiingo", "TiingoSource", None),
        ("alpaca", "AlpacaSource", None),
        ("finnhub", "FinnhubSource", None),
        ("twelvedata", "TwelveDataSource", None),
        ("alphavantage", "AlphaVantageSource", None),
        ("binance", "CCXTSource", None),
        ("ccxt", "CCXTSource", "bybit"),
    ],
)
def test_make_source_variants(tmp_path, monkeypatch, source, cls_name, exchange):
    created_args: dict[str, tuple] = {}

    class _DummySource:
        def __init__(self, *args):
            created_args["args"] = args

    monkeypatch.setattr(f"src.backtest.runner.{cls_name}", _DummySource)
    monkeypatch.setattr("src.backtest.runner.discover_external_strategies", lambda root: {})

    cfg = Config(
        collections=[
            CollectionConfig(
                name="c",
                source=source,
                symbols=["AAPL"],
                exchange=exchange,
            )
        ],
        timeframes=["1d"],
        metric="sharpe",
        strategies=[],
        cache_dir=str(tmp_path / "cache"),
        engine="pybroker",
        param_search="grid",
        param_trials=1,
        max_workers=1,
        asset_workers=1,
        param_workers=1,
        max_fetch_concurrency=1,
        fees=0.0,
        slippage=0.0,
        risk_free_rate=0.0,
        notifications=None,
    )
    runner = BacktestRunner(cfg, strategies_root=tmp_path, run_id="sources")
    result = runner._make_source(cfg.collections[0])
    assert isinstance(result, _DummySource)
    if cls_name == "CCXTSource" and exchange is None:
        # source binance default exchange is derived from source string
        assert created_args["args"][0] == "binance"
        assert created_args["args"][1] == (tmp_path / "cache")
    elif cls_name == "CCXTSource":
        assert created_args["args"] == (exchange, tmp_path / "cache")
    else:
        assert created_args["args"] == ((tmp_path / "cache"),)


def test_make_source_ccxt_without_exchange(tmp_path, monkeypatch):
    monkeypatch.setattr("src.backtest.runner.discover_external_strategies", lambda root: {})
    cfg = Config(
        collections=[CollectionConfig(name="c", source="ccxt", symbols=["BTC/USDT"])],
        timeframes=["1d"],
        metric="sharpe",
        strategies=[],
        cache_dir=str(tmp_path / "cache"),
        engine="pybroker",
        param_search="grid",
        param_trials=1,
        max_workers=1,
        asset_workers=1,
        param_workers=1,
        max_fetch_concurrency=1,
        fees=0.0,
        slippage=0.0,
        risk_free_rate=0.0,
        notifications=None,
    )
    runner = BacktestRunner(cfg, strategies_root=tmp_path, run_id="sources")
    with pytest.raises(ValueError):
        runner._make_source(cfg.collections[0])


def test_prepare_pybroker_frame_orders_columns(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    _, _, _, _, data_col_enum = runner._ensure_pybroker()
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {"Close": [1, 2, 3], "High": [1, 2, 3], "Low": [1, 2, 3], "Open": [1, 2, 3]},
        index=dates[::-1],
    )
    prepared, prepared_dates = runner._prepare_pybroker_frame(df, "AAPL", data_col_enum)
    assert list(prepared.columns) == [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert prepared["symbol"].tolist() == ["AAPL"] * len(prepared)
    assert prepared_dates.is_monotonic_increasing


def test_compute_cagr_positive_growth(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    equity = pd.Series([1.0, 1.1, 1.2, 1.25, 1.4], index=dates)
    cagr = runner._compute_cagr(equity, dates, periods_per_year=252)
    assert cagr > 0


def test_evaluate_metric_handles_sharpe_sortino_profit(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    returns = pd.Series(
        [0.01, -0.005, 0.02], index=pd.date_range("2024-01-01", periods=3, freq="D")
    )
    equity = (1 + returns).cumprod()
    sharpe = runner._evaluate_metric("sharpe", returns, equity, 252)
    sortino = runner._evaluate_metric("sortino", returns, equity, 252)
    profit = runner._evaluate_metric("profit", returns, equity, 252)
    assert pytest.approx(profit, rel=1e-6) == equity.iloc[-1] / equity.iloc[0] - 1
    assert isfinite(sharpe)
    assert isfinite(sortino)
    with pytest.raises(ValueError):
        runner._evaluate_metric("unknown", returns, equity, 252)


def test_fees_slippage_for_defaults(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    col = CollectionConfig(name="c", source="binance", symbols=["ETH/USDT"])
    fees, slip = runner._fees_slippage_for(col)
    assert fees == pytest.approx(0.0006)
    assert slip == pytest.approx(0.0005)
    custom = CollectionConfig(
        name="c2",
        source="custom",
        symbols=["AAPL"],
        fees=0.001,
        slippage=0.002,
    )
    fees2, slip2 = runner._fees_slippage_for(custom)
    assert fees2 == pytest.approx(0.001)
    assert slip2 == pytest.approx(0.002)


def test_run_all_produces_best_result(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    results = runner.run_all()
    assert isinstance(results, list)
    assert len(results) == 1
    best = results[0]
    assert isinstance(best, BestResult)
    assert best.metric_value > float("-inf")
    # results cache set should have been called with run_id
    assert runner.results_cache.saved
    assert runner.results_cache.saved[0]["run_id"] == "test-run"
    assert "data_reliability" in best.stats
    assert "continuity" in best.stats["data_reliability"]
    assert best.stats["data_reliability"]["continuity"]["score"] == pytest.approx(1.0)
    assert runner.metrics["result_cache_misses"] >= 1
    assert runner.metrics["fresh_metric_evals"] >= 1


def test_run_all_accepts_weekly_timeframe_alias(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.timeframes = ["1wk"]

    results = runner.run_all()
    assert len(results) == 1


def test_evaluation_cache_persists_raw_stats_only(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)

    results = runner.run_all()
    assert len(results) == 1
    assert runner.evaluation_cache.saved
    cached_stats = runner.evaluation_cache.saved[0]["stats"]
    assert "data_reliability" not in cached_stats
    assert "optimization" not in cached_stats


def test_run_all_skips_strategy_when_plan_gate_fails(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)

    def _skip_plan(self, state, validated_data, plan):
        return GateDecision(False, "skip_job", ["plan_gate_blocked"], "strategy_optimization")

    def _fail_strategy_run(*args, **kwargs):  # pragma: no cover - defensive assertion
        raise AssertionError("strategy should not execute when plan gate fails")

    monkeypatch.setattr(BacktestRunner, "_strategy_validate_plan", _skip_plan)
    monkeypatch.setattr(BacktestRunner, "_strategy_run", _fail_strategy_run)

    results = runner.run_all()
    assert results == []
    assert runner.metrics["symbols_tested"] == 1
    assert runner.failures
    assert runner.failures[0]["stage"] == "strategy_optimization"
    assert runner.failures[0]["error"] == "plan_gate_blocked"


def test_run_all_strategy_skip_job_stops_remaining_strategies_for_job(tmp_path, monkeypatch):
    class _AltStrategy(BaseStrategy):
        name = "alt"

        def param_grid(self) -> dict[str, list[int]]:
            return {}

        def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
            entries = pd.Series([True] + [False] * (len(df.index) - 1), index=df.index)
            exits = pd.Series([False] * (len(df.index) - 1) + [True], index=df.index)
            return entries, exits

    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.strategies = []
    monkeypatch.setattr(
        "src.backtest.runner.discover_external_strategies",
        lambda root: {"dummy": _DummyStrategy, "alt": _AltStrategy},
    )
    runner.external_index = {"dummy": _DummyStrategy, "alt": _AltStrategy}

    def _plan_gate(self, state, validated_data, plan):
        if plan.strategy.name == "dummy":
            return GateDecision(False, "skip_job", ["plan_gate_blocked"], "strategy_optimization")
        return GateDecision(True, "continue", [], "strategy_optimization")

    original_strategy_run = BacktestRunner._strategy_run
    strategy_runs = {"count": 0}

    def _count_strategy_run(self, plan, state, validated_data, prepared):
        strategy_runs["count"] += 1
        return original_strategy_run(self, plan, state, validated_data, prepared)

    monkeypatch.setattr(BacktestRunner, "_strategy_validate_plan", _plan_gate)
    monkeypatch.setattr(BacktestRunner, "_strategy_run", _count_strategy_run)

    results = runner.run_all()
    assert results == []
    assert strategy_runs["count"] == 0
    assert len(runner.failures) == 1
    assert runner.failures[0]["stage"] == "strategy_optimization"
    assert runner.failures[0]["strategy"] == "dummy"


def test_run_all_strategy_skip_collection_blocks_current_and_remaining_jobs(tmp_path, monkeypatch):
    class _AltStrategy(BaseStrategy):
        name = "alt"

        def param_grid(self) -> dict[str, list[int]]:
            return {}

        def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
            entries = pd.Series([True] + [False] * (len(df.index) - 1), index=df.index)
            exits = pd.Series([False] * (len(df.index) - 1) + [True], index=df.index)
            return entries, exits

    collections = [
        CollectionConfig(
            name="demo",
            source="custom",
            symbols=["AAPL", "MSFT"],
            fees=0.0004,
            slippage=0.0003,
        )
    ]
    runner = _make_runner(tmp_path, monkeypatch, collections=collections)
    runner.cfg.strategies = []
    monkeypatch.setattr(
        "src.backtest.runner.discover_external_strategies",
        lambda root: {"dummy": _DummyStrategy, "alt": _AltStrategy},
    )
    runner.external_index = {"dummy": _DummyStrategy, "alt": _AltStrategy}

    fetch_calls = {"count": 0}

    class _Source:
        def fetch(self, symbol, timeframe, only_cached=False):
            fetch_calls["count"] += 1
            return _make_ohlcv(20)

    def _plan_gate(self, state, validated_data, plan):
        if plan.strategy.name == "dummy":
            return GateDecision(False, "skip_collection", ["plan_gate_blocked"], "strategy_optimization")
        return GateDecision(True, "continue", [], "strategy_optimization")

    original_strategy_run = BacktestRunner._strategy_run
    strategy_runs = {"count": 0}

    def _count_strategy_run(self, plan, state, validated_data, prepared):
        strategy_runs["count"] += 1
        return original_strategy_run(self, plan, state, validated_data, prepared)

    monkeypatch.setattr(BacktestRunner, "_make_source", lambda self, col: _Source())
    monkeypatch.setattr(BacktestRunner, "_strategy_validate_plan", _plan_gate)
    monkeypatch.setattr(BacktestRunner, "_strategy_run", _count_strategy_run)

    results = runner.run_all()
    assert results == []
    assert fetch_calls["count"] == 1
    assert strategy_runs["count"] == 0
    assert len(runner.failures) == 1
    assert runner.failures[0]["stage"] == "strategy_optimization"
    assert runner.failures[0]["strategy"] == "dummy"


def test_run_all_uses_cached_results(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)

    class _CachedResults(_StubResultsCache):
        def __init__(self):
            super().__init__()
            self.set_calls = 0

        def set(self, **kwargs):
            self.set_calls += 1
            super().set(**kwargs)

    class _CachedEval(_StubEvaluationCache):
        def get(self, **kwargs):
            self.retrieved.append(kwargs)
            return {"metric_value": 2.0, "stats": {"cached": True}}

    runner.results_cache = _CachedResults()
    runner.evaluation_cache = _CachedEval()
    runner.mode_config_hash = runner.evaluation_cache.hash_mode_config(runner.mode_config)

    def _fail_sim(*args, **kwargs):  # should never be called
        raise AssertionError("Simulation should not execute when cache hits")

    monkeypatch.setattr(BacktestRunner, "_run_pybroker_simulation", _fail_sim)
    results = runner.run_all()
    assert len(results) == 1
    assert runner.metrics["result_cache_hits"] >= 1
    assert runner.metrics["result_cache_misses"] == 0
    assert runner.metrics["fresh_metric_evals"] == 0
    assert runner.results_cache.set_calls == runner.metrics["result_cache_hits"]


def test_run_all_handles_failed_and_nan_metrics(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.validation = ValidationConfig(
        optimization=OptimizationPolicyConfig(
            on_fail="baseline_only",
            min_bars=1,
            dof_multiplier=1,
        )
    )
    call_state = {"count": 0}

    def _sim_with_none(self, *args, **kwargs):
        call_state["count"] += 1
        if call_state["count"] == 1:
            return None
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        returns = pd.Series([0.0, 0.0, 0.0], index=dates)
        equity = (1 + returns).cumprod()
        stats = {"equity_curve": [], "drawdown_curve": [], "trades_log": []}
        stats.update(
            {
                "sharpe": 0.0,
                "sortino": 0.0,
                "omega": 0.0,
                "tail_ratio": 0.0,
                "profit": 0.0,
                "pain_index": 0.0,
                "trades": 0,
                "max_drawdown": 0.0,
                "cagr": 0.0,
                "calmar": 0.0,
            }
        )
        return returns, equity, stats

    monkeypatch.setattr(BacktestRunner, "_run_pybroker_simulation", _sim_with_none)
    monkeypatch.setattr(
        BacktestRunner, "_evaluate_metric", lambda self, *args, **kwargs: float("nan")
    )
    runner.results_cache = _StubResultsCache()
    results = runner.run_all()
    assert results == []
    assert runner.metrics["result_cache_misses"] >= 1
    assert runner.metrics.get("fresh_metric_evals", 0) >= 1
    assert len(runner.evaluation_cache.saved) == 1
    assert len(runner.results_cache.saved) == 1
    assert runner.evaluation_cache.saved[0]["metric_value"] == float("-inf")
    assert runner.results_cache.saved[0]["metric_value"] == float("-inf")


def test_run_pybroker_simulation_generates_metrics(tmp_path, monkeypatch):
    runner = _make_runner(
        tmp_path,
        monkeypatch,
        patch_pybroker=False,
        patch_sim=False,
        patch_source=False,
    )

    class _StubStrategy:
        def __init__(self, data, start_date, end_date, config):
            self.data = data
            self.start_date = start_date
            self.end_date = end_date
            self.config = config
            self.execs = []

        def add_execution(self, fn, symbol):
            self.execs.append((fn, symbol))

        def backtest(self, calc_bootstrap=False):
            dates = pd.to_datetime(self.data["date"])
            portfolio = pd.DataFrame(
                {
                    "date": dates,
                    "equity": [10000.0, 10200.0, 10100.0, 10400.0],
                }
            )
            trades = pd.DataFrame(
                {
                    "entry_time": pd.to_datetime([dates[0]]),
                    "exit_time": pd.to_datetime([dates[1]]),
                    "profit": [Decimal("200.0")],
                }
            )

            class _Metrics:
                trade_count = len(trades)

            return SimpleNamespace(portfolio=portfolio, trades=trades, metrics=_Metrics())

    class _StubConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _StubFeeMode:
        ORDER_PERCENT = "order_percent"

    class _StubPriceType:
        CLOSE = "close"

    class _DataEnum:
        SYMBOL = SimpleNamespace(value="symbol")
        DATE = SimpleNamespace(value="date")
        OPEN = SimpleNamespace(value="open")
        HIGH = SimpleNamespace(value="high")
        LOW = SimpleNamespace(value="low")
        CLOSE = SimpleNamespace(value="close")
        VOLUME = SimpleNamespace(value="volume")

    monkeypatch.setattr(
        BacktestRunner,
        "_ensure_pybroker",
        lambda self: (_StubStrategy, _StubConfig, _StubFeeMode, _StubPriceType, _DataEnum),
    )

    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    data = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 4,
            "date": dates,
            "open": [10.0, 10.5, 10.8, 11.0],
            "high": [10.6, 11.0, 11.2, 11.5],
            "low": [9.8, 10.1, 10.4, 10.6],
            "close": [10.5, 10.9, 10.7, 11.2],
            "volume": [100, 110, 105, 120],
        }
    )

    entries = pd.Series([True, False, False, False], index=dates)
    exits = pd.Series([False, True, False, False], index=dates)

    result = runner._run_pybroker_simulation(
        data,
        dates,
        "AAPL",
        entries,
        exits,
        fee_percent=0.0005,
        slippage_percent=0.0004,
        timeframe="1d",
        fractional=True,
        periods_per_year=252,
    )

    assert result is not None
    returns, equity_curve, stats = result
    assert len(returns) == len(equity_curve)
    assert "sharpe" in stats and "trades" in stats
    assert stats["trades"] == 1


def test_run_all_records_fetch_failures(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)

    class _FailingSource:
        def fetch(self, symbol, timeframe, only_cached=False):
            raise RuntimeError("boom")

    monkeypatch.setattr(BacktestRunner, "_make_source", lambda self, col: _FailingSource())

    runner.results_cache = _StubResultsCache()
    results = runner.run_all()
    assert results == []
    assert runner.failures
    failure = runner.failures[0]
    assert failure["error"] == "boom"


def test_run_all_skips_empty_frames(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)

    class _EmptySource:
        def fetch(self, symbol, timeframe, only_cached=False):
            dates = pd.date_range("2024-01-01", periods=0, freq="D")
            return pd.DataFrame(index=dates)

    monkeypatch.setattr(BacktestRunner, "_make_source", lambda self, col: _EmptySource())
    runner.results_cache = _StubResultsCache()
    results = runner.run_all()
    assert results == []
    assert not runner.results_cache.saved
    assert runner.failures
    failure = runner.failures[0]
    assert failure["stage"] == "data_validation"
    assert failure["error"] == "empty_dataframe"


def test_run_all_data_validation_handles_non_value_error_continuity_failure(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    _patch_source_with_bars(monkeypatch, bars=5)
    eval_calls = _patch_pybroker_simulation(monkeypatch)

    def _raise_type_error(
        _cls,
        _df,
        _timeframe,
        **_kwargs,
    ):
        raise TypeError("bad_index_type")

    monkeypatch.setattr(BacktestRunner, "compute_continuity_score", classmethod(_raise_type_error))

    results = runner.run_all()
    assert results == []
    assert eval_calls["count"] == 0
    assert runner.failures
    failure = runner.failures[0]
    assert failure["stage"] == "data_validation"
    assert "bad_index_type" in failure["error"]


def _make_ohlcv(periods: int) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    return pd.DataFrame(
        {
            "Open": [10] * len(dates),
            "High": [11] * len(dates),
            "Low": [9] * len(dates),
            "Close": [10.5] * len(dates),
            "Volume": [100] * len(dates),
        },
        index=dates,
    )


def _patch_source_with_bars(monkeypatch, bars: int) -> None:
    class _Source:
        def fetch(self, symbol, timeframe, only_cached=False):
            return _make_ohlcv(bars)

    monkeypatch.setattr(BacktestRunner, "_make_source", lambda self, col: _Source())


def _patch_pybroker_simulation(monkeypatch) -> dict[str, int]:
    eval_calls = {"count": 0}

    def _fake_sim(self, *args, **kwargs):
        eval_calls["count"] += 1
        returns = pd.Series(
            [0.01, -0.005, 0.02],
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
        equity = (1 + returns.fillna(0.0)).cumprod()
        stats = {
            "sharpe": 1.0,
            "sortino": 0.8,
            "omega": 1.2,
            "tail_ratio": 1.1,
            "profit": 0.1,
            "pain_index": 0.02,
            "trades": 2,
            "max_drawdown": -0.05,
            "cagr": 0.12,
            "calmar": -2.4,
            "equity_curve": [],
            "drawdown_curve": [],
            "trades_log": [],
        }
        return returns, equity, stats

    monkeypatch.setattr(BacktestRunner, "_run_pybroker_simulation", _fake_sim)
    return eval_calls


def test_run_all_uses_default_optimization_guard_when_policy_missing(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.validation = None
    runner.cfg.param_search = "grid"
    _patch_source_with_bars(monkeypatch, bars=5)
    eval_calls = _patch_pybroker_simulation(monkeypatch)

    results = runner.run_all()
    assert results
    assert eval_calls["count"] == 1
    optimization = results[0].stats.get("optimization")
    assert optimization is not None
    assert optimization["reason"] == "insufficient_bars_for_optimization"
    assert optimization["min_bars_required"] == 2000
    assert optimization["bars_available"] == 5


@pytest.mark.parametrize(
    ("dof_multiplier", "min_bars", "bars", "expected_eval_calls", "expect_skip"),
    [
        (100, 2000, 50, 1, True),  # skip via min-bars floor
        (60, 1, 50, 1, True),  # skip via DoF threshold
        (1, 1, 50, 2, False),  # no guard, full grid evals
        (50, 1, 50, 2, False),  # boundary: len(df) == required => no skip
    ],
    ids=["min_bars_floor_skip", "dof_skip", "no_guard", "dof_boundary_no_skip"],
)
def test_run_all_min_bars_and_dof_guard_behavior(
    tmp_path,
    monkeypatch,
    dof_multiplier,
    min_bars,
    bars,
    expected_eval_calls,
    expect_skip,
):
    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.validation = ValidationConfig(
        optimization=OptimizationPolicyConfig(
            on_fail="baseline_only",
            min_bars=min_bars,
            dof_multiplier=dof_multiplier,
        )
    )
    runner.cfg.param_search = "grid"

    _patch_source_with_bars(monkeypatch, bars)
    eval_calls = _patch_pybroker_simulation(monkeypatch)

    results = runner.run_all()
    assert results
    assert eval_calls["count"] == expected_eval_calls

    optimization = results[0].stats.get("optimization")
    if expect_skip:
        assert optimization is not None
        assert optimization["skipped"] is True
        assert optimization["reason"] == "insufficient_bars_for_optimization"
        # search_space has one dimension (`window`) in _make_runner, so n_params=1.
        expected_min_bars = max(min_bars, dof_multiplier * 1)
        assert optimization["min_bars_required"] == expected_min_bars
        assert optimization["bars_available"] == bars
    else:
        assert optimization is None


def test_run_all_optimization_policy_skip_job_on_infeasible_search(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.param_search = "grid"
    runner.cfg.validation = ValidationConfig(
        optimization=OptimizationPolicyConfig(
            on_fail="skip_job",
            min_bars=60,
            dof_multiplier=1,
        )
    )
    _patch_source_with_bars(monkeypatch, bars=5)
    eval_calls = _patch_pybroker_simulation(monkeypatch)

    results = runner.run_all()
    assert results == []
    assert eval_calls["count"] == 0
    assert len(runner.failures) == 1
    assert runner.failures[0]["stage"] == "strategy_optimization"
    assert "insufficient_bars_for_optimization" in runner.failures[0]["error"]


def test_run_all_reliability_min_data_points_skips_optimization(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.validation = ValidationConfig(
        data_quality=ValidationDataQualityConfig(min_data_points=10, on_fail="skip_optimization"),
    )
    _patch_source_with_bars(monkeypatch, bars=5)
    eval_calls = _patch_pybroker_simulation(monkeypatch)

    results = runner.run_all()
    assert results
    assert eval_calls["count"] == 1
    optimization = results[0].stats.get("optimization")
    assert optimization is not None
    assert optimization["reason"] == "reliability_threshold_skip_optimization"
    reasons = optimization.get("reliability_reasons", [])
    assert any("min_data_points_not_met" in r for r in reasons)


def test_run_all_reliability_skip_evaluation_on_continuity_threshold(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.validation = ValidationConfig(
        data_quality=ValidationDataQualityConfig(min_continuity_score=0.95, on_fail="skip_job"),
    )

    class _GappySource:
        def fetch(self, symbol, timeframe, only_cached=False):
            idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"])
            return pd.DataFrame(
                {
                    "Open": [10, 11, 13, 14],
                    "High": [11, 12, 14, 15],
                    "Low": [9, 10, 12, 13],
                    "Close": [10.5, 11.5, 13.5, 14.5],
                    "Volume": [100, 110, 130, 140],
                },
                index=idx,
            )

    monkeypatch.setattr(BacktestRunner, "_make_source", lambda self, col: _GappySource())
    eval_calls = _patch_pybroker_simulation(monkeypatch)

    results = runner.run_all()
    assert results == []
    assert eval_calls["count"] == 0
    assert runner.failures
    failure = runner.failures[0]
    assert failure["stage"] == "data_validation"
    assert "min_continuity_score_not_met" in failure["error"]


def test_run_all_reliability_skip_evaluation_on_missing_bar_pct_threshold(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.validation = ValidationConfig(
        data_quality=ValidationDataQualityConfig(max_missing_bar_pct=10.0, on_fail="skip_job"),
    )

    class _GappySource:
        def fetch(self, symbol, timeframe, only_cached=False):
            idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"])
            return pd.DataFrame(
                {
                    "Open": [10, 11, 13, 14],
                    "High": [11, 12, 14, 15],
                    "Low": [9, 10, 12, 13],
                    "Close": [10.5, 11.5, 13.5, 14.5],
                    "Volume": [100, 110, 130, 140],
                },
                index=idx,
            )

    monkeypatch.setattr(BacktestRunner, "_make_source", lambda self, col: _GappySource())
    eval_calls = _patch_pybroker_simulation(monkeypatch)

    results = runner.run_all()
    assert results == []
    assert eval_calls["count"] == 0
    assert runner.failures
    failure = runner.failures[0]
    assert failure["stage"] == "data_validation"
    assert "max_missing_bar_pct_exceeded" in failure["error"]


def test_run_all_reliability_skip_evaluation_on_max_kurtosis(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.validation = ValidationConfig(
        data_quality=ValidationDataQualityConfig(max_kurtosis=1.0, on_fail="skip_job"),
    )

    class _LeptokurticSource:
        def fetch(self, symbol, timeframe, only_cached=False):
            idx = pd.date_range("2024-01-01", periods=13, freq="D")
            closes = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 400.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
            return pd.DataFrame(
                {
                    "Open": closes,
                    "High": closes,
                    "Low": closes,
                    "Close": closes,
                    "Volume": [100] * len(closes),
                },
                index=idx,
            )

    monkeypatch.setattr(BacktestRunner, "_make_source", lambda self, col: _LeptokurticSource())
    eval_calls = _patch_pybroker_simulation(monkeypatch)

    results = runner.run_all()
    assert results == []
    assert eval_calls["count"] == 0
    assert runner.failures
    failure = runner.failures[0]
    assert failure["stage"] == "data_validation"
    assert "max_kurtosis_exceeded" in failure["error"]


def test_run_all_fetches_once_per_symbol_timeframe_with_multiple_strategies(tmp_path, monkeypatch):
    class _AltStrategy(BaseStrategy):
        name = "alt"

        def param_grid(self) -> dict[str, list[int]]:
            return {}

        def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
            entries = pd.Series([True] + [False] * (len(df.index) - 1), index=df.index)
            exits = pd.Series([False] * (len(df.index) - 1) + [True], index=df.index)
            return entries, exits

    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.strategies = []
    monkeypatch.setattr(
        "src.backtest.runner.discover_external_strategies",
        lambda root: {"dummy": _DummyStrategy, "alt": _AltStrategy},
    )
    runner.external_index = {"dummy": _DummyStrategy, "alt": _AltStrategy}

    fetch_calls = {"count": 0}

    class _Source:
        def fetch(self, symbol, timeframe, only_cached=False):
            fetch_calls["count"] += 1
            return _make_ohlcv(5)

    monkeypatch.setattr(BacktestRunner, "_make_source", lambda self, col: _Source())
    _patch_pybroker_simulation(monkeypatch)

    results = runner.run_all()
    assert fetch_calls["count"] == 1
    assert len(results) == 2
    assert runner.metrics["symbols_tested"] == 1


def test_run_all_skip_evaluation_adds_single_failure_for_multiple_strategies(tmp_path, monkeypatch):
    class _AltStrategy(BaseStrategy):
        name = "alt"

        def param_grid(self) -> dict[str, list[int]]:
            return {}

        def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
            entries = pd.Series([True] + [False] * (len(df.index) - 1), index=df.index)
            exits = pd.Series([False] * (len(df.index) - 1) + [True], index=df.index)
            return entries, exits

    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.strategies = []
    runner.cfg.validation = ValidationConfig(
        data_quality=ValidationDataQualityConfig(min_data_points=10, on_fail="skip_job"),
    )
    monkeypatch.setattr(
        "src.backtest.runner.discover_external_strategies",
        lambda root: {"dummy": _DummyStrategy, "alt": _AltStrategy},
    )
    runner.external_index = {"dummy": _DummyStrategy, "alt": _AltStrategy}
    _patch_source_with_bars(monkeypatch, bars=5)
    eval_calls = _patch_pybroker_simulation(monkeypatch)

    results = runner.run_all()
    assert results == []
    assert eval_calls["count"] == 0
    assert len(runner.failures) == 1
    assert runner.failures[0]["stage"] == "data_validation"
    assert "min_data_points_not_met" in runner.failures[0]["error"]


def test_run_all_skip_optimization_still_evaluates_each_strategy(tmp_path, monkeypatch):
    class _AltStrategy(BaseStrategy):
        name = "alt"

        def param_grid(self) -> dict[str, list[int]]:
            return {"window": [5, 6]}

        def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
            entries = pd.Series([True] + [False] * (len(df.index) - 1), index=df.index)
            exits = pd.Series([False] * (len(df.index) - 1) + [True], index=df.index)
            return entries, exits

    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.strategies = []
    runner.cfg.validation = ValidationConfig(
        data_quality=ValidationDataQualityConfig(min_data_points=10, on_fail="skip_optimization"),
        optimization=OptimizationPolicyConfig(
            on_fail="baseline_only",
            min_bars=100,
            dof_multiplier=100,
        ),
    )
    monkeypatch.setattr(
        "src.backtest.runner.discover_external_strategies",
        lambda root: {"dummy": _DummyStrategy, "alt": _AltStrategy},
    )
    runner.external_index = {"dummy": _DummyStrategy, "alt": _AltStrategy}
    _patch_source_with_bars(monkeypatch, bars=5)
    eval_calls = _patch_pybroker_simulation(monkeypatch)

    results = runner.run_all()
    assert len(results) == 2
    assert eval_calls["count"] == 2
    for result in results:
        optimization = result.stats.get("optimization")
        assert optimization is not None
        assert optimization["reason"] == "reliability_threshold_skip_optimization"
        reasons = optimization.get("reliability_reasons", [])
        assert any("min_data_points_not_met" in reason for reason in reasons)
        assert all("insufficient_bars_for_optimization" not in reason for reason in optimization["reasons"])


def test_run_all_skip_optimization_does_not_skip_no_param_strategy(tmp_path, monkeypatch):
    class _NoParamStrategy(BaseStrategy):
        name = "no_param"

        def param_grid(self) -> dict[str, list[int]]:
            return {}

        def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
            entries = pd.Series([True] + [False] * (len(df.index) - 1), index=df.index)
            exits = pd.Series([False] * (len(df.index) - 1) + [True], index=df.index)
            return entries, exits

    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.strategies = []
    runner.cfg.validation = ValidationConfig(
        data_quality=ValidationDataQualityConfig(min_data_points=10, on_fail="skip_optimization"),
        optimization=OptimizationPolicyConfig(
            on_fail="skip_job",
            min_bars=60,
            dof_multiplier=1,
        ),
    )
    monkeypatch.setattr(
        "src.backtest.runner.discover_external_strategies",
        lambda root: {"no_param": _NoParamStrategy},
    )
    runner.external_index = {"no_param": _NoParamStrategy}
    _patch_source_with_bars(monkeypatch, bars=5)
    eval_calls = _patch_pybroker_simulation(monkeypatch)

    results = runner.run_all()
    assert len(results) == 1
    assert eval_calls["count"] == 1
    assert not any(failure["stage"] == "strategy_optimization" for failure in runner.failures)


def test_strategy_plan_skip_optimization_does_not_leak_to_next_strategy(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.validation = ValidationConfig(
        optimization=OptimizationPolicyConfig(
            on_fail="baseline_only",
            min_bars=1,
            dof_multiplier=3,
        )
    )
    runner.cfg.strategies = []

    class _WideStrategy(BaseStrategy):
        name = "wide"

        def param_grid(self) -> dict[str, list[int]]:
            return {"a": [1, 2], "b": [1, 2], "c": [1, 2]}

        def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
            entries = pd.Series([True] + [False] * (len(df.index) - 1), index=df.index)
            exits = pd.Series([False] * (len(df.index) - 1) + [True], index=df.index)
            return entries, exits

    class _NarrowStrategy(BaseStrategy):
        name = "narrow"

        def param_grid(self) -> dict[str, list[int]]:
            return {"x": [1, 2]}

        def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
            entries = pd.Series([True] + [False] * (len(df.index) - 1), index=df.index)
            exits = pd.Series([False] * (len(df.index) - 1) + [True], index=df.index)
            return entries, exits

    sim_calls = {"count": 0}

    def _counting_sim(self, *args, **kwargs):
        sim_calls["count"] += 1
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        returns = pd.Series([0.01, 0.0, 0.0, 0.0, 0.0], index=dates)
        equity = (1 + returns).cumprod()
        stats = {
            "sharpe": 1.0,
            "sortino": 1.0,
            "omega": 1.0,
            "tail_ratio": 1.0,
            "profit": 0.01,
            "pain_index": 0.0,
            "trades": 1,
            "max_drawdown": 0.0,
            "cagr": 0.0,
            "calmar": 0.0,
            "equity_curve": [],
            "drawdown_curve": [],
            "trades_log": [],
        }
        return returns, equity, stats

    monkeypatch.setattr(BacktestRunner, "_run_pybroker_simulation", _counting_sim)
    monkeypatch.setattr(
        "src.backtest.runner.discover_external_strategies",
        lambda root: {"wide": _WideStrategy, "narrow": _NarrowStrategy},
    )
    runner.external_index = {"wide": _WideStrategy, "narrow": _NarrowStrategy}

    results = runner.run_all()
    assert len(results) == 2
    assert sim_calls["count"] == 3


def test_run_all_collection_reliability_override_takes_precedence(tmp_path, monkeypatch):
    collection = CollectionConfig(
        name="demo",
        source="custom",
        symbols=["AAPL"],
        fees=0.0004,
        slippage=0.0003,
        validation=ValidationConfig(
            data_quality=ValidationDataQualityConfig(min_data_points=10, on_fail="skip_optimization"),
        ),
    )
    runner = _make_runner(tmp_path, monkeypatch, collections=[collection])
    runner.cfg.validation = ValidationConfig(
        data_quality=ValidationDataQualityConfig(min_data_points=10, on_fail="skip_job"),
    )
    _patch_source_with_bars(monkeypatch, bars=5)
    eval_calls = _patch_pybroker_simulation(monkeypatch)

    results = runner.run_all()
    assert len(results) == 1
    assert eval_calls["count"] == 1
    optimization = results[0].stats.get("optimization")
    assert optimization is not None
    assert optimization["reason"] == "reliability_threshold_skip_optimization"
    reasons = optimization.get("reliability_reasons", [])
    assert any("min_data_points_not_met" in reason for reason in reasons)


def test_run_all_reliability_skip_collection_blocks_remaining_jobs_in_collection(
    tmp_path, monkeypatch
):
    collections = [
        CollectionConfig(
            name="bad_col",
            source="custom",
            symbols=["AAPL", "MSFT"],
            fees=0.0004,
            slippage=0.0003,
        ),
        CollectionConfig(
            name="good_col",
            source="custom",
            symbols=["NVDA"],
            fees=0.0004,
            slippage=0.0003,
        ),
    ]
    runner = _make_runner(tmp_path, monkeypatch, collections=collections)
    runner.cfg.validation = ValidationConfig(
        data_quality=ValidationDataQualityConfig(min_data_points=10, on_fail="skip_collection"),
    )

    fetch_calls = {"bad_col": 0, "good_col": 0}

    class _Source:
        def __init__(self, collection_name: str):
            self.collection_name = collection_name

        def fetch(self, symbol, timeframe, only_cached=False):
            fetch_calls[self.collection_name] += 1
            if self.collection_name == "bad_col":
                return _make_ohlcv(5)
            return _make_ohlcv(20)

    monkeypatch.setattr(BacktestRunner, "_make_source", lambda self, col: _Source(col.name))
    eval_calls = _patch_pybroker_simulation(monkeypatch)

    results = runner.run_all()
    assert len(results) == 1
    assert fetch_calls["bad_col"] == 1
    assert fetch_calls["good_col"] == 1
    assert eval_calls["count"] == 2
    assert len(runner.failures) == 1
    failure = runner.failures[0]
    assert failure["collection"] == "bad_col"
    assert failure["stage"] == "data_validation"
    assert "min_data_points_not_met" in failure["error"]


def test_run_all_collection_cache_isolation_for_same_name_collections(tmp_path, monkeypatch):
    collections = [
        CollectionConfig(
            name="crypto",
            source="good",
            symbols=["BTCUSD"],
            fees=0.0004,
            slippage=0.0003,
        ),
        CollectionConfig(
            name="crypto",
            source="bad",
            symbols=["ETHUSD"],
            fees=0.0004,
            slippage=0.0003,
        ),
    ]
    runner = _make_runner(tmp_path, monkeypatch, collections=collections, patch_source=False)

    make_source_calls = {"good": 0, "bad": 0}

    class _Source:
        def fetch(self, symbol, timeframe, only_cached=False):
            return _make_ohlcv(20)

    def _make_source(self, col):
        make_source_calls[col.source] += 1
        if col.source == "good":
            return _Source()
        raise ValueError("bad source config")

    monkeypatch.setattr(BacktestRunner, "_make_source", _make_source)
    eval_calls = _patch_pybroker_simulation(monkeypatch)

    results = runner.run_all()
    assert len(results) == 1
    assert eval_calls["count"] == 2
    # Reusing validation-built source avoids duplicate construction for the passing collection.
    assert make_source_calls == {"good": 1, "bad": 1}
    assert any(
        failure["stage"] == "collection_validation" and "bad source config" in failure["error"]
        for failure in runner.failures
    )


def test_runner_rejects_walk_forward_mode_until_implemented(tmp_path, monkeypatch):
    cfg = Config(
        collections=[CollectionConfig(name="demo", source="yfinance", symbols=["AAPL"])],
        timeframes=["1d"],
        metric="sharpe",
        strategies=[],
        cache_dir=str(tmp_path / "cache"),
        evaluation_mode="walk_forward",
    )
    monkeypatch.setattr("src.backtest.runner.discover_external_strategies", lambda root: {})
    with pytest.raises(NotImplementedError):
        BacktestRunner(cfg, strategies_root=tmp_path, run_id="wf")
