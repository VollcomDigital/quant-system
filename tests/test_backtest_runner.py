from __future__ import annotations

from decimal import Decimal
from math import isfinite
from types import SimpleNamespace

import pandas as pd
import pytest

from src.backtest.runner import BacktestRunner, BestResult
from src.config import CollectionConfig, Config, StrategyConfig
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
    return runner


def test_bars_per_year_various_units():
    assert BacktestRunner._bars_per_year("1d") == 252
    assert BacktestRunner._bars_per_year("2w") == 26
    assert BacktestRunner._bars_per_year("4h") == pytest.approx(int(round(24 * 365 / 4)))
    assert BacktestRunner._bars_per_year("15m") == pytest.approx(int(round(60 * 24 * 365 / 15)))
    assert BacktestRunner._bars_per_year("10s") == pytest.approx(
        int(round(60 * 60 * 24 * 365 / 10))
    )


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
    assert runner.metrics["result_cache_misses"] >= 1
    assert runner.metrics["param_evals"] >= 1


def test_run_all_uses_cached_results(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)

    class _CachedResults(_StubResultsCache):
        def __init__(self):
            super().__init__()
            self.set_calls = 0

        def get(self, **kwargs):
            self.retrieved.append(kwargs)
            return {"metric_value": 2.0, "stats": {"cached": True}}

        def set(self, **kwargs):
            self.set_calls += 1
            super().set(**kwargs)

    runner.results_cache = _CachedResults()

    def _fail_sim(*args, **kwargs):  # should never be called
        raise AssertionError("Simulation should not execute when cache hits")

    monkeypatch.setattr(BacktestRunner, "_run_pybroker_simulation", _fail_sim)
    results = runner.run_all()
    assert len(results) == 1
    assert runner.metrics["result_cache_hits"] >= 1
    assert runner.metrics["result_cache_misses"] == 0
    assert runner.metrics["param_evals"] == 0
    assert runner.results_cache.set_calls == runner.metrics["result_cache_hits"]


def test_run_all_handles_failed_and_nan_metrics(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
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
    assert runner.metrics.get("param_evals", 0) >= 1


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
