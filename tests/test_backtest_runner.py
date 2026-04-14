from __future__ import annotations

import json
from decimal import Decimal
from math import isfinite
from types import MethodType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.backtest.evaluation.contracts import (
    EvaluationModeConfig,
    EvaluationOutcome,
    EvaluationRequest,
)
from src.backtest.runner import BacktestRunner, BestResult, GateDecision, StrategyEvalOutcome
from src.backtest.runner import (
    ExecutionPreparedData,
    JobContext,
    JobState,
    TransactionCostRobustnessRunContext,
    ValidationContext,
    ValidatedData,
)
from src.config import (
    CollectionConfig,
    Config,
    OptimizationPolicyConfig,
    ResultConsistencyConfig,
    ResultConsistencyExecutionPriceVarianceConfig,
    ResultConsistencyTransactionCostBreakevenConfig,
    ResultConsistencyTransactionCostRobustnessConfig,
    ResultConsistencyOutlierDependencyConfig,
    StrategyConfig,
    ValidationCalendarConfig,
    ValidationConfig,
    ValidationContinuityConfig,
    ValidationDataQualityConfig,
    ValidationLookaheadShuffleTestConfig,
    ValidationOutlierDetectionConfig,
    ValidationStationarityConfig,
    ValidationStationarityRegimeShiftConfig,
    resolve_validation_overrides,
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


class _LeakyShuffleStrategy(BaseStrategy):
    name = "leaky_shuffle"

    def param_grid(self) -> dict[str, list[int]]:
        return {}

    def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)
        entries.iloc[0] = True
        close_col = df["Close"].astype(float)
        exit_idx = close_col.idxmax()
        if exit_idx == df.index[0] and len(df.index) > 1:
            exit_idx = df.index[-1]
        exits.loc[exit_idx] = True
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
            return returns, equity, stats, pd.DataFrame()

        monkeypatch.setattr(BacktestRunner, "_run_pybroker_simulation", _fake_sim)

    runner.results_cache = _StubResultsCache()
    runner.evaluation_cache = _StubEvaluationCache()
    runner.mode_config_hash = runner.evaluation_cache.hash_mode_config(runner.mode_config)
    original_run_all = BacktestRunner.run_all

    def _run_all_with_resolved_cfg(self, only_cached=False):
        # Tests often mutate runner.cfg after construction; resolve overrides
        # at invocation time to keep config ownership in src/config.py.
        resolve_validation_overrides(self.cfg)
        return original_run_all(self, only_cached=only_cached)

    runner.run_all = MethodType(_run_all_with_resolved_cfg, runner)
    return runner


def _result_consistency_config(**overrides) -> ResultConsistencyConfig:
    payload = {
        "min_metric": -1e9,
        "min_trades": 1,
        "outlier_dependency": None,
        "execution_price_variance": None,
        "lookahead_shuffle_test": None,
        "transaction_cost_robustness": None,
    }
    payload.update(overrides)
    return ResultConsistencyConfig(**payload)


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


def test_compute_outlier_mask_rejects_unsupported_method():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02], dtype=float)
    mask, issue = BacktestRunner._compute_outlier_mask(
        returns=returns, method="bad_method", threshold=3.0
    )
    assert mask is None
    assert issue == "unsupported_method:bad_method"


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


def test_compute_continuity_score_exchange_calendar_invalid_exchange_raises_value_error():
    pytest.importorskip("exchange_calendars")
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=idx)

    with pytest.raises(ValueError, match="Failed to use exchange calendar"):
        BacktestRunner.compute_continuity_score(
            df, "1d", calendar_kind="exchange", exchange_calendar="INVALID_EXCHANGE"
        )


def test_data_validation_calendar_timezone_changes_weekday_continuity(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.validation = ValidationConfig(
        data_quality=ValidationDataQualityConfig(
            continuity=ValidationContinuityConfig(
                calendar=ValidationCalendarConfig(kind="weekday", timezone="UTC-05:00")
            ),
            on_fail="skip_job",
        )
    )
    resolve_validation_overrides(runner.cfg)
    idx = pd.to_datetime(["2024-01-06 00:30:00+00:00", "2024-01-09 00:30:00+00:00"], utc=True)
    df = pd.DataFrame({"Close": [100.0, 101.0]}, index=idx)
    context = SimpleNamespace(
        job=SimpleNamespace(collection=runner.cfg.collections[0], timeframe="1d"),
        fetched_data=SimpleNamespace(raw_df=df),
    )

    decision, validated_data = runner._data_validation_common(context)

    assert decision.passed
    assert validated_data is not None
    assert validated_data.continuity["missing_bars"] == 0


def test_data_validation_without_data_quality_does_not_skip_on_continuity_errors(
    tmp_path, monkeypatch
):
    runner = _make_runner(tmp_path, monkeypatch)
    df = pd.DataFrame({"Close": [100.0]}, index=pd.to_datetime(["2024-01-01"]))
    context = SimpleNamespace(
        job=SimpleNamespace(collection=runner.cfg.collections[0], timeframe="1d"),
        fetched_data=SimpleNamespace(raw_df=df),
    )

    decision, validated_data = runner._data_validation_common(context)

    assert decision.passed
    assert decision.action == "continue"
    assert validated_data is not None
    assert validated_data.continuity == {}


def test_serialize_data_quality_profile_keeps_schema_and_key_order():
    data_quality = SimpleNamespace(
        on_fail="skip_job",
        min_data_points=42,
        is_verified=True,
        continuity=SimpleNamespace(
            min_score=0.8,
            max_missing_bar_pct=10.0,
            calendar=SimpleNamespace(kind="weekday", exchange=None, timezone="UTC"),
        ),
        kurtosis=3.5,
        outlier_detection=SimpleNamespace(
            max_outlier_pct=5.0,
            method="zscore",
            zscore_threshold=3.0,
        ),
        stationarity=SimpleNamespace(
            adf_pvalue_max=0.05,
            kpss_pvalue_min=0.05,
            min_points=20,
            regime_shift=SimpleNamespace(window=30, mean_shift_max=0.25, vol_ratio_max=1.5),
        ),
    )

    payload = BacktestRunner._serialize_data_quality_profile(data_quality)

    assert list(payload.keys()) == [
        "on_fail",
        "min_data_points",
        "is_verified",
        "continuity",
        "kurtosis",
        "outlier_detection",
        "stationarity",
    ]
    assert payload["continuity"] == {
        "min_score": 0.8,
        "max_missing_bar_pct": 10.0,
        "calendar": {"kind": "weekday", "exchange": None, "timezone": "UTC"},
    }
    assert payload["stationarity"] == {
        "adf_pvalue_max": 0.05,
        "kpss_pvalue_min": 0.05,
        "min_points": 20,
        "regime_shift": {
            "window": 30,
            "mean_shift_max": 0.25,
            "vol_ratio_max": 1.5,
        },
    }


def test_serialize_result_consistency_profile_keeps_schema_and_key_order():
    result_consistency = SimpleNamespace(
        min_metric=0.5,
        min_trades=20,
        outlier_dependency=SimpleNamespace(
            slices=5,
            profit_share_threshold=0.8,
            trade_share_threshold=0.05,
        ),
        execution_price_variance=SimpleNamespace(price_tolerance_bps=1.0),
        lookahead_shuffle_test=SimpleNamespace(
            permutations=100,
            pvalue_max=0.05,
            seed=1337,
            max_failed_permutations=2,
        ),
        transaction_cost_robustness=SimpleNamespace(
            mode="analytics",
            stress_multipliers=[2.0, 5.0],
            max_metric_drop_pct=0.3,
            breakeven=SimpleNamespace(
                enabled=True,
                min_multiplier=1.0,
                max_multiplier=5.0,
                max_iterations=8,
                tolerance=0.05,
            ),
        ),
    )

    payload = BacktestRunner._serialize_result_consistency_profile(result_consistency)

    assert list(payload.keys()) == [
        "min_metric",
        "min_trades",
        "outlier_dependency",
        "execution_price_variance",
        "lookahead_shuffle_test",
        "transaction_cost_robustness",
    ]
    assert payload["min_metric"] == pytest.approx(0.5)
    assert payload["min_trades"] == 20
    assert payload["outlier_dependency"] == {
        "slices": 5,
        "profit_share_threshold": 0.8,
        "trade_share_threshold": 0.05,
    }
    assert payload["execution_price_variance"] == {"price_tolerance_bps": 1.0}
    assert payload["lookahead_shuffle_test"] == {
        "permutations": 100,
        "pvalue_max": 0.05,
        "seed": 1337,
        "max_failed_permutations": 2,
    }
    assert payload["transaction_cost_robustness"] == {
        "mode": "analytics",
        "stress_multipliers": [2.0, 5.0],
        "max_metric_drop_pct": 0.3,
        "breakeven": {
            "enabled": True,
            "min_multiplier": 1.0,
            "max_multiplier": 5.0,
            "max_iterations": 8,
            "tolerance": 0.05,
        },
    }


def test_data_validation_stationarity_rejects_constant_series(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.validation = ValidationConfig(
        data_quality=ValidationDataQualityConfig(
            on_fail="skip_job",
            stationarity=ValidationStationarityConfig(
                adf_pvalue_max=0.05,
                min_points=20,
            ),
        )
    )
    resolve_validation_overrides(runner.cfg)
    df = pd.DataFrame({"Close": [100.0] * 8}, index=pd.date_range("2024-01-01", periods=8, freq="D"))
    context = SimpleNamespace(
        job=SimpleNamespace(collection=runner.cfg.collections[0], timeframe="1d"),
        fetched_data=SimpleNamespace(raw_df=df),
    )

    decision, validated_data = runner._data_validation_common(context)

    assert not decision.passed
    assert decision.action == "skip_job"
    assert validated_data is not None
    assert any(reason.startswith("stationarity_") for reason in validated_data.reliability_reasons)


def test_data_validation_collects_multiple_reliability_reasons(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.validation = ValidationConfig(
        data_quality=ValidationDataQualityConfig(
            continuity=ValidationContinuityConfig(max_missing_bar_pct=10.0),
            kurtosis=1.0,
            is_verified=False,
            on_fail="skip_job",
        )
    )
    resolve_validation_overrides(runner.cfg)
    closes = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 400.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    df = pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [100] * len(closes),
        },
        index=pd.date_range("2024-01-01", periods=len(closes), freq="D"),
    )
    context = SimpleNamespace(
        job=SimpleNamespace(collection=runner.cfg.collections[0], timeframe="1d"),
        fetched_data=SimpleNamespace(raw_df=df),
    )

    def _mock_continuity_score(cls, df, timeframe, **kwargs):
        return {
            "score": 0.7,
            "coverage_ratio": 0.8,
            "expected_bars": 20,
            "actual_bars": 13,
            "unique_bars": 13,
            "duplicate_bars": 0,
            "missing_bars": 4,
            "largest_gap_bars": 2,
        }

    monkeypatch.setattr(
        BacktestRunner,
        "compute_continuity_score",
        classmethod(_mock_continuity_score),
    )

    decision, validated_data = runner._data_validation_common(context)

    assert not decision.passed
    assert decision.action == "skip_job"
    assert validated_data is not None
    assert any(reason.startswith("max_missing_bar_pct_exceeded") for reason in validated_data.reliability_reasons)
    assert any(reason.startswith("max_kurtosis_exceeded") for reason in validated_data.reliability_reasons)
    assert "collection_not_verified" in validated_data.reliability_reasons


def test_stationarity_adf_reason_returns_indeterminate_when_statsmodels_missing():
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    raw_df = pd.DataFrame({"Close": np.linspace(100.0, 120.0, len(idx))}, index=idx)
    stationarity_cfg = ValidationStationarityConfig(
        adf_pvalue_max=0.05,
        min_points=20,
    )

    original_adfuller = BacktestRunner._stationarity_adfuller
    try:
        BacktestRunner._stationarity_adfuller = staticmethod(lambda: None)
        _, reason, _ = BacktestRunner._stationarity_adf_assessment(raw_df, stationarity_cfg)
    finally:
        BacktestRunner._stationarity_adfuller = original_adfuller

    assert reason == "stationarity_indeterminate(reason=statsmodels_missing)"


def test_stationarity_kpss_reason_returns_indeterminate_when_statsmodels_missing():
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    raw_df = pd.DataFrame({"Close": np.linspace(100.0, 120.0, len(idx))}, index=idx)
    stationarity_cfg = ValidationStationarityConfig(
        adf_pvalue_max=0.05,
        kpss_pvalue_min=0.05,
        min_points=20,
    )

    original_kpss = BacktestRunner._stationarity_kpss
    try:
        BacktestRunner._stationarity_kpss = staticmethod(lambda: None)
        _, reason, _ = BacktestRunner._stationarity_kpss_assessment(raw_df, stationarity_cfg)
    finally:
        BacktestRunner._stationarity_kpss = original_kpss

    assert reason == "stationarity_indeterminate(reason=statsmodels_missing)"


def test_stationarity_regime_shift_reason_flags_mean_and_vol_shift():
    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    first_half = np.array([0.001, 0.002, -0.001, 0.0] * 10, dtype=float)
    second_half = np.array([0.04, 0.05, 0.03, 0.06] * 10, dtype=float)
    returns = np.concatenate([first_half, second_half])
    close = 100.0 * np.exp(np.cumsum(returns))
    raw_df = pd.DataFrame({"Close": close}, index=idx)
    stationarity_cfg = ValidationStationarityConfig(
        adf_pvalue_max=0.05,
        min_points=20,
        regime_shift=ValidationStationarityRegimeShiftConfig(
            window=20,
            mean_shift_max=1.0,
            vol_ratio_max=1.25,
        ),
    )

    reasons = BacktestRunner._stationarity_regime_shift_reason(raw_df, stationarity_cfg)

    assert len(reasons) == 2
    assert any(reason.startswith("stationarity_regime_shift_mean_shift_exceeded(") for reason in reasons)
    assert any(reason.startswith("stationarity_regime_shift_vol_ratio_exceeded(") for reason in reasons)


def test_stationarity_reasons_handles_none_min_points():
    idx = pd.date_range("2024-01-01", periods=29, freq="D")
    close = pd.Series(np.linspace(100.0, 140.0, len(idx)), index=idx)
    raw_df = pd.DataFrame({"Close": close})
    stationarity_cfg = ValidationStationarityConfig(
        adf_pvalue_max=0.05,
        min_points=None,
        regime_shift=ValidationStationarityRegimeShiftConfig(
            window=10,
            mean_shift_max=0.1,
            vol_ratio_max=1.25,
        ),
    )

    reasons = BacktestRunner._stationarity_reasons(raw_df, stationarity_cfg)

    assert "stationarity_min_points_not_met(required=30, available=28)" in reasons
    assert (
        "stationarity_regime_shift_not_enough_points(required=30, available=28)"
        in reasons
    )


def test_stationarity_reasons_deduplicates_min_points_reason_for_adf_and_kpss():
    idx = pd.date_range("2024-01-01", periods=29, freq="D")
    close = pd.Series(np.linspace(100.0, 140.0, len(idx)), index=idx)
    raw_df = pd.DataFrame({"Close": close})
    stationarity_cfg = ValidationStationarityConfig(
        adf_pvalue_max=0.05,
        kpss_pvalue_min=0.05,
        min_points=None,
    )

    reasons = BacktestRunner._stationarity_reasons(raw_df, stationarity_cfg)

    expected_reason = "stationarity_min_points_not_met(required=30, available=28)"
    assert reasons.count(expected_reason) == 1


def test_stationarity_reasons_flags_adf_kpss_conflict():
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    raw_df = pd.DataFrame({"Close": np.linspace(100.0, 120.0, len(idx))}, index=idx)
    stationarity_cfg = ValidationStationarityConfig(
        adf_pvalue_max=0.05,
        kpss_pvalue_min=0.05,
        min_points=20,
    )

    original_adf_assessment = BacktestRunner._stationarity_adf_assessment
    original_kpss_assessment = BacktestRunner._stationarity_kpss_assessment
    try:
        BacktestRunner._stationarity_adf_assessment = classmethod(
            lambda cls, raw_df, stationarity_cfg, **kwargs: (True, None, 0.01)
        )
        BacktestRunner._stationarity_kpss_assessment = classmethod(
            lambda cls, raw_df, stationarity_cfg, **kwargs: (False, None, 0.01)
        )
        reasons = BacktestRunner._stationarity_reasons(raw_df, stationarity_cfg)
    finally:
        BacktestRunner._stationarity_adf_assessment = original_adf_assessment
        BacktestRunner._stationarity_kpss_assessment = original_kpss_assessment

    assert any(reason.startswith("stationarity_test_conflict(") for reason in reasons)


def test_stationarity_adf_reason_flags_constant_returns_series():
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    close = pd.Series([100.0 * (1.01 ** i) for i in range(len(idx))], index=idx)
    raw_df = pd.DataFrame({"Close": close})
    stationarity_cfg = ValidationStationarityConfig(
        adf_pvalue_max=0.01,
        min_points=20,
    )

    _, reason, _ = BacktestRunner._stationarity_adf_assessment(raw_df, stationarity_cfg)

    assert reason is not None
    assert reason.startswith("stationarity_")


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


def test_compose_gate_decisions_prefers_skip_collection_over_reject_result(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)

    reject_result = GateDecision(
        False,
        "reject_result",
        ["reject_reason"],
        "strategy_validation",
    )
    skip_collection = GateDecision(
        False,
        "skip_collection",
        ["collection_reason"],
        "strategy_validation",
    )

    decision = runner._compose_gate_decisions("strategy_validation", reject_result, skip_collection)
    assert decision.action == "skip_collection"
    assert decision.passed is False
    assert set(decision.reasons) == {"reject_reason", "collection_reason"}


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


def test_run_all_cached_invalid_outcomes_remain_invalid(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)

    class _ReplayEvaluationCache(_StubEvaluationCache):
        def __init__(self):
            super().__init__()
            self._store = {}

        @staticmethod
        def _cache_key(payload):
            return (
                payload["collection"],
                payload["symbol"],
                payload["timeframe"],
                payload["strategy"],
                json.dumps(payload["params"], sort_keys=True),
                payload["metric_name"],
                payload["data_fingerprint"],
                float(payload["fees"]),
                float(payload["slippage"]),
                payload["evaluation_mode"],
                payload["mode_config_hash"],
                payload["validation_config_hash"],
                payload.get("strategy_fingerprint", ""),
            )

        def get(self, **kwargs):
            self.retrieved.append(kwargs)
            return self._store.get(self._cache_key(kwargs))

        def set(self, **kwargs):
            self.saved.append(kwargs)
            self._store[self._cache_key(kwargs)] = {
                "metric_value": float(kwargs["metric_value"]),
                "stats": dict(kwargs["stats"]),
            }

    class _AlwaysInvalidEvaluator:
        def __init__(self, calls):
            self._calls = calls

        def evaluate(self, *args, **kwargs):
            self._calls["count"] += 1
            return EvaluationOutcome(
                metric_value=123.0,
                stats={"forced_invalid": True},
                valid=False,
                attempted=True,
                simulation_executed=True,
                metric_computed=True,
                reject_reason="forced_invalid",
            )

    runner.evaluation_cache = _ReplayEvaluationCache()
    runner.mode_config_hash = runner.evaluation_cache.hash_mode_config(runner.mode_config)
    eval_calls = {"count": 0}
    monkeypatch.setattr(
        BacktestRunner,
        "_get_evaluator",
        lambda self: _AlwaysInvalidEvaluator(eval_calls),
    )

    first_results = runner.run_all()
    assert first_results == []
    assert eval_calls["count"] > 0
    assert runner.evaluation_cache.saved
    assert all(
        saved["metric_value"] == float("-inf") for saved in runner.evaluation_cache.saved
    )

    def _fail_if_fresh_eval(self):
        raise AssertionError("Fresh evaluator should not run when evaluation cache is warm")

    monkeypatch.setattr(BacktestRunner, "_get_evaluator", _fail_if_fresh_eval)
    second_results = runner.run_all()
    assert second_results == []
    assert runner.metrics["result_cache_hits"] == 0


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
        return returns, equity, stats, pd.DataFrame()

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
    returns, equity_curve, stats, trades_frame = result
    assert len(returns) == len(equity_curve)
    assert "sharpe" in stats and "trades" in stats
    assert stats["trades"] == 1
    assert isinstance(trades_frame, pd.DataFrame)


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


def test_run_all_evaluator_failure_surfaces_as_strategy_optimization_failure(
    tmp_path, monkeypatch
):
    runner = _make_runner(tmp_path, monkeypatch)

    class _BoomEvaluator:
        def evaluate(self, *args, **kwargs):
            raise RuntimeError("evaluator boom")

    monkeypatch.setattr(BacktestRunner, "_get_evaluator", lambda self: _BoomEvaluator())

    results = runner.run_all()

    assert results == []
    assert len(runner.failures) == 1
    failure = runner.failures[0]
    assert failure["stage"] == "strategy_optimization"
    assert failure["error"] == "evaluator boom"


def test_run_all_evaluation_cache_failure_surfaces_as_strategy_optimization_failure(
    tmp_path, monkeypatch
):
    runner = _make_runner(tmp_path, monkeypatch)

    def _boom_cache_get(**kwargs):
        raise RuntimeError("cache boom")

    runner.evaluation_cache.get = _boom_cache_get  # type: ignore[assignment]

    results = runner.run_all()

    assert results == []
    assert len(runner.failures) == 1
    failure = runner.failures[0]
    assert failure["stage"] == "strategy_optimization"
    assert failure["error"] == "cache boom"


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


def _make_trending_ohlcv(periods: int) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    base = np.arange(1, periods + 1, dtype=float)
    return pd.DataFrame(
        {
            "Open": base,
            "High": base + 0.5,
            "Low": base - 0.5,
            "Close": base,
            "Volume": np.full(periods, 100.0),
        },
        index=dates,
    )


def _make_prepared_data(
    *,
    fees: float = 0.00005,
    slippage: float = 0.00005,
) -> ExecutionPreparedData:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    data_frame = pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0, 14.0],
            "high": [10.5, 11.5, 12.5, 13.5, 14.5],
            "low": [9.5, 10.5, 11.5, 12.5, 13.5],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0],
            "volume": [100.0] * 5,
        },
        index=dates,
    )
    return ExecutionPreparedData(
        data_frame=data_frame,
        dates=dates,
        fees=fees,
        slippage=slippage,
        fractional=True,
        bars_per_year=252,
        fingerprint="prepared-fingerprint",
    )


def _patch_source_with_bars(monkeypatch, bars: int) -> None:
    class _Source:
        def fetch(self, symbol, timeframe, only_cached=False):
            return _make_ohlcv(bars)

    monkeypatch.setattr(BacktestRunner, "_make_source", lambda self, col: _Source())


def _patch_trending_source(monkeypatch, bars: int = 25) -> None:
    class _Source:
        def fetch(self, symbol, timeframe, only_cached=False):
            return _make_trending_ohlcv(bars)

    monkeypatch.setattr(BacktestRunner, "_make_source", lambda self, col: _Source())


def _lookahead_shuffle_test_config(
    *,
    permutations: int = 100,
    pvalue_max: float = 0.05,
    seed: int = 7,
    max_failed_permutations: int | None = None,
) -> ValidationLookaheadShuffleTestConfig:
    return ValidationLookaheadShuffleTestConfig(
        permutations=permutations,
        pvalue_max=pvalue_max,
        seed=seed,
        max_failed_permutations=max_failed_permutations,
    )


def _transaction_cost_robustness_config(
    *,
    mode: str = "analytics",
    stress_multipliers: list[float] | None = None,
    max_metric_drop_pct: float = 0.3,
    breakeven: ResultConsistencyTransactionCostBreakevenConfig | None = None,
) -> ResultConsistencyTransactionCostRobustnessConfig:
    return ResultConsistencyTransactionCostRobustnessConfig(
        mode=mode,
        stress_multipliers=stress_multipliers if stress_multipliers is not None else [2.0, 5.0],
        max_metric_drop_pct=max_metric_drop_pct,
        breakeven=breakeven,
    )


def _configure_result_consistency_runner(
    runner: BacktestRunner,
    *,
    strategy_name: str,
    strategy_cls: type[BaseStrategy],
    result_consistency: ResultConsistencyConfig,
) -> None:
    runner.cfg.strategies = [
        StrategyConfig(
            name=strategy_name,
            module=None,
            cls=None,
            params={},
        )
    ]
    runner.external_index = {strategy_name: strategy_cls}
    runner.cfg.validation = ValidationConfig(result_consistency=result_consistency)
    resolve_validation_overrides(runner.cfg)


def _build_strategy_validation_artifacts(
    runner: BacktestRunner,
    *,
    strategy_name: str,
    raw_df: pd.DataFrame | None = None,
    prepared_data=None,
    outcome: StrategyEvalOutcome | None = None,
):
    effective_raw_df = raw_df if raw_df is not None else _make_trending_ohlcv(25)
    state = JobState(
        job=JobContext(
            collection=runner.cfg.collections[0],
            symbol="AAPL",
            timeframe="1d",
            source="custom",
        )
    )
    plan = runner._strategy_create_plan(state, strategy_name)
    validated_data = ValidatedData(
        raw_df=effective_raw_df,
        continuity={},
        reliability_on_fail="skip_optimization",
        reliability_reasons=[],
    )
    context_kwargs = {
        "stage": "strategy_validation",
        "state": state,
        "mode": "backtest",
        "job": state.job,
        "validated_data": validated_data,
    }
    if prepared_data is not None:
        context_kwargs["prepared_data"] = prepared_data
    if outcome is not None:
        context_kwargs["plan"] = plan
        context_kwargs["outcome"] = outcome
    context = ValidationContext(**context_kwargs)
    policy = runner._load_lookahead_shuffle_test_policy(state.job.collection)
    return state, plan, validated_data, context, policy


def _build_transaction_cost_validation_artifacts(
    runner: BacktestRunner,
    *,
    strategy_name: str,
    prepared_data: ExecutionPreparedData | None = None,
    policy: ResultConsistencyTransactionCostRobustnessConfig | None = None,
    baseline_metric: float = 0.85,
    baseline_profit: float = 0.15,
):
    effective_prepared = prepared_data if prepared_data is not None else _make_prepared_data()
    state, plan, validated_data, context, _ = _build_strategy_validation_artifacts(
        runner,
        strategy_name=strategy_name,
        prepared_data=effective_prepared,
    )
    tc_policy = policy if policy is not None else _transaction_cost_robustness_config()
    run_ctx = TransactionCostRobustnessRunContext(
        context=context,
        plan=plan,
        policy=tc_policy,
        prepared=effective_prepared,
        full_params=plan.fixed_params.copy(),
        baseline_metric=baseline_metric,
        baseline_profit=baseline_profit,
    )
    return state, plan, validated_data, context, run_ctx


def _fake_transaction_cost_scenario(
    self,
    run_ctx: TransactionCostRobustnessRunContext,
    multiplier: float,
):
    metric_value = 0.85 - 0.05 * (float(multiplier) - 1.0)
    profit = 0.15 - 0.05 * (float(multiplier) - 1.0)
    metric_drop_pct = max(0.0, (run_ctx.baseline_metric - metric_value) / run_ctx.baseline_metric)
    return {
        "is_complete": True,
        "status": "complete",
        "metric_name": self.cfg.metric,
        "multiplier": float(multiplier),
        "fees": float(run_ctx.prepared.fees) * float(multiplier),
        "slippage": float(run_ctx.prepared.slippage) * float(multiplier),
        "baseline_metric": run_ctx.baseline_metric,
        "baseline_profit": run_ctx.baseline_profit,
        "metric_value": metric_value,
        "profit": profit,
        "metric_drop_pct": metric_drop_pct,
        "metric_drop_exceeded": self._transaction_cost_drop_exceeds_threshold(
            metric_drop_pct,
            run_ctx.policy.max_metric_drop_pct,
        ),
        "profit_negative": profit < 0.0,
    }


def _transaction_cost_eval_outcome_from_request(request: EvaluationRequest) -> EvaluationOutcome:
    metric_value = 1.0 - 2000.0 * float(request.fees) - 1000.0 * float(request.slippage)
    profit = 0.3 - 2000.0 * float(request.fees) - 1000.0 * float(request.slippage)
    stats = {
        "sharpe": metric_value,
        "sortino": metric_value,
        "omega": 1.0,
        "tail_ratio": 1.0,
        "profit": profit,
        "pain_index": 0.0,
        "trades": 2,
        "max_drawdown": -0.1,
        "cagr": 0.1,
        "calmar": 1.0,
        "equity_curve": [],
        "drawdown_curve": [],
        "trades_log": [],
    }
    return EvaluationOutcome(
        metric_value=metric_value,
        stats=stats,
        valid=True,
        attempted=True,
        simulation_executed=True,
        metric_computed=True,
    )


def _patch_transaction_cost_evaluator(monkeypatch) -> None:
    def _fake_evaluate(self, request, prepared, entries, exits):
        return _transaction_cost_eval_outcome_from_request(request)

    monkeypatch.setattr(BacktestRunner, "_evaluate_strategy_outcome", _fake_evaluate)


def _setup_transaction_cost_run_all_runner(
    tmp_path,
    monkeypatch,
    *,
    mode: str,
    max_metric_drop_pct: float,
):
    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.collections[0].fees = 0.00005
    runner.cfg.collections[0].slippage = 0.00005
    _configure_result_consistency_runner(
        runner,
        strategy_name="dummy",
        strategy_cls=_DummyStrategy,
        result_consistency=_result_consistency_config(
            transaction_cost_robustness=_transaction_cost_robustness_config(
                mode=mode,
                max_metric_drop_pct=max_metric_drop_pct,
            )
        ),
    )
    _patch_trending_source(monkeypatch)
    _patch_transaction_cost_evaluator(monkeypatch)
    return runner


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
        return returns, equity, stats, pd.DataFrame()

    monkeypatch.setattr(BacktestRunner, "_run_pybroker_simulation", _fake_sim)
    return eval_calls


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


def test_run_all_runtime_signal_errors_threshold_one_skips_remaining_params(tmp_path, monkeypatch):
    calls = {"count": 0}

    class _BoomStrategy(BaseStrategy):
        name = "boom"

        def param_grid(self) -> dict[str, list[int]]:
            return {"window": [1, 2, 3, 4]}

        def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
            calls["count"] += 1
            raise RuntimeError("signal boom")

    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.strategies = []
    runner.external_index = {"boom": _BoomStrategy}
    runner.cfg.validation = ValidationConfig(
        optimization=OptimizationPolicyConfig(
            on_fail="baseline_only",
            min_bars=1,
            dof_multiplier=1,
            runtime_error_max_per_tuple=1,
        )
    )
    runner.cfg.param_search = "grid"

    results = runner.run_all()
    assert results == []
    assert calls["count"] == 1
    generate_failures = [f for f in runner.failures if f.get("stage") == "generate_signals"]
    assert len(generate_failures) == 1
    assert any(
        "runtime_error_threshold_exceeded(count=1, max_per_tuple=1)" in f.get("error", "")
        for f in runner.failures
    )


def test_run_all_runtime_signal_errors_are_not_capped_without_optimization_policy(tmp_path, monkeypatch):
    calls = {"count": 0}

    class _BoomStrategy(BaseStrategy):
        name = "boom"

        def param_grid(self) -> dict[str, list[int]]:
            return {"window": [1, 2, 3, 4]}

        def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
            calls["count"] += 1
            raise RuntimeError("signal boom")

    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.strategies = []
    runner.cfg.validation = None
    runner.external_index = {"boom": _BoomStrategy}
    runner.cfg.param_search = "grid"

    results = runner.run_all()
    assert results == []
    assert calls["count"] == 4
    generate_failures = [f for f in runner.failures if f.get("stage") == "generate_signals"]
    assert len(generate_failures) == 4
    assert all("runtime_error_threshold_exceeded" not in f.get("error", "") for f in runner.failures)


def test_load_optimization_policy_defaults_runtime_error_threshold_when_none(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    collection = CollectionConfig(
        name="demo",
        source="custom",
        symbols=["AAPL"],
        validation=ValidationConfig(
            optimization=OptimizationPolicyConfig(
                on_fail="baseline_only",
                min_bars=1,
                dof_multiplier=1,
                runtime_error_max_per_tuple=None,
            )
        ),
    )

    policy = runner._load_optimization_policy(collection)

    assert policy is not None
    assert policy[3] == 1


def test_run_all_runtime_signal_errors_threshold_n_skips_after_n(tmp_path, monkeypatch):
    calls = {"count": 0}

    class _BoomStrategy(BaseStrategy):
        name = "boom"

        def param_grid(self) -> dict[str, list[int]]:
            return {"window": [1, 2, 3, 4, 5]}

        def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
            calls["count"] += 1
            raise RuntimeError("signal boom")

    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.strategies = []
    runner.external_index = {"boom": _BoomStrategy}
    runner.cfg.validation = ValidationConfig(
        optimization=OptimizationPolicyConfig(
            on_fail="baseline_only",
            min_bars=1,
            dof_multiplier=1,
            runtime_error_max_per_tuple=3,
        )
    )
    runner.cfg.param_search = "grid"

    results = runner.run_all()
    assert results == []
    assert calls["count"] == 3
    generate_failures = [f for f in runner.failures if f.get("stage") == "generate_signals"]
    assert len(generate_failures) == 3
    assert any(
        "runtime_error_threshold_exceeded(count=3, max_per_tuple=3)" in f.get("error", "")
        for f in runner.failures
    )


def test_run_all_runtime_signal_error_threshold_is_tuple_isolated(tmp_path, monkeypatch):
    calls = {"count": 0}

    class _BoomStrategy(BaseStrategy):
        name = "boom"

        def param_grid(self) -> dict[str, list[int]]:
            return {"window": [1, 2, 3]}

        def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
            calls["count"] += 1
            raise RuntimeError("signal boom")

    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.strategies = []
    runner.cfg.timeframes = ["1d", "1wk"]
    runner.external_index = {"boom": _BoomStrategy}
    runner.cfg.validation = ValidationConfig(
        optimization=OptimizationPolicyConfig(
            on_fail="baseline_only",
            min_bars=1,
            dof_multiplier=1,
            runtime_error_max_per_tuple=1,
        )
    )
    runner.cfg.param_search = "grid"

    results = runner.run_all()
    assert results == []
    # Two tuples: (boom, AAPL, 1d) and (boom, AAPL, 1wk), each capped after first failure.
    assert calls["count"] == 2


def test_run_all_runtime_tuple_cap_still_records_strategy_validation_for_repeated_tuple(
    tmp_path, monkeypatch
):
    calls = {"count": 0}

    class _BoomStrategy(BaseStrategy):
        name = "boom"

        def param_grid(self) -> dict[str, list[int]]:
            return {"window": [1, 2, 3]}

        def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
            calls["count"] += 1
            raise RuntimeError("signal boom")

    collections = [
        CollectionConfig(
            name="demo",
            source="custom",
            symbols=["AAPL", "AAPL"],
            fees=0.0004,
            slippage=0.0003,
        )
    ]
    runner = _make_runner(tmp_path, monkeypatch, collections=collections)
    runner.cfg.strategies = []
    runner.external_index = {"boom": _BoomStrategy}
    runner.cfg.validation = ValidationConfig(
        optimization=OptimizationPolicyConfig(
            on_fail="baseline_only",
            min_bars=1,
            dof_multiplier=1,
            runtime_error_max_per_tuple=1,
        )
    )
    runner.cfg.param_search = "grid"

    results = runner.run_all()
    assert results == []
    assert calls["count"] == 1
    strategy_validation_failures = [
        failure
        for failure in runner.failures
        if failure.get("stage") == "strategy_validation" and failure.get("error") == "no_valid_candidate"
    ]
    assert len(strategy_validation_failures) == 2
    assert runner.metrics["symbols_tested"] == 2


def test_run_all_runtime_signal_error_threshold_isolated_per_collection(tmp_path, monkeypatch):
    calls = {"count": 0}

    class _BoomStrategy(BaseStrategy):
        name = "boom"

        def param_grid(self) -> dict[str, list[int]]:
            return {"window": [1, 2, 3, 4, 5]}

        def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
            calls["count"] += 1
            raise RuntimeError("signal boom")

    collections = [
        CollectionConfig(
            name="strict",
            source="custom",
            symbols=["AAPL"],
            validation=ValidationConfig(
                optimization=OptimizationPolicyConfig(
                    on_fail="baseline_only",
                    min_bars=1,
                    dof_multiplier=1,
                    runtime_error_max_per_tuple=1,
                )
            ),
        ),
        CollectionConfig(
            name="lenient",
            source="custom",
            symbols=["AAPL"],
            validation=ValidationConfig(
                optimization=OptimizationPolicyConfig(
                    on_fail="baseline_only",
                    min_bars=1,
                    dof_multiplier=1,
                    runtime_error_max_per_tuple=3,
                )
            ),
        ),
    ]
    runner = _make_runner(tmp_path, monkeypatch, collections=collections)
    runner.cfg.strategies = []
    runner.external_index = {"boom": _BoomStrategy}
    runner.cfg.param_search = "grid"

    results = runner.run_all()
    assert results == []
    assert calls["count"] == 4
    assert any(
        "runtime_error_threshold_exceeded(count=1, max_per_tuple=1)" in f.get("error", "")
        for f in runner.failures
    )
    assert any(
        "runtime_error_threshold_exceeded(count=3, max_per_tuple=3)" in f.get("error", "")
        for f in runner.failures
    )


def test_run_all_runtime_signal_error_threshold_resets_between_runs(tmp_path, monkeypatch):
    calls = {"count": 0}

    class _BoomStrategy(BaseStrategy):
        name = "boom"

        def param_grid(self) -> dict[str, list[int]]:
            return {"window": [1, 2, 3]}

        def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
            calls["count"] += 1
            raise RuntimeError("signal boom")

    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.strategies = []
    runner.external_index = {"boom": _BoomStrategy}
    runner.cfg.validation = ValidationConfig(
        optimization=OptimizationPolicyConfig(
            on_fail="baseline_only",
            min_bars=1,
            dof_multiplier=1,
            runtime_error_max_per_tuple=1,
        )
    )
    runner.cfg.param_search = "grid"

    first = runner.run_all()
    second = runner.run_all()
    assert first == []
    assert second == []
    assert calls["count"] == 2


def test_run_all_runtime_signal_error_threshold_does_not_escalate_job(tmp_path, monkeypatch):
    calls = {"count": 0}

    class _BoomStrategy(BaseStrategy):
        name = "boom"

        def param_grid(self) -> dict[str, list[int]]:
            return {"window": [1, 2, 3]}

        def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
            calls["count"] += 1
            raise RuntimeError("signal boom")

    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.strategies = []
    runner.external_index = {"boom": _BoomStrategy, "dummy": _DummyStrategy}
    runner.cfg.validation = ValidationConfig(
        optimization=OptimizationPolicyConfig(
            on_fail="baseline_only",
            min_bars=1,
            dof_multiplier=1,
            runtime_error_max_per_tuple=1,
        )
    )
    runner.cfg.param_search = "grid"
    eval_calls = _patch_pybroker_simulation(monkeypatch)

    results = runner.run_all()
    assert calls["count"] == 1
    assert eval_calls["count"] >= 1
    assert len(results) == 1
    assert results[0].strategy == "dummy"
    assert not any(
        failure["stage"] == "strategy_optimization" and failure["error"] == "skip_job"
        for failure in runner.failures
    )


def test_run_all_runtime_signal_error_threshold_stops_optuna_search(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    calls = {"count": 0}

    class _BoomStrategy(BaseStrategy):
        name = "boom"

        def param_grid(self) -> dict[str, list[int]]:
            return {"window": list(range(1, 11))}

        def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
            calls["count"] += 1
            raise RuntimeError("signal boom")

    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.strategies = []
    runner.external_index = {"boom": _BoomStrategy}
    runner.cfg.param_search = "optuna"
    runner.cfg.param_trials = 10
    runner.cfg.validation = ValidationConfig(
        optimization=OptimizationPolicyConfig(
            on_fail="baseline_only",
            min_bars=1,
            dof_multiplier=1,
            runtime_error_max_per_tuple=1,
        )
    )

    results = runner.run_all()
    assert results == []
    assert calls["count"] == 1


def test_strategy_run_baseline_skips_evaluation_when_runtime_tuple_is_capped(tmp_path, monkeypatch):
    eval_calls = {"count": 0}

    class _BaselineStrategy(BaseStrategy):
        name = "baseline_only"

        def param_grid(self) -> dict[str, list[int]]:
            return {}

        def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
            entries = pd.Series([True] + [False] * (len(df.index) - 1), index=df.index)
            exits = pd.Series([False] * (len(df.index) - 1) + [True], index=df.index)
            return entries, exits

    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.strategies = []
    runner.external_index = {"baseline_only": _BaselineStrategy}
    state = JobState(
        job=JobContext(
            collection=runner.cfg.collections[0],
            symbol="AAPL",
            timeframe="1d",
            source="custom",
        )
    )
    plan = runner._strategy_create_plan(state, "baseline_only")
    validated_data = ValidatedData(
        raw_df=_make_ohlcv(10),
        continuity={},
        reliability_on_fail="skip_optimization",
        reliability_reasons=[],
    )
    prep_decision, prepared = runner._execution_context_prepare(state, validated_data)
    assert prep_decision.passed is True
    assert prepared is not None

    def _track_eval(*args, **kwargs):
        eval_calls["count"] += 1
        return 0.0

    monkeypatch.setattr(BacktestRunner, "_strategy_evaluation", _track_eval)
    runner._runtime_signal_error_capped.add(runner._runtime_error_tuple_key(plan, state))

    outcome = runner._strategy_run(plan, state, validated_data, prepared)

    assert outcome is not None
    assert outcome.has_valid_candidate is False
    assert plan.optimization_skip_reason == "runtime_error_threshold_exceeded"
    assert eval_calls["count"] == 0


def test_run_all_reliability_min_data_points_skips_optimization(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.validation = ValidationConfig(
        data_quality=ValidationDataQualityConfig(
            min_data_points=10,
            on_fail="skip_optimization",
        ),
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


def test_run_all_reliability_not_verified_skips_optimization(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.validation = ValidationConfig(
        data_quality=ValidationDataQualityConfig(
            is_verified=False,
            on_fail="skip_optimization",
        ),
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
    assert "collection_not_verified" in reasons


def test_run_all_data_quality_without_continuity_still_computes_diagnostics(
    tmp_path, monkeypatch
):
    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.validation = ValidationConfig(
        data_quality=ValidationDataQualityConfig(
            on_fail="skip_job",
        ),
    )

    _patch_source_with_bars(monkeypatch, bars=5)
    _patch_pybroker_simulation(monkeypatch)

    results = runner.run_all()
    assert len(results) == 1
    continuity = results[0].stats["data_reliability"]["continuity"]
    assert continuity["expected_bars"] == 5
    assert continuity["actual_bars"] == 5
    assert continuity["missing_bars"] == 0


def test_run_all_data_quality_without_continuity_rejects_insufficient_bars(
    tmp_path, monkeypatch
):
    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.validation = ValidationConfig(
        data_quality=ValidationDataQualityConfig(
            on_fail="skip_optimization",
        ),
    )
    _patch_source_with_bars(monkeypatch, bars=1)
    eval_calls = _patch_pybroker_simulation(monkeypatch)

    results = runner.run_all()
    assert results == []
    assert eval_calls["count"] == 0
    assert len(runner.failures) == 1
    assert runner.failures[0]["stage"] == "data_validation"
    assert "insufficient_bars_for_continuity" in runner.failures[0]["error"]


def test_run_all_reliability_skip_evaluation_on_continuity_threshold(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.validation = ValidationConfig(
        data_quality=ValidationDataQualityConfig(
            continuity=ValidationContinuityConfig(min_score=0.95),
            on_fail="skip_job",
        ),
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
        data_quality=ValidationDataQualityConfig(
            continuity=ValidationContinuityConfig(max_missing_bar_pct=10.0),
            on_fail="skip_job",
        ),
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
        data_quality=ValidationDataQualityConfig(
            kurtosis=1.0,
            on_fail="skip_job",
        ),
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


def test_run_all_reliability_skip_evaluation_on_max_outlier_pct(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.validation = ValidationConfig(
        data_quality=ValidationDataQualityConfig(
            outlier_detection=ValidationOutlierDetectionConfig(
                max_outlier_pct=1.0,
                method="modified_zscore",
                zscore_threshold=3.5,
            ),
            on_fail="skip_job",
        ),
    )

    class _SpikySource:
        def fetch(self, symbol, timeframe, only_cached=False):
            idx = pd.date_range("2024-01-01", periods=52, freq="D")
            returns = [0.01, -0.008, 0.012, -0.009] * 12 + [0.70, 0.01, -0.008]
            closes = [100.0]
            for ret in returns:
                closes.append(closes[-1] * (1.0 + ret))
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

    monkeypatch.setattr(BacktestRunner, "_make_source", lambda self, col: _SpikySource())
    eval_calls = _patch_pybroker_simulation(monkeypatch)

    results = runner.run_all()
    assert results == []
    assert eval_calls["count"] == 0
    assert runner.failures
    failure = runner.failures[0]
    assert failure["stage"] == "data_validation"
    assert "max_outlier_pct_exceeded" in failure["error"]


def test_lookahead_shuffle_test_result_is_deterministic(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch, patch_source=False)
    _configure_result_consistency_runner(
        runner,
        strategy_name="leaky_shuffle",
        strategy_cls=_LeakyShuffleStrategy,
        result_consistency=_result_consistency_config(
            lookahead_shuffle_test=_lookahead_shuffle_test_config()
        ),
    )
    _, plan, _, context, policy = _build_strategy_validation_artifacts(
        runner,
        strategy_name="leaky_shuffle",
    )
    runner.metrics = {"result_cache_misses": 0}
    reason_one, meta_one = runner._lookahead_shuffle_test_result(context, plan, policy)
    reason_two, meta_two = runner._lookahead_shuffle_test_result(context, plan, policy)

    assert reason_one == reason_two
    assert meta_one == meta_two
    assert meta_one is not None
    assert meta_one["is_complete"] is True
    assert meta_one["seed"] == meta_two["seed"]
    assert meta_one["median_shuffled_metric"] > 0
    assert runner.metrics["result_cache_misses"] == 0
    assert runner.evaluation_cache.saved == []
    assert runner._runtime_signal_error_counts == {}


def test_strategy_validation_result_consistency_min_gates_fail_fast_before_lookahead(
    tmp_path, monkeypatch
):
    runner = _make_runner(tmp_path, monkeypatch, patch_source=False)
    _configure_result_consistency_runner(
        runner,
        strategy_name="leaky_shuffle",
        strategy_cls=_LeakyShuffleStrategy,
        result_consistency=_result_consistency_config(
            min_metric=1.0,
            min_trades=10,
            lookahead_shuffle_test=_lookahead_shuffle_test_config(),
        ),
    )
    outcome = StrategyEvalOutcome(
        best_val=0.2,
        best_params={},
        best_stats={"trades": 2},
        has_valid_candidate=True,
        evaluations=1,
        skipped_reason=None,
        strategy="leaky_shuffle",
        job=JobContext(
            collection=runner.cfg.collections[0],
            symbol="AAPL",
            timeframe="1d",
            source="custom",
        ),
    )
    _, _, _, context, _ = _build_strategy_validation_artifacts(
        runner,
        strategy_name="leaky_shuffle",
        outcome=outcome,
    )

    def _unexpected_lookahead(*_args, **_kwargs):
        raise AssertionError("lookahead should be skipped after min gates fail")

    monkeypatch.setattr(
        runner,
        "_lookahead_shuffle_test_result",
        MethodType(_unexpected_lookahead, runner),
    )

    decision = runner._strategy_validate_results_common(context)

    assert not decision.passed
    assert decision.action == "reject_result"
    assert "min_metric_not_met(required=1.0, available=0.2)" in decision.reasons
    assert "min_trades_not_met(required=10, available=2)" in decision.reasons


def test_lookahead_shuffle_test_rejects_on_pvalue_threshold(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch, patch_source=False)
    _configure_result_consistency_runner(
        runner,
        strategy_name="leaky_shuffle",
        strategy_cls=_LeakyShuffleStrategy,
        result_consistency=_result_consistency_config(
            lookahead_shuffle_test=_lookahead_shuffle_test_config()
        ),
    )
    _, plan, _, context, policy = _build_strategy_validation_artifacts(
        runner,
        strategy_name="leaky_shuffle",
    )
    assert policy is not None

    def _stub_permutations(self, run_ctx):
        return ([0.8] * 100), 0, None

    monkeypatch.setattr(
        runner,
        "_run_lookahead_shuffle_permutations",
        MethodType(_stub_permutations, runner),
    )

    reason, meta = runner._lookahead_shuffle_test_result(
        context,
        plan,
        policy,
        observed_metric=0.1,
    )

    assert reason is not None
    assert "lookahead_shuffle_test_pvalue_exceeded" in reason
    assert meta is not None
    assert meta["shuffle_pvalue"] == pytest.approx(1.0)
    assert meta["observed_metric"] == pytest.approx(0.1)


def test_lookahead_shuffle_uses_isolated_plan_state(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch, patch_source=False)
    _configure_result_consistency_runner(
        runner,
        strategy_name="leaky_shuffle",
        strategy_cls=_LeakyShuffleStrategy,
        result_consistency=_result_consistency_config(
            lookahead_shuffle_test=_lookahead_shuffle_test_config(pvalue_max=1.0)
        ),
    )
    _, plan, _, context, policy = _build_strategy_validation_artifacts(
        runner,
        strategy_name="leaky_shuffle",
    )
    assert policy is not None

    original_build_request = runner._build_evaluation_request

    def _mutating_build_request(self, eval_plan, state, prepared, full_params, *, cacheable=True):
        eval_plan.evaluations += 99
        eval_plan.best_val = 1_234.0
        eval_plan.best_stats = {"mutated": True}
        return original_build_request(
            eval_plan,
            state,
            prepared,
            full_params,
            cacheable=cacheable,
        )

    monkeypatch.setattr(
        runner,
        "_build_evaluation_request",
        MethodType(_mutating_build_request, runner),
    )

    reason, meta = runner._lookahead_shuffle_test_result(context, plan, policy)

    assert reason is None
    assert meta is not None
    assert plan.evaluations == 0
    assert plan.best_val == float("-inf")
    assert plan.best_stats is None
    assert plan.best_params is None


def test_lookahead_shuffle_requests_are_marked_non_cacheable(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch, patch_source=False)
    _configure_result_consistency_runner(
        runner,
        strategy_name="leaky_shuffle",
        strategy_cls=_LeakyShuffleStrategy,
        result_consistency=_result_consistency_config(
            lookahead_shuffle_test=_lookahead_shuffle_test_config(pvalue_max=1.0)
        ),
    )
    _, plan, _, context, policy = _build_strategy_validation_artifacts(
        runner,
        strategy_name="leaky_shuffle",
    )
    assert policy is not None
    seen_cacheable_flags: list[bool] = []

    def _capture_evaluation_request(self, request, prepared, entries, exits):
        seen_cacheable_flags.append(request.cacheable)
        return EvaluationOutcome(
            metric_value=1.0,
            stats={},
            valid=True,
            attempted=True,
            simulation_executed=True,
            metric_computed=True,
        )

    monkeypatch.setattr(
        runner,
        "_evaluate_strategy_outcome",
        MethodType(_capture_evaluation_request, runner),
    )

    reason, meta = runner._lookahead_shuffle_test_result(context, plan, policy)

    assert reason is None
    assert meta is not None
    assert seen_cacheable_flags == [False] * policy.permutations


def test_lookahead_shuffle_test_result_does_not_track_runtime_errors(
    tmp_path, monkeypatch
):
    class _BoomShuffleStrategy(BaseStrategy):
        name = "boom_shuffle"

        def param_grid(self) -> dict[str, list[int]]:
            return {}

        def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
            raise RuntimeError("signal boom")

    runner = _make_runner(tmp_path, monkeypatch, patch_source=False)
    _configure_result_consistency_runner(
        runner,
        strategy_name="boom_shuffle",
        strategy_cls=_BoomShuffleStrategy,
        result_consistency=_result_consistency_config(
            lookahead_shuffle_test=_lookahead_shuffle_test_config()
        ),
    )
    _, plan, _, context, policy = _build_strategy_validation_artifacts(
        runner,
        strategy_name="boom_shuffle",
    )
    runner.metrics = {"result_cache_misses": 0}

    reason, meta = runner._lookahead_shuffle_test_result(context, plan, policy)

    assert reason is not None
    assert reason.startswith("lookahead_shuffle_test_indeterminate(")
    assert meta is not None
    assert meta["is_complete"] is False
    assert meta["reason"] == "no_finite_metrics"
    assert meta["failed_permutations"] == policy.permutations
    assert meta["max_failed_permutations"] is None
    assert runner._runtime_signal_error_counts == {}
    assert runner.metrics["result_cache_misses"] == 0
    assert runner.evaluation_cache.saved == []


def test_lookahead_shuffle_test_result_limits_failed_permutations(
    tmp_path, monkeypatch
):
    class _BoomShuffleStrategy(BaseStrategy):
        name = "boom_shuffle"

        def param_grid(self) -> dict[str, list[int]]:
            return {}

        def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
            raise RuntimeError("signal boom")

    runner = _make_runner(tmp_path, monkeypatch, patch_source=False)
    _configure_result_consistency_runner(
        runner,
        strategy_name="boom_shuffle",
        strategy_cls=_BoomShuffleStrategy,
        result_consistency=_result_consistency_config(
            lookahead_shuffle_test=_lookahead_shuffle_test_config(
                max_failed_permutations=1
            )
        ),
    )
    _, plan, _, context, policy = _build_strategy_validation_artifacts(
        runner,
        strategy_name="boom_shuffle",
    )
    runner.metrics = {"result_cache_misses": 0}

    reason, meta = runner._lookahead_shuffle_test_result(context, plan, policy)

    assert reason is not None
    assert "too_many_failed_permutations" in reason
    assert meta is not None
    assert meta["is_complete"] is False
    assert meta["reason"] == "too_many_failed_permutations"
    assert meta["failed_permutations"] == 2
    assert meta["max_failed_permutations"] == 1


def test_run_all_lookahead_shuffle_test_indeterminate_rejects_result(
    tmp_path, monkeypatch
):
    class _ShuffleSensitiveStrategy(BaseStrategy):
        name = "shuffle_sensitive"

        def param_grid(self) -> dict[str, list[int]]:
            return {}

        def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
            if not df["Close"].is_monotonic_increasing:
                raise RuntimeError("signal boom")
            entries = pd.Series(False, index=df.index)
            exits = pd.Series(False, index=df.index)
            entries.iloc[0] = True
            exits.iloc[-1] = True
            return entries, exits

    runner = _make_runner(tmp_path, monkeypatch, patch_source=False)
    _configure_result_consistency_runner(
        runner,
        strategy_name="shuffle_sensitive",
        strategy_cls=_ShuffleSensitiveStrategy,
        result_consistency=_result_consistency_config(
            lookahead_shuffle_test=_lookahead_shuffle_test_config()
        ),
    )
    _patch_trending_source(monkeypatch)
    eval_calls = _patch_pybroker_simulation(monkeypatch)

    results = runner.run_all()

    assert results == []
    assert eval_calls["count"] == 1
    assert len(runner.failures) == 1
    failure = runner.failures[0]
    assert failure["stage"] == "strategy_validation"
    assert failure["error"].startswith("lookahead_shuffle_test_indeterminate(")
    assert failure["strategy"] == "shuffle_sensitive"
    assert "result_validation" not in failure


def test_run_all_lookahead_shuffle_test_rejects_result_in_strategy_validation(
    tmp_path, monkeypatch
):
    runner = _make_runner(tmp_path, monkeypatch)
    _configure_result_consistency_runner(
        runner,
        strategy_name="leaky_shuffle",
        strategy_cls=_LeakyShuffleStrategy,
        result_consistency=_result_consistency_config(
            lookahead_shuffle_test=_lookahead_shuffle_test_config()
        ),
    )
    _patch_trending_source(monkeypatch)
    eval_calls = _patch_pybroker_simulation(monkeypatch)

    results = runner.run_all()

    assert results == []
    assert eval_calls["count"] == 1 + 100
    assert len(runner.failures) == 1
    failure = runner.failures[0]
    assert failure["stage"] == "strategy_validation"
    assert "lookahead_shuffle_test_pvalue_exceeded" in failure["error"]
    assert failure["strategy"] == "leaky_shuffle"
    assert "result_validation" not in failure


def test_run_all_lookahead_shuffle_test_attaches_post_run_meta(
    tmp_path, monkeypatch
):
    runner = _make_runner(tmp_path, monkeypatch)
    _configure_result_consistency_runner(
        runner,
        strategy_name="leaky_shuffle",
        strategy_cls=_LeakyShuffleStrategy,
        result_consistency=_result_consistency_config(
            lookahead_shuffle_test=_lookahead_shuffle_test_config(pvalue_max=1.0)
        ),
    )
    _patch_trending_source(monkeypatch)
    eval_calls = _patch_pybroker_simulation(monkeypatch)

    results = runner.run_all()

    assert len(results) == 1
    assert eval_calls["count"] == 1 + 100
    post_run_meta = results[0].stats.get("post_run_meta")
    assert post_run_meta is not None
    assert post_run_meta["lookahead_shuffle_test"]["is_complete"] is True


def test_run_all_lookahead_shuffle_test_does_not_mutate_cached_stats_payload(
    tmp_path, monkeypatch
):
    runner = _make_runner(tmp_path, monkeypatch)
    _configure_result_consistency_runner(
        runner,
        strategy_name="leaky_shuffle",
        strategy_cls=_LeakyShuffleStrategy,
        result_consistency=_result_consistency_config(
            lookahead_shuffle_test=_lookahead_shuffle_test_config(pvalue_max=1.0)
        ),
    )
    _patch_trending_source(monkeypatch)
    _patch_pybroker_simulation(monkeypatch)

    results = runner.run_all()

    assert len(results) == 1
    assert runner.results_cache.saved
    assert "post_run_meta" not in runner.results_cache.saved[0]["stats"]
    post_run_meta = results[0].stats.get("post_run_meta")
    assert post_run_meta is not None
    assert post_run_meta["lookahead_shuffle_test"]["is_complete"] is True


def test_transaction_cost_robustness_result_attaches_meta_without_cache_pollution(
    tmp_path, monkeypatch
):
    runner = _make_runner(tmp_path, monkeypatch, patch_source=False)
    _, _, _, _, run_ctx = _build_transaction_cost_validation_artifacts(
        runner,
        strategy_name="dummy",
        policy=_transaction_cost_robustness_config(mode="analytics"),
    )
    monkeypatch.setattr(
        runner,
        "_transaction_cost_robustness_scenario",
        MethodType(_fake_transaction_cost_scenario, runner),
    )

    reason, meta = runner._transaction_cost_robustness_result(run_ctx)

    assert reason is None
    assert meta is not None
    assert meta["is_complete"] is True
    assert meta["status"] == "complete"
    assert meta["stress_multipliers"] == [2.0, 5.0]
    assert meta["stress_scenarios"][0]["multiplier"] == pytest.approx(2.0)
    assert any(
        "transaction_cost_robustness_negative_profit" in breach_reason
        for breach_reason in meta["breach_reasons"]
    )
    assert runner.evaluation_cache.retrieved == []
    assert runner.results_cache.saved == []


def test_transaction_cost_robustness_result_enforce_rejects_on_metric_drop_boundary(
    tmp_path, monkeypatch
):
    runner = _make_runner(tmp_path, monkeypatch, patch_source=False)
    _, _, _, _, run_ctx = _build_transaction_cost_validation_artifacts(
        runner,
        strategy_name="dummy",
        policy=_transaction_cost_robustness_config(mode="enforce", max_metric_drop_pct=0.05),
    )
    monkeypatch.setattr(
        runner,
        "_transaction_cost_robustness_scenario",
        MethodType(_fake_transaction_cost_scenario, runner),
    )

    reason, meta = runner._transaction_cost_robustness_result(run_ctx)

    assert reason is not None
    assert "transaction_cost_robustness_metric_drop_exceeded" in reason
    assert meta is not None
    assert meta["status"] == "complete"
    assert meta["smallest_multiplier_metric_drop_pct"] > 0.05


def test_transaction_cost_robustness_result_enforce_rejects_on_negative_profit(
    tmp_path, monkeypatch
):
    runner = _make_runner(tmp_path, monkeypatch, patch_source=False)
    _, _, _, _, run_ctx = _build_transaction_cost_validation_artifacts(
        runner,
        strategy_name="dummy",
        policy=_transaction_cost_robustness_config(mode="enforce", max_metric_drop_pct=0.3),
    )
    monkeypatch.setattr(
        runner,
        "_transaction_cost_robustness_scenario",
        MethodType(_fake_transaction_cost_scenario, runner),
    )

    reason, meta = runner._transaction_cost_robustness_result(run_ctx)

    assert reason is not None
    assert "transaction_cost_robustness_negative_profit" in reason
    assert meta is not None
    assert meta["status"] == "complete"
    assert any(
        scenario["profit_negative"] is True
        for scenario in meta["stress_scenarios"]
        if np.isclose(float(scenario["multiplier"]), 5.0)
    )


def test_transaction_cost_robustness_scenario_reuses_aligned_signals(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch, patch_source=False)
    _, _, _, _, run_ctx = _build_transaction_cost_validation_artifacts(
        runner,
        strategy_name="dummy",
        policy=_transaction_cost_robustness_config(mode="analytics"),
    )
    _patch_transaction_cost_evaluator(monkeypatch)
    signal_calls = {"count": 0}
    original_generate_aligned_signals = runner._generate_aligned_signals

    def _count_generate_aligned_signals(self, strategy, raw_df, params, **kwargs):
        signal_calls["count"] += 1
        return original_generate_aligned_signals(strategy, raw_df, params, **kwargs)

    monkeypatch.setattr(
        runner,
        "_generate_aligned_signals",
        MethodType(_count_generate_aligned_signals, runner),
    )

    first = runner._transaction_cost_robustness_scenario(run_ctx, 2.0)
    second = runner._transaction_cost_robustness_scenario(run_ctx, 5.0)

    assert first["is_complete"] is True
    assert second["is_complete"] is True
    assert signal_calls["count"] == 1
    assert run_ctx.aligned_signals is not None


def test_transaction_cost_robustness_scenario_non_positive_baseline_is_indeterminate(
    tmp_path, monkeypatch
):
    runner = _make_runner(tmp_path, monkeypatch, patch_source=False)
    _, _, _, _, run_ctx = _build_transaction_cost_validation_artifacts(
        runner,
        strategy_name="dummy",
        policy=_transaction_cost_robustness_config(mode="analytics"),
        baseline_metric=0.0,
    )
    _patch_transaction_cost_evaluator(monkeypatch)

    scenario = runner._transaction_cost_robustness_scenario(run_ctx, 2.0)

    assert scenario["metric_drop_pct"] is None
    assert scenario["is_complete"] is False
    assert scenario["status"] == "indeterminate"
    assert scenario["metric_drop_exceeded"] is False


def test_transaction_cost_robustness_scenario_uses_stable_evaluation_exception_reason(
    tmp_path, monkeypatch
):
    runner = _make_runner(tmp_path, monkeypatch, patch_source=False)
    _, _, _, _, run_ctx = _build_transaction_cost_validation_artifacts(
        runner,
        strategy_name="dummy",
        policy=_transaction_cost_robustness_config(mode="analytics"),
    )
    index = run_ctx.context.validated_data.raw_df.index
    entries = np.zeros(len(index), dtype=int)
    exits = np.zeros(len(index), dtype=int)
    run_ctx.aligned_signals = (
        pd.Series(entries, index=index),
        pd.Series(exits, index=index),
    )

    def _raise_eval_error(*_args, **_kwargs):
        raise RuntimeError("transient eval failure")

    monkeypatch.setattr(runner, "_evaluate_strategy_outcome", _raise_eval_error)

    scenario = runner._transaction_cost_robustness_scenario(run_ctx, 2.0)

    assert scenario["is_complete"] is False
    assert scenario["status"] == "indeterminate"
    assert scenario["reason"] == "evaluation_exception"
    assert scenario["exception_type"] == "RuntimeError"
    assert scenario["exception_message"] == "transient eval failure"


@pytest.mark.parametrize(
    ("breakeven_cfg", "policy", "expected_status"),
    [
        (
            ResultConsistencyTransactionCostBreakevenConfig(
                enabled=True,
                min_multiplier=1.0,
                max_multiplier=5.0,
                max_iterations=8,
                tolerance=0.01,
            ),
            _transaction_cost_robustness_config(mode="analytics", max_metric_drop_pct=0.1),
            "found",
        ),
        (
            ResultConsistencyTransactionCostBreakevenConfig(
                enabled=True,
                min_multiplier=3.0,
                max_multiplier=5.0,
                max_iterations=8,
                tolerance=0.01,
            ),
            _transaction_cost_robustness_config(mode="analytics", max_metric_drop_pct=0.05),
            "below_range",
        ),
        (
            ResultConsistencyTransactionCostBreakevenConfig(
                enabled=True,
                min_multiplier=1.0,
                max_multiplier=2.0,
                max_iterations=8,
                tolerance=0.01,
            ),
            _transaction_cost_robustness_config(mode="analytics", max_metric_drop_pct=0.3),
            "above_range",
        ),
    ],
)
def test_transaction_cost_breakeven_statuses(
    tmp_path,
    monkeypatch,
    breakeven_cfg,
    policy,
    expected_status,
):
    runner = _make_runner(tmp_path, monkeypatch, patch_source=False)
    _, _, _, _, run_ctx = _build_transaction_cost_validation_artifacts(
        runner,
        strategy_name="dummy",
        policy=_transaction_cost_robustness_config(
            mode=policy.mode,
            stress_multipliers=policy.stress_multipliers,
            max_metric_drop_pct=policy.max_metric_drop_pct,
            breakeven=breakeven_cfg,
        ),
    )
    monkeypatch.setattr(
        runner,
        "_transaction_cost_robustness_scenario",
        MethodType(_fake_transaction_cost_scenario, runner),
    )

    meta = runner._transaction_cost_breakeven_result(run_ctx)

    assert meta is not None
    assert meta["status"] == expected_status
    assert meta["enabled"] is True


def test_transaction_cost_breakeven_binary_search_uses_strict_threshold_partition(
    tmp_path, monkeypatch
):
    runner = _make_runner(tmp_path, monkeypatch, patch_source=False)
    _, _, _, _, run_ctx = _build_transaction_cost_validation_artifacts(
        runner,
        strategy_name="dummy",
        policy=_transaction_cost_robustness_config(mode="analytics", max_metric_drop_pct=0.1),
    )
    epsilon = runner._TRANSACTION_COST_ROBUSTNESS_DROP_EPSILON
    threshold = float(run_ctx.policy.max_metric_drop_pct)
    midpoint_drop = threshold + (epsilon / 2.0)

    def _scenario_with_midpoint_boundary(self, _run_ctx, multiplier):
        if np.isclose(float(multiplier), 2.0):
            metric_drop_pct = midpoint_drop
        elif float(multiplier) < 2.0:
            metric_drop_pct = 0.05
        else:
            metric_drop_pct = 0.2
        return {
            "is_complete": True,
            "metric_drop_pct": metric_drop_pct,
        }

    monkeypatch.setattr(
        runner,
        "_transaction_cost_robustness_scenario",
        MethodType(_scenario_with_midpoint_boundary, runner),
    )
    min_result = runner._transaction_cost_robustness_scenario(run_ctx, 1.0)
    max_result = runner._transaction_cost_robustness_scenario(run_ctx, 3.0)
    meta = runner._transaction_cost_breakeven_binary_search(
        run_ctx,
        min_multiplier=1.0,
        max_multiplier=3.0,
        threshold=threshold,
        max_iterations=1,
        tolerance=0.0,
        min_result=min_result,
        max_result=max_result,
        base_meta=runner._transaction_cost_breakeven_base_meta(run_ctx),
    )

    assert meta["upper_multiplier"] == pytest.approx(2.0)
    assert meta["lower_multiplier"] == pytest.approx(1.0)
    assert meta["metric_drop_pct"] == pytest.approx(midpoint_drop)


def test_transaction_cost_breakeven_is_indeterminate_for_invalid_baseline_metric(
    tmp_path, monkeypatch
):
    runner = _make_runner(tmp_path, monkeypatch, patch_source=False)
    _, _, _, _, run_ctx = _build_transaction_cost_validation_artifacts(
        runner,
        strategy_name="dummy",
        policy=_transaction_cost_robustness_config(
            mode="analytics",
            breakeven=ResultConsistencyTransactionCostBreakevenConfig(
                enabled=True,
                min_multiplier=1.0,
                max_multiplier=5.0,
                max_iterations=8,
                tolerance=0.01,
            ),
        ),
        baseline_metric=0.0,
    )
    monkeypatch.setattr(
        runner,
        "_transaction_cost_robustness_scenario",
        MethodType(_fake_transaction_cost_scenario, runner),
    )

    meta = runner._transaction_cost_breakeven_result(run_ctx)

    assert meta is not None
    assert meta["status"] == "indeterminate"
    assert meta["reason"] == "invalid_baseline_metric"


def test_run_all_transaction_cost_robustness_attaches_post_run_meta(
    tmp_path, monkeypatch
):
    runner = _setup_transaction_cost_run_all_runner(
        tmp_path,
        monkeypatch,
        mode="analytics",
        max_metric_drop_pct=0.3,
    )

    results = runner.run_all()

    assert len(results) == 1
    post_run_meta = results[0].stats.get("post_run_meta")
    assert post_run_meta is not None
    tc_meta = post_run_meta["transaction_cost_robustness"]
    assert tc_meta["status"] == "complete"
    assert tc_meta["stress_scenarios"][0]["multiplier"] == pytest.approx(2.0)
    assert all("post_run_meta" not in saved["stats"] for saved in runner.results_cache.saved)


@pytest.mark.parametrize(
    ("max_metric_drop_pct", "expected_failure_substring"),
    [
        (0.15, "transaction_cost_robustness_metric_drop_exceeded"),
        (0.3, "transaction_cost_robustness_negative_profit"),
    ],
)
def test_run_all_transaction_cost_robustness_rejects_in_enforce_mode(
    tmp_path,
    monkeypatch,
    max_metric_drop_pct,
    expected_failure_substring,
):
    runner = _setup_transaction_cost_run_all_runner(
        tmp_path,
        monkeypatch,
        mode="enforce",
        max_metric_drop_pct=max_metric_drop_pct,
    )

    results = runner.run_all()

    assert results == []
    assert any(
        failure["stage"] == "strategy_validation"
        and expected_failure_substring in failure["error"]
        for failure in runner.failures
    )


def test_strategy_validation_transaction_cost_robustness_enforce_rejects_non_dict_best_params(
    tmp_path, monkeypatch
):
    runner = _make_runner(tmp_path, monkeypatch, patch_source=False)
    _configure_result_consistency_runner(
        runner,
        strategy_name="dummy",
        strategy_cls=_DummyStrategy,
        result_consistency=_result_consistency_config(
            transaction_cost_robustness=_transaction_cost_robustness_config(mode="enforce")
        ),
    )
    outcome = StrategyEvalOutcome(
        best_val=0.85,
        best_params=None,
        best_stats={"profit": 0.15, "trades": 2},
        has_valid_candidate=True,
        evaluations=1,
        skipped_reason=None,
        strategy="dummy",
        job=JobContext(
            collection=runner.cfg.collections[0],
            symbol="AAPL",
            timeframe="1d",
            source="custom",
        ),
    )
    _, _, _, context, _ = _build_strategy_validation_artifacts(
        runner,
        strategy_name="dummy",
        prepared_data=_make_prepared_data(),
        outcome=outcome,
    )

    decision = runner._strategy_validate_results_common(context)

    assert decision.passed is False
    assert decision.action == "reject_result"
    assert "transaction_cost_robustness_indeterminate" in decision.reasons
    post_run_meta = outcome.best_stats.get("post_run_meta")
    assert post_run_meta is not None
    assert post_run_meta["transaction_cost_robustness"] == {
        "status": "indeterminate",
        "reason": "missing_transaction_cost_robustness_params",
        "best_params_type": "NoneType",
    }


def test_run_all_reliability_not_verified_skips_job(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.validation = ValidationConfig(
        data_quality=ValidationDataQualityConfig(
            is_verified=False,
            on_fail="skip_job",
        ),
    )
    _patch_source_with_bars(monkeypatch, bars=5)
    eval_calls = _patch_pybroker_simulation(monkeypatch)

    results = runner.run_all()
    assert results == []
    assert eval_calls["count"] == 0
    assert runner.failures
    failure = runner.failures[0]
    assert failure["stage"] == "data_validation"
    assert "collection_not_verified" in failure["error"]


def test_run_all_outlier_indeterminate_on_mad_zero(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    runner.cfg.validation = ValidationConfig(
        data_quality=ValidationDataQualityConfig(
            outlier_detection=ValidationOutlierDetectionConfig(
                max_outlier_pct=1.0,
                method="modified_zscore",
                zscore_threshold=3.5,
            ),
            on_fail="skip_job",
        ),
    )

    class _FlatWithSingleSpikeSource:
        def fetch(self, symbol, timeframe, only_cached=False):
            idx = pd.date_range("2024-01-01", periods=50, freq="D")
            closes = [100.0] * 25 + [300.0] + [100.0] * 24
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

    monkeypatch.setattr(
        BacktestRunner, "_make_source", lambda self, col: _FlatWithSingleSpikeSource()
    )
    eval_calls = _patch_pybroker_simulation(monkeypatch)

    results = runner.run_all()
    assert results == []
    assert eval_calls["count"] == 0
    assert runner.failures
    failure = runner.failures[0]
    assert failure["stage"] == "data_validation"
    assert "outlier_check_indeterminate(method=modified_zscore, reason=mad_zero)" in failure["error"]


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
    assert runner.metrics["symbols_tested"] == 2


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
        data_quality=ValidationDataQualityConfig(
            min_data_points=10,
            on_fail="skip_job",
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
        data_quality=ValidationDataQualityConfig(
            min_data_points=10,
            on_fail="skip_optimization",
        ),
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
        data_quality=ValidationDataQualityConfig(
            min_data_points=10,
            on_fail="skip_optimization",
        ),
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
        return returns, equity, stats, pd.DataFrame()

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
            data_quality=ValidationDataQualityConfig(
                min_data_points=10,
                on_fail="skip_optimization",
            ),
        ),
    )
    runner = _make_runner(tmp_path, monkeypatch, collections=[collection])
    runner.cfg.validation = ValidationConfig(
        data_quality=ValidationDataQualityConfig(
            min_data_points=10,
            on_fail="skip_job",
        ),
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
        data_quality=ValidationDataQualityConfig(
            min_data_points=10,
            on_fail="skip_collection",
        ),
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


def test_run_all_outlier_skip_collection_blocks_remaining_jobs_in_collection(
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
        data_quality=ValidationDataQualityConfig(
            outlier_detection=ValidationOutlierDetectionConfig(
                max_outlier_pct=1.0,
                method="modified_zscore",
                zscore_threshold=3.5,
            ),
            on_fail="skip_collection",
        ),
    )

    fetch_calls = {"bad_col": 0, "good_col": 0}

    class _Source:
        def __init__(self, collection_name: str):
            self.collection_name = collection_name

        def fetch(self, symbol, timeframe, only_cached=False):
            fetch_calls[self.collection_name] += 1
            if self.collection_name == "bad_col":
                idx = pd.date_range("2024-01-01", periods=52, freq="D")
                returns = [0.01, -0.008, 0.012, -0.009] * 12 + [0.70, 0.01, -0.008]
                closes = [100.0]
                for ret in returns:
                    closes.append(closes[-1] * (1.0 + ret))
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
            idx = pd.date_range("2024-01-01", periods=53, freq="D")
            returns = [0.01, -0.008, 0.012, -0.009] * 13
            closes = [100.0]
            for ret in returns:
                closes.append(closes[-1] * (1.0 + ret))
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
    assert "max_outlier_pct_exceeded" in failure["error"]


def test_run_all_not_verified_skip_collection_blocks_remaining_jobs_in_collection(
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
            validation=ValidationConfig(
                data_quality=ValidationDataQualityConfig(
                    is_verified=True,
                    on_fail="skip_collection",
                ),
            ),
        ),
    ]
    runner = _make_runner(tmp_path, monkeypatch, collections=collections)
    runner.cfg.validation = ValidationConfig(
        data_quality=ValidationDataQualityConfig(
            is_verified=False,
            on_fail="skip_collection",
        ),
    )

    fetch_calls = {"bad_col": 0, "good_col": 0}

    class _Source:
        def __init__(self, collection_name: str):
            self.collection_name = collection_name

        def fetch(self, symbol, timeframe, only_cached=False):
            fetch_calls[self.collection_name] += 1
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
    assert "collection_not_verified" in failure["error"]


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


def test_run_all_rejects_result_on_outlier_dependency(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch, patch_sim=False)
    runner.cfg.validation = ValidationConfig(
        result_consistency=_result_consistency_config(
            outlier_dependency=ResultConsistencyOutlierDependencyConfig(
                slices=5,
                profit_share_threshold=0.80,
                trade_share_threshold=0.05,
            ),
        ),
    )
    resolve_validation_overrides(runner.cfg)

    def _sim_with_concentrated_pnl(self, *args, **kwargs):
        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        returns = pd.Series([0.001] * 120, index=dates)
        equity = (1 + returns).cumprod()
        trades_log = [
            {"pnl": 1.0, "exit_date": f"2024-01-{(i % 28) + 1:02d}"} for i in range(99)
        ] + [{"pnl": 500.0, "exit_date": "2024-03-15"}]
        trades_frame = pd.DataFrame(trades_log)
        stats = {
            "sharpe": 1.0,
            "sortino": 1.0,
            "omega": 1.0,
            "tail_ratio": 1.0,
            "profit": 1.0,
            "pain_index": 0.0,
            "trades": 100,
            "max_drawdown": -0.1,
            "cagr": 0.1,
            "calmar": 1.0,
            "equity_curve": [],
            "drawdown_curve": [],
            "trades_log": [],
        }
        return returns, equity, stats, trades_frame

    monkeypatch.setattr(BacktestRunner, "_run_pybroker_simulation", _sim_with_concentrated_pnl)
    results = runner.run_all()

    assert results == []
    assert any(
        failure["stage"] == "strategy_validation"
        and "outlier_dependency_exceeded" in failure["error"]
        for failure in runner.failures
    )


def test_trade_meta_slice_profit_share_includes_earliest_exit_timestamp(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    evaluator = runner._get_evaluator()
    request = EvaluationRequest(
        collection="demo",
        symbol="AAPL",
        timeframe="1d",
        source="custom",
        strategy="dummy",
        params={},
        metric_name="sharpe",
        data_fingerprint="fingerprint",
        fees=0.0,
        slippage=0.0,
        bars_per_year=252,
        mode_config=EvaluationModeConfig(mode="backtest", payload={}),
        result_consistency_outlier_dependency_slices=2,
        result_consistency_outlier_dependency_profit_share_threshold=None,
    )
    trades_frame = pd.DataFrame(
        [
            {"pnl": 100.0, "exit_date": "2024-01-01"},
            {"pnl": 100.0, "exit_date": "2024-01-02"},
            {"pnl": 200.0, "exit_date": "2024-01-03"},
        ]
    )

    data_frame = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.0],
            "high": [11.0, 11.0, 11.0],
            "low": [9.0, 9.0, 9.0],
            "close": [10.0, 10.0, 10.0],
            "volume": [100.0, 100.0, 100.0],
        }
    )
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    trade_meta = evaluator._build_trade_meta(
        trades_frame,
        data_frame,
        dates,
        3,
        request,
    )

    assert trade_meta["outlier_dependency"]["max_slice_profit_share"] == pytest.approx(0.5)


def test_trade_meta_truncated_trades_reports_observed_and_expected_counts(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    evaluator = runner._get_evaluator()
    request = EvaluationRequest(
        collection="demo",
        symbol="AAPL",
        timeframe="1d",
        source="custom",
        strategy="dummy",
        params={},
        metric_name="sharpe",
        data_fingerprint="fingerprint",
        fees=0.0,
        slippage=0.0,
        bars_per_year=252,
        mode_config=EvaluationModeConfig(mode="backtest", payload={}),
        result_consistency_outlier_dependency_slices=2,
        result_consistency_outlier_dependency_profit_share_threshold=0.8,
    )
    trades_frame = pd.DataFrame(
        [
            {"pnl": 1.0, "exit_date": "2024-01-01"},
            {"pnl": -1.0, "exit_date": "2024-01-02"},
        ]
    )
    data_frame = pd.DataFrame(
        {
            "open": [10.0, 10.0],
            "high": [11.0, 11.0],
            "low": [9.0, 9.0],
            "close": [10.0, 10.0],
            "volume": [100.0, 100.0],
        }
    )
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])

    trade_meta = evaluator._build_trade_meta(
        trades_frame,
        data_frame,
        dates,
        5,
        request,
    )
    outlier_meta = trade_meta["outlier_dependency"]
    assert outlier_meta["is_complete"] is False
    assert outlier_meta["reason"] == "truncated_trades_frame"
    assert outlier_meta["analyzed_trades_count"] == 2
    assert outlier_meta["total_trades"] == 2
    assert outlier_meta["expected_trades"] == 5


def test_compute_dominant_trade_share_uses_all_trades_denominator(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    evaluator = runner._get_evaluator()
    pnls = pd.Series([10.0] * 10 + [-1.0] * 90).to_numpy()

    dominant_trade_count, dominant_trade_share = evaluator._compute_dominant_trade_share(pnls, 0.80)

    assert dominant_trade_count == 8
    assert dominant_trade_share == pytest.approx(0.08)


def test_collect_reliability_reasons_dispatches_outlier_reason_to_subclass():
    class _DispatchRunner(BacktestRunner):
        @classmethod
        def _outlier_pct_reason(
            cls,
            *,
            raw_df: pd.DataFrame,
            outlier_detection: ValidationOutlierDetectionConfig | None,
        ) -> str | None:
            return "subclass_outlier_reason"

    reasons = _DispatchRunner._collect_reliability_reasons(
        raw_df=pd.DataFrame({"Close": [1.0, 2.0, 3.0]}),
        continuity={},
        min_data_points_cfg=None,
        continuity_cfg=None,
        kurtosis_cfg=None,
        outlier_detection=None,
        stationarity_cfg=None,
        is_verified=None,
    )

    assert reasons == ["subclass_outlier_reason"]


def test_run_all_result_consistency_skips_check_for_missing_trades_frame(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch, patch_sim=False)
    runner.cfg.validation = ValidationConfig(
        result_consistency=_result_consistency_config(
            outlier_dependency=ResultConsistencyOutlierDependencyConfig(
                slices=5,
                profit_share_threshold=0.80,
                trade_share_threshold=0.05,
            ),
            execution_price_variance=ResultConsistencyExecutionPriceVarianceConfig(
                price_tolerance_bps=1.0
            ),
        ),
    )
    resolve_validation_overrides(runner.cfg)

    def _sim_with_missing_trades_frame(self, *args, **kwargs):
        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        returns = pd.Series([0.001] * 120, index=dates)
        equity = (1 + returns).cumprod()
        stats = {
            "sharpe": 1.0,
            "sortino": 1.0,
            "omega": 1.0,
            "tail_ratio": 1.0,
            "profit": 1.0,
            "pain_index": 0.0,
            "trades": 100,
            "max_drawdown": -0.1,
            "cagr": 0.1,
            "calmar": 1.0,
            "equity_curve": [],
            "drawdown_curve": [],
            "trades_log": [],
        }
        return returns, equity, stats, pd.DataFrame()

    monkeypatch.setattr(BacktestRunner, "_run_pybroker_simulation", _sim_with_missing_trades_frame)
    results = runner.run_all()

    assert len(results) >= 1
    assert not any("outlier_dependency_exceeded" in failure["error"] for failure in runner.failures)
    assert not any("execution_price_variance_exceeded" in failure["error"] for failure in runner.failures)


def test_run_all_rejects_result_on_execution_price_variance(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch, patch_sim=False)
    runner.cfg.validation = ValidationConfig(
        result_consistency=_result_consistency_config(
            execution_price_variance=ResultConsistencyExecutionPriceVarianceConfig(
                price_tolerance_bps=0.0
            )
        ),
    )
    resolve_validation_overrides(runner.cfg)

    def _sim_with_impossible_fill(self, *args, **kwargs):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        returns = pd.Series([0.01, -0.005, 0.002], index=dates)
        equity = (1 + returns).cumprod()
        trades_frame = pd.DataFrame(
            [
                {
                    "entry_price": 120.0,
                    "entry_date": "2024-01-01",
                    "exit_price": 90.0,
                    "exit_date": "2024-01-02",
                    "pnl": -1.0,
                }
            ]
        )
        stats = {
            "sharpe": 1.0,
            "sortino": 1.0,
            "omega": 1.0,
            "tail_ratio": 1.0,
            "profit": 1.0,
            "pain_index": 0.0,
            "trades": 1,
            "max_drawdown": -0.1,
            "cagr": 0.1,
            "calmar": 1.0,
            "equity_curve": [],
            "drawdown_curve": [],
            "trades_log": [],
        }
        return returns, equity, stats, trades_frame

    monkeypatch.setattr(BacktestRunner, "_run_pybroker_simulation", _sim_with_impossible_fill)
    results = runner.run_all()

    assert results == []
    assert any(
        failure["stage"] == "strategy_validation"
        and "execution_price_variance_exceeded" in failure["error"]
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
