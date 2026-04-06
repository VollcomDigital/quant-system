from __future__ import annotations

from pathlib import Path

import pytest

from src.config import (
    CollectionConfig,
    Config,
    ResultConsistencyConfig,
    ValidationConfig,
    ValidationDataQualityConfig,
    ValidationLookaheadShuffleTestConfig,
    ValidationOutlierDetectionConfig,
    ValidationStationarityConfig,
    ValidationStationarityRegimeShiftConfig,
    load_config,
    resolve_validation_overrides,
)


def test_load_config_allows_missing_strategies(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.strategies == []
    assert cfg.evaluation_mode == "backtest"


def test_load_config_accepts_evaluation_mode(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
evaluation_mode: walk_forward
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.evaluation_mode == "walk_forward"


def test_load_config_rejects_invalid_evaluation_mode(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
evaluation_mode: invalid_mode
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_reliability_thresholds(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    min_data_points: 500
    continuity:
      min_score: 0.98
    on_fail: skip_job
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.validation is not None
    assert cfg.validation.data_quality is not None
    assert cfg.validation.data_quality.min_data_points == 500
    assert cfg.validation.data_quality.continuity is not None
    assert cfg.validation.data_quality.continuity.min_score == pytest.approx(0.98)
    assert cfg.validation.data_quality.on_fail == "skip_job"


def test_load_config_reliability_thresholds_accepts_is_verified(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    is_verified: false
    on_fail: skip_job
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.validation is not None
    assert cfg.validation.data_quality is not None
    assert cfg.validation.data_quality.is_verified is False


def test_load_config_lookahead_shuffle_test_defaults(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  result_consistency:
    lookahead_shuffle_test:
      permutations: 100
      pvalue_max: 0.05
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.validation is not None
    assert cfg.validation.result_consistency is not None
    assert cfg.validation.result_consistency.lookahead_shuffle_test is not None
    assert cfg.validation.result_consistency.lookahead_shuffle_test.permutations == 100
    assert cfg.validation.result_consistency.lookahead_shuffle_test.pvalue_max == pytest.approx(0.05)
    assert cfg.validation.result_consistency.lookahead_shuffle_test.seed is None
    assert cfg.validation.result_consistency.lookahead_shuffle_test.max_failed_permutations is None
    assert cfg.collections[0].validation is not None
    assert cfg.collections[0].validation.result_consistency is not None
    assert cfg.collections[0].validation.result_consistency.lookahead_shuffle_test is not None
    assert (
        cfg.collections[0].validation.result_consistency.lookahead_shuffle_test.permutations == 100
    )
    assert cfg.collections[0].validation.result_consistency.lookahead_shuffle_test.pvalue_max == pytest.approx(
        0.05
    )
    assert cfg.collections[0].validation.result_consistency.lookahead_shuffle_test.seed == 1337
    assert (
        cfg.collections[
            0
        ].validation.result_consistency.lookahead_shuffle_test.max_failed_permutations
        is None
    )


def test_load_config_collection_reliability_thresholds_override(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
    validation:
      data_quality:
        min_data_points: 250
        continuity:
          min_score: 0.95
        on_fail: skip_optimization
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    min_data_points: 500
    continuity:
      min_score: 0.98
    on_fail: skip_job
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.validation is not None
    assert len(cfg.collections) == 1
    col = cfg.collections[0]
    assert col.validation is not None
    assert col.validation.data_quality is not None
    assert col.validation.data_quality.min_data_points == 250
    assert col.validation.data_quality.continuity is not None
    assert col.validation.data_quality.continuity.min_score == pytest.approx(0.95)
    assert col.validation.data_quality.on_fail == "skip_optimization"


def test_load_config_collection_reliability_thresholds_override_is_verified(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
    validation:
      data_quality:
        is_verified: true
        on_fail: skip_optimization
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    is_verified: false
    on_fail: skip_job
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.validation is not None
    assert cfg.validation.data_quality is not None
    assert cfg.validation.data_quality.is_verified is False
    col = cfg.collections[0]
    assert col.validation is not None
    assert col.validation.data_quality is not None
    assert col.validation.data_quality.is_verified is True
    assert col.validation.data_quality.on_fail == "skip_optimization"


def test_load_config_lookahead_shuffle_test_legacy_data_quality_location_is_ignored(
    tmp_path: Path,
):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    lookahead_shuffle_test:
      permutations: 100
      pvalue_max: 0.25
      seed: 11
    on_fail: skip_job
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.validation is not None
    assert cfg.validation.data_quality is not None
    assert cfg.validation.data_quality.on_fail == "skip_job"
    assert cfg.validation.result_consistency is None


@pytest.mark.parametrize(
    "on_fail_input",
    [
        "skip_job",
        "skip_optimization",
        "skip_collection",
    ],
)
def test_load_config_reliability_thresholds_on_fail_values(tmp_path: Path, on_fail_input: str):
    config_text = f"""
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    on_fail: {on_fail_input}
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.validation is not None
    assert cfg.validation.data_quality is not None
    assert cfg.validation.data_quality.on_fail == on_fail_input


@pytest.mark.parametrize(
    "reliability_yaml",
    [
        "on_fail: abort_run",
    ],
    ids=["invalid_on_fail"],
)
def test_load_config_reliability_thresholds_invalid_values(tmp_path: Path, reliability_yaml: str):
    config_text = f"""
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    {reliability_yaml}
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_reliability_thresholds_invalid_continuity_score(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    on_fail: skip_job
    continuity:
      min_score: 1.2
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError, match=r"validation\.data_quality\.continuity\.min_score"):
        load_config(path)


def test_load_config_collection_reliability_thresholds_invalid_values(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
    validation:
      data_quality:
        on_fail: abort_run
timeframes: ['1d']
metric: sharpe
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_optimization_policy(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  optimization:
    on_fail: skip_job
    min_bars: 123
    dof_multiplier: 7
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.validation is not None
    assert cfg.validation.optimization is not None
    assert cfg.validation.optimization.on_fail == "skip_job"
    assert cfg.validation.optimization.min_bars == 123
    assert cfg.validation.optimization.dof_multiplier == 7
    assert cfg.validation.optimization.runtime_error_max_per_tuple is None
    assert cfg.collections[0].validation is not None
    assert cfg.collections[0].validation.optimization is not None
    assert cfg.collections[0].validation.optimization.runtime_error_max_per_tuple == 1


def test_load_config_optimization_policy_accepts_runtime_error_threshold(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  optimization:
    on_fail: skip_job
    min_bars: 123
    dof_multiplier: 7
    runtime_error_max_per_tuple: 4
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.validation is not None
    assert cfg.validation.optimization is not None
    assert cfg.validation.optimization.runtime_error_max_per_tuple == 4


def test_load_config_optimization_policy_invalid_on_fail(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  optimization:
    on_fail: skip_optimization
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_optimization_policy_invalid_runtime_error_threshold(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  optimization:
    on_fail: skip_job
    min_bars: 123
    dof_multiplier: 7
    runtime_error_max_per_tuple: 0
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError, match=r"validation\.optimization\.runtime_error_max_per_tuple"):
        load_config(path)


def test_load_config_optimization_policy_requires_all_fields(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  optimization:
    on_fail: baseline_only
    min_bars: 100
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_optimization_policy_inherited_to_collections(tmp_path: Path):
    config_text = """
collections:
  - name: test_a
    source: yfinance
    symbols: ['AAPL']
  - name: test_b
    source: yfinance
    symbols: ['MSFT']
timeframes: ['1d']
metric: sharpe
validation:
  optimization:
    on_fail: baseline_only
    min_bars: 321
    dof_multiplier: 11
    runtime_error_max_per_tuple: 2
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    for col in cfg.collections:
        assert col.validation is not None
        assert col.validation.optimization is not None
        assert col.validation.optimization.on_fail == "baseline_only"
        assert col.validation.optimization.min_bars == 321
        assert col.validation.optimization.dof_multiplier == 11
        assert col.validation.optimization.runtime_error_max_per_tuple == 2


def test_load_config_optimization_policy_collection_override_merges_with_global(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
    validation:
      optimization:
        on_fail: baseline_only
        min_bars: 200
        dof_multiplier: 9
        runtime_error_max_per_tuple: 3
timeframes: ['1d']
metric: sharpe
validation:
  optimization:
    on_fail: skip_job
    min_bars: 123
    dof_multiplier: 7
    runtime_error_max_per_tuple: 2
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    col = cfg.collections[0]
    assert col.validation is not None
    assert col.validation.optimization is not None
    # Collection-level optimization policy takes precedence over global values.
    assert col.validation.optimization.on_fail == "baseline_only"
    assert col.validation.optimization.min_bars == 200
    assert col.validation.optimization.dof_multiplier == 9
    assert col.validation.optimization.runtime_error_max_per_tuple == 3


def test_load_config_optimization_policy_collection_override_inherits_runtime_error_threshold(
    tmp_path: Path,
):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
    validation:
      optimization:
        on_fail: baseline_only
        min_bars: 200
        dof_multiplier: 9
timeframes: ['1d']
metric: sharpe
validation:
  optimization:
    on_fail: skip_job
    min_bars: 123
    dof_multiplier: 7
    runtime_error_max_per_tuple: 2
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    col = cfg.collections[0]
    assert col.validation is not None
    assert col.validation.optimization is not None
    assert col.validation.optimization.on_fail == "baseline_only"
    assert col.validation.optimization.min_bars == 200
    assert col.validation.optimization.dof_multiplier == 9
    assert col.validation.optimization.runtime_error_max_per_tuple == 2


def test_load_config_result_consistency_policy(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  result_consistency:
    outlier_dependency:
      slices: 6
      profit_share_threshold: 0.80
      trade_share_threshold: 0.05
    execution_price_variance:
      price_tolerance_bps: 1.0
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.validation is not None
    assert cfg.validation.result_consistency is not None
    assert cfg.validation.result_consistency.outlier_dependency is not None
    assert cfg.validation.result_consistency.execution_price_variance is not None
    assert cfg.validation.result_consistency.outlier_dependency.slices == 6
    assert cfg.validation.result_consistency.outlier_dependency.profit_share_threshold == pytest.approx(
        0.80
    )
    assert cfg.validation.result_consistency.outlier_dependency.trade_share_threshold == pytest.approx(
        0.05
    )
    assert (
        cfg.validation.result_consistency.execution_price_variance.price_tolerance_bps
        == pytest.approx(1.0)
    )
    col = cfg.collections[0]
    assert col.validation is not None
    assert col.validation.result_consistency is not None
    assert col.validation.result_consistency.outlier_dependency is not None
    assert col.validation.result_consistency.outlier_dependency.slices == 6
    assert col.validation.result_consistency.outlier_dependency.profit_share_threshold == pytest.approx(
        0.80
    )
    assert col.validation.result_consistency.outlier_dependency.trade_share_threshold == pytest.approx(
        0.05
    )


def test_load_config_result_consistency_collection_override(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
    validation:
      result_consistency:
        outlier_dependency:
          slices: 4
          profit_share_threshold: 0.75
          trade_share_threshold: 0.10
timeframes: ['1d']
metric: sharpe
validation:
  result_consistency:
    outlier_dependency:
      slices: 8
      profit_share_threshold: 0.80
      trade_share_threshold: 0.05
    execution_price_variance:
      price_tolerance_bps: 0.5
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    col = cfg.collections[0]
    assert col.validation is not None
    assert col.validation.result_consistency is not None
    assert col.validation.result_consistency.outlier_dependency is not None
    assert col.validation.result_consistency.outlier_dependency.slices == 4
    assert col.validation.result_consistency.outlier_dependency.profit_share_threshold == pytest.approx(
        0.75
    )
    assert col.validation.result_consistency.outlier_dependency.trade_share_threshold == pytest.approx(
        0.10
    )
    # Collection inherits global execution_price_variance when not overridden.
    assert col.validation.result_consistency.execution_price_variance is not None
    assert (
        col.validation.result_consistency.execution_price_variance.price_tolerance_bps
        == pytest.approx(0.5)
    )


def test_load_config_result_consistency_invalid_slices(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  result_consistency:
    outlier_dependency:
      slices: 1
      profit_share_threshold: 0.80
      trade_share_threshold: 0.05
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_result_consistency_requires_profit_share_threshold(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  result_consistency:
    outlier_dependency:
      slices: 5
      trade_share_threshold: 0.05
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_result_consistency_requires_trade_share_threshold(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  result_consistency:
    outlier_dependency:
      slices: 5
      profit_share_threshold: 0.80
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_result_consistency_threshold_out_of_range(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  result_consistency:
    outlier_dependency:
      slices: 5
      profit_share_threshold: 1.2
      trade_share_threshold: 0.05
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_result_consistency_execution_price_variance_requires_tolerance(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  result_consistency:
    execution_price_variance: {}
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_result_consistency_execution_price_variance_tolerance_non_negative(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  result_consistency:
    execution_price_variance:
      price_tolerance_bps: -0.1
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_data_quality_calendar(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    on_fail: skip_job
    continuity:
      calendar:
        kind: exchange
        exchange: XNYS
        timezone: UTC-05:00
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.validation is not None
    assert cfg.validation.data_quality is not None
    assert cfg.validation.data_quality.continuity is not None
    assert cfg.validation.data_quality.continuity.calendar is not None
    assert cfg.validation.data_quality.continuity.calendar.kind == "exchange"
    assert cfg.validation.data_quality.continuity.calendar.exchange == "XNYS"
    assert cfg.validation.data_quality.continuity.calendar.timezone == "UTC-05:00"


def test_load_config_data_quality_calendar_invalid_kind(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    on_fail: skip_job
    continuity:
      calendar:
        kind: invalid
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_data_quality_calendar_invalid_timezone(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    on_fail: skip_job
    continuity:
      calendar:
        kind: exchange
        timezone: America/New_York
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_data_quality_continuity_without_calendar_keeps_calendar_none(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    on_fail: skip_job
    continuity:
      min_score: 0.98
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.validation is not None
    assert cfg.validation.data_quality is not None
    assert cfg.validation.data_quality.continuity is not None
    assert cfg.validation.data_quality.continuity.calendar is None


def test_load_config_data_quality_calendar_defaults_are_applied_at_effective_stage(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    on_fail: skip_job
    continuity:
      calendar: {}
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.validation is not None
    assert cfg.validation.data_quality is not None
    assert cfg.validation.data_quality.continuity is not None
    assert cfg.validation.data_quality.continuity.calendar is not None
    assert cfg.validation.data_quality.continuity.calendar.kind is None
    assert cfg.collections[0].validation is not None
    assert cfg.collections[0].validation.data_quality is not None
    assert cfg.collections[0].validation.data_quality.continuity is not None
    assert cfg.collections[0].validation.data_quality.continuity.calendar is not None
    assert cfg.collections[0].validation.data_quality.continuity.calendar.kind == "auto"


def test_load_config_data_quality_outlier_settings(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    on_fail: skip_job
    outlier_detection:
      max_outlier_pct: 1.5
      method: modified_zscore
      zscore_threshold: 3.5
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.validation is not None
    assert cfg.validation.data_quality is not None
    assert cfg.validation.data_quality.outlier_detection is not None
    assert cfg.validation.data_quality.outlier_detection.max_outlier_pct == pytest.approx(1.5)
    assert cfg.validation.data_quality.outlier_detection.method == "modified_zscore"
    assert cfg.validation.data_quality.outlier_detection.zscore_threshold == pytest.approx(3.5)


def test_load_config_data_quality_stationarity_settings(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    on_fail: skip_job
    stationarity:
      adf_pvalue_max: 0.05
      kpss_pvalue_min: 0.05
      min_points: 40
      regime_shift:
        window: 20
        mean_shift_max: 1.5
        vol_ratio_max: 1.75
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.validation is not None
    assert cfg.validation.data_quality is not None
    assert cfg.validation.data_quality.stationarity is not None
    assert cfg.validation.data_quality.stationarity.adf_pvalue_max == pytest.approx(0.05)
    assert cfg.validation.data_quality.stationarity.kpss_pvalue_min == pytest.approx(0.05)
    assert cfg.validation.data_quality.stationarity.min_points == 40
    assert cfg.validation.data_quality.stationarity.regime_shift is not None
    assert cfg.validation.data_quality.stationarity.regime_shift.window == 20
    assert cfg.validation.data_quality.stationarity.regime_shift.mean_shift_max == pytest.approx(1.5)
    assert cfg.validation.data_quality.stationarity.regime_shift.vol_ratio_max == pytest.approx(1.75)


def test_load_config_data_quality_stationarity_defaults_min_points(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    on_fail: skip_job
    stationarity:
      adf_pvalue_max: 0.05
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.validation is not None
    assert cfg.validation.data_quality is not None
    assert cfg.validation.data_quality.stationarity is not None
    # Global parse preserves explicit user input shape; default is resolved on
    # effective collection-level policy during override resolution.
    assert cfg.validation.data_quality.stationarity.min_points is None
    assert cfg.collections[0].validation is not None
    assert cfg.collections[0].validation.data_quality is not None
    assert cfg.collections[0].validation.data_quality.stationarity is not None
    assert cfg.collections[0].validation.data_quality.stationarity.min_points == 30


def test_load_config_collection_data_quality_stationarity_override(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
    validation:
      data_quality:
        on_fail: skip_collection
        stationarity:
          adf_pvalue_max: 0.1
          regime_shift:
            window: 25
            mean_shift_max: 1.2
            vol_ratio_max: 1.5
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    on_fail: skip_job
    stationarity:
      adf_pvalue_max: 0.05
      min_points: 50
      regime_shift:
        window: 20
        mean_shift_max: 1.5
        vol_ratio_max: 2.0
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.validation is not None
    assert len(cfg.collections) == 1
    col = cfg.collections[0]
    assert col.validation is not None
    assert col.validation.data_quality is not None
    assert col.validation.data_quality.on_fail == "skip_collection"
    assert col.validation.data_quality.stationarity is not None
    assert col.validation.data_quality.stationarity.adf_pvalue_max == pytest.approx(0.1)
    assert col.validation.data_quality.stationarity.min_points == 50
    assert col.validation.data_quality.stationarity.regime_shift is not None
    assert col.validation.data_quality.stationarity.regime_shift.window == 25
    assert col.validation.data_quality.stationarity.regime_shift.mean_shift_max == pytest.approx(1.2)
    assert col.validation.data_quality.stationarity.regime_shift.vol_ratio_max == pytest.approx(1.5)


def test_load_config_collection_lookahead_shuffle_test_override(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
    validation:
      result_consistency:
        lookahead_shuffle_test:
          permutations: 100
          seed: 7
          max_failed_permutations: 2
timeframes: ['1d']
metric: sharpe
validation:
  result_consistency:
    lookahead_shuffle_test:
      permutations: 100
      pvalue_max: 0.10
      seed: 1337
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.collections[0].validation is not None
    assert cfg.collections[0].validation.result_consistency is not None
    lookahead = cfg.collections[0].validation.result_consistency.lookahead_shuffle_test
    assert lookahead is not None
    assert lookahead.permutations == 100
    assert lookahead.pvalue_max == pytest.approx(0.10)
    assert lookahead.seed == 7
    assert lookahead.max_failed_permutations == 2


def test_load_config_collection_lookahead_shuffle_test_partial_override_inherits_base(
    tmp_path: Path,
):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
    validation:
      result_consistency:
        lookahead_shuffle_test:
          permutations: 100
          pvalue_max: 0.25
timeframes: ['1d']
metric: sharpe
validation:
  result_consistency:
    lookahead_shuffle_test:
      permutations: 100
      pvalue_max: 0.5
      seed: 17
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.collections[0].validation is not None
    assert cfg.collections[0].validation.result_consistency is not None
    lookahead = cfg.collections[0].validation.result_consistency.lookahead_shuffle_test
    assert lookahead is not None
    assert lookahead.permutations == 100
    assert lookahead.pvalue_max == pytest.approx(0.25)
    assert lookahead.seed == 17


def test_load_config_data_quality_stationarity_invalid_values(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    on_fail: skip_job
    stationarity:
      adf_pvalue_max: 1.2
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_lookahead_shuffle_test_invalid_permutations(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  result_consistency:
    lookahead_shuffle_test:
      permutations: 99
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError, match=r"validation\.result_consistency\.lookahead_shuffle_test\.permutations"):
        load_config(path)


def test_load_config_lookahead_shuffle_test_invalid_max_failed_permutations(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  result_consistency:
    lookahead_shuffle_test:
      permutations: 100
      max_failed_permutations: 101
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(
        ValueError,
        match=r"validation\.result_consistency\.lookahead_shuffle_test\.max_failed_permutations",
    ):
        load_config(path)


def test_load_config_lookahead_shuffle_test_invalid_max_failed_permutations_default_permutations(
    tmp_path: Path,
):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  result_consistency:
    lookahead_shuffle_test:
      max_failed_permutations: 101
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(
        ValueError,
        match=r"validation\.result_consistency\.lookahead_shuffle_test\.max_failed_permutations",
    ):
        load_config(path)


def test_load_config_lookahead_shuffle_test_requires_pvalue_max(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  result_consistency:
    lookahead_shuffle_test:
      permutations: 100
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError, match=r"missing required field\(s\): pvalue_max"):
        load_config(path)


def test_load_config_lookahead_shuffle_test_rejects_threshold_key(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  result_consistency:
    lookahead_shuffle_test:
      permutations: 100
      pvalue_max: 0.05
      threshold: 0.2
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError, match=r"lookahead_shuffle_test\.threshold"):
        load_config(path)


def test_load_config_lookahead_shuffle_test_invalid_pvalue_max(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  result_consistency:
    lookahead_shuffle_test:
      permutations: 100
      pvalue_max: 1.2
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(
        ValueError,
        match=r"validation\.result_consistency\.lookahead_shuffle_test\.pvalue_max",
    ):
        load_config(path)


def test_load_config_data_quality_stationarity_invalid_kpss_pvalue(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    on_fail: skip_job
    stationarity:
      adf_pvalue_max: 0.05
      kpss_pvalue_min: 1.2
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_data_quality_stationarity_missing_required_field(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    on_fail: skip_job
    stationarity: {}
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)


def test_resolve_validation_overrides_revalidates_stationarity_mutations():
    stationarity = ValidationStationarityConfig(
        adf_pvalue_max=0.05,
        min_points=40,
        regime_shift=ValidationStationarityRegimeShiftConfig(
            window=20,
            mean_shift_max=1.5,
            vol_ratio_max=1.75,
        ),
    )
    cfg = Config(
        collections=[
            CollectionConfig(
                name="test",
                source="yfinance",
                symbols=["AAPL"],
            )
        ],
        timeframes=["1d"],
        metric="sharpe",
        strategies=[],
        validation=ValidationConfig(
            data_quality=ValidationDataQualityConfig(
                on_fail="skip_job",
                stationarity=stationarity,
            )
        ),
    )

    stationarity.min_points = None
    resolve_validation_overrides(cfg)
    assert cfg.collections[0].validation is not None
    assert cfg.collections[0].validation.data_quality is not None
    assert cfg.collections[0].validation.data_quality.stationarity is not None
    assert cfg.collections[0].validation.data_quality.stationarity.min_points == 30

    cfg.collections[0].validation.data_quality.stationarity.regime_shift = (
        ValidationStationarityRegimeShiftConfig(
            window=5,
            mean_shift_max=1.5,
            vol_ratio_max=1.75,
        )
    )
    with pytest.raises(ValueError):
        resolve_validation_overrides(cfg)


def test_resolve_validation_overrides_rejects_mutated_stationarity_missing_field():
    stationarity = ValidationStationarityConfig(
        adf_pvalue_max=0.05,
        min_points=40,
        regime_shift=ValidationStationarityRegimeShiftConfig(
            window=20,
            mean_shift_max=1.5,
            vol_ratio_max=1.75,
        ),
    )
    cfg = Config(
        collections=[
            CollectionConfig(
                name="test",
                source="yfinance",
                symbols=["AAPL"],
            )
        ],
        timeframes=["1d"],
        metric="sharpe",
        strategies=[],
        validation=ValidationConfig(
            data_quality=ValidationDataQualityConfig(
                on_fail="skip_job",
                stationarity=stationarity,
            )
        ),
    )

    stationarity.adf_pvalue_max = None
    with pytest.raises(ValueError, match="adf_pvalue_max"):
        resolve_validation_overrides(cfg)


def test_load_config_data_quality_outlier_collection_requires_method_and_threshold(
    tmp_path: Path,
):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
    validation:
      data_quality:
        on_fail: skip_job
        outlier_detection:
          max_outlier_pct: 2.0
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    on_fail: skip_job
    outlier_detection:
      max_outlier_pct: 1.5
      method: zscore
      zscore_threshold: 3.0
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_data_quality_invalid_outlier_method(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    on_fail: skip_job
    outlier_detection:
      max_outlier_pct: 2.0
      method: invalid
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_data_quality_outlier_missing_threshold(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    on_fail: skip_job
    outlier_detection:
      max_outlier_pct: 2.0
      method: zscore
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)


def test_resolve_validation_overrides_normalizes_programmatic_outlier_method(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    on_fail: skip_job
    outlier_detection:
      max_outlier_pct: 2.0
      method: zscore
      zscore_threshold: 3.0
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.collections[0].validation is not None
    assert cfg.collections[0].validation.data_quality is not None
    assert cfg.collections[0].validation.data_quality.outlier_detection is not None
    cfg.collections[0].validation.data_quality.outlier_detection = ValidationOutlierDetectionConfig(
        max_outlier_pct=2.0,
        method="Modified_Zscore",
        zscore_threshold=3.5,
    )

    resolve_validation_overrides(cfg)
    normalized = cfg.collections[0].validation.data_quality.outlier_detection
    assert normalized is not None
    assert normalized.method == "modified_zscore"


def test_resolve_validation_overrides_rejects_programmatic_outlier_percent_out_of_range(
    tmp_path: Path,
):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    on_fail: skip_job
    outlier_detection:
      max_outlier_pct: 2.0
      method: zscore
      zscore_threshold: 3.0
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.collections[0].validation is not None
    assert cfg.collections[0].validation.data_quality is not None
    assert cfg.collections[0].validation.data_quality.outlier_detection is not None
    cfg.collections[0].validation.data_quality.outlier_detection = ValidationOutlierDetectionConfig(
        max_outlier_pct=200.0,
        method="zscore",
        zscore_threshold=3.0,
    )

    with pytest.raises(ValueError, match="max_outlier_pct"):
        resolve_validation_overrides(cfg)
