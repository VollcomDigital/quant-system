from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import pytest
import yaml

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
    parse_optional_float,
    parse_optional_int,
    parse_optional_float_list,
    resolve_validation_overrides,
)


def _load_from_blocks(
    tmp_path: Path,
    *,
    validation_block: dict | None = None,
    collection_validation_block: dict | None = None,
):
    raw = {
        "collections": [
            {
                "name": "test",
                "source": "yfinance",
                "symbols": ["AAPL"],
            }
        ],
        "timeframes": ["1d"],
        "metric": "sharpe",
    }
    if validation_block is not None:
        raw["validation"] = deepcopy(validation_block)
    if collection_validation_block is not None:
        raw["collections"][0]["validation"] = deepcopy(collection_validation_block)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return load_config(path)


def _result_consistency_block(**overrides):
    block = {"min_metric": 0.5, "min_trades": 20}
    block.update(overrides)
    return block


def _transaction_cost_robustness_block(**overrides):
    block = {
        "mode": "analytics",
        "stress_multipliers": [2.0, 5.0],
        "max_metric_drop_pct": 0.3,
    }
    block.update(overrides)
    return block


def test_parse_optional_float_rejects_non_finite_values():
    with pytest.raises(ValueError, match=r"must be finite"):
        parse_optional_float({"threshold": math.nan}, "validation.result_consistency", "threshold")

    with pytest.raises(ValueError, match=r"must be finite"):
        parse_optional_float({"threshold": math.inf}, "validation.result_consistency", "threshold")


def test_parse_optional_int_rejects_boolean_values():
    with pytest.raises(ValueError, match=r"expected an integer"):
        parse_optional_int({"permutations": True}, "validation.result_consistency", "permutations")


def test_parse_optional_int_rejects_numeric_strings_and_floats():
    with pytest.raises(ValueError, match=r"expected an integer"):
        parse_optional_int({"permutations": "100"}, "validation.result_consistency", "permutations")

    with pytest.raises(ValueError, match=r"expected an integer"):
        parse_optional_int({"permutations": 100.0}, "validation.result_consistency", "permutations")


def test_parse_optional_float_rejects_numeric_strings():
    with pytest.raises(ValueError, match=r"expected a number"):
        parse_optional_float({"threshold": "0.25"}, "validation.result_consistency", "threshold")


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


def test_load_config_rejects_non_mapping_root(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("- not-a-mapping\n")

    with pytest.raises(ValueError, match=r"Invalid `config`: expected a mapping"):
        load_config(path)


def test_load_config_rejects_missing_required_root_keys(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("metric: sharpe\n")

    with pytest.raises(
        ValueError,
        match=r"Invalid `config`: missing required key\(s\): `collections`, `timeframes`",
    ):
        load_config(path)


def test_load_config_rejects_non_list_timeframes(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: 1d
metric: sharpe
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError, match=r"Invalid `timeframes`: expected a list"):
        load_config(path)


def test_load_config_rejects_non_mapping_strategy_item(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
strategies:
  - invalid
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError, match=r"Invalid `strategies\[0\]`: expected a mapping"):
        load_config(path)


def test_load_config_rejects_strategy_missing_name(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
strategies:
  - params: {}
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError, match=r"Invalid `strategies\[0\]`: missing required key\(s\): `name`"):
        load_config(path)


def test_load_config_rejects_non_mapping_notifications(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
notifications: true
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError, match=r"Invalid `notifications`: expected a mapping"):
        load_config(path)


def test_load_config_rejects_non_mapping_notifications_slack(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
notifications:
  slack: true
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError, match=r"Invalid `notifications.slack`: expected a mapping"):
        load_config(path)


def test_load_config_rejects_non_finite_top_level_numeric_values(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
fees: .inf
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError, match=r"`fees` must be finite"):
        load_config(path)


def test_load_config_rejects_non_list_collections(tmp_path: Path):
    config_text = """
collections:
  name: test
  source: yfinance
  symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError, match=r"expected a list at `collections`"):
        load_config(path)


def test_load_config_rejects_non_mapping_collection_item(tmp_path: Path):
    config_text = """
collections:
  - invalid
timeframes: ['1d']
metric: sharpe
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError, match=r"expected a mapping at `collections\[0\]`"):
        load_config(path)


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


def test_load_config_reliability_thresholds_rejects_string_is_verified(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  data_quality:
    is_verified: "false"
    on_fail: skip_job
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError, match=r"validation\.data_quality\.is_verified"):
        load_config(path)


def test_load_config_lookahead_shuffle_test_defaults(tmp_path: Path):
    cfg = _load_from_blocks(
        tmp_path,
        validation_block={
            "result_consistency": _result_consistency_block(
                lookahead_shuffle_test={"permutations": 100, "pvalue_max": 0.05}
            )
        },
    )
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


def test_load_config_reference_source_enables_data_integrity_audit_defaults(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    reference_source: alphavantage
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.collections[0].reference_source == "alphavantage"
    assert cfg.collections[0].validation is not None
    assert cfg.collections[0].validation.result_consistency is not None
    audit = cfg.collections[0].validation.result_consistency.data_integrity_audit
    assert audit is not None
    assert audit.min_overlap_ratio == pytest.approx(0.99)
    assert audit.max_median_ohlc_diff_bps == pytest.approx(5.0)
    assert audit.max_p95_ohlc_diff_bps == pytest.approx(20.0)


def test_load_config_data_integrity_audit_rejects_non_mapping(tmp_path: Path):
    with pytest.raises(
        ValueError,
        match=r"validation\.result_consistency\.data_integrity_audit",
    ):
        _load_from_blocks(
            tmp_path,
            validation_block={
                "result_consistency": _result_consistency_block(data_integrity_audit=True)
            },
        )


def test_load_config_data_integrity_audit_collection_override_inherits_global(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    reference_source: alphavantage
    symbols: ['AAPL']
    validation:
      result_consistency:
        data_integrity_audit:
          min_overlap_ratio: 0.97
timeframes: ['1d']
metric: sharpe
validation:
  result_consistency:
    data_integrity_audit:
      max_median_ohlc_diff_bps: 2.0
      max_p95_ohlc_diff_bps: 10.0
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.collections[0].validation is not None
    assert cfg.collections[0].validation.result_consistency is not None
    audit = cfg.collections[0].validation.result_consistency.data_integrity_audit
    assert audit is not None
    assert audit.min_overlap_ratio == pytest.approx(0.97)
    assert audit.max_median_ohlc_diff_bps == pytest.approx(2.0)
    assert audit.max_p95_ohlc_diff_bps == pytest.approx(10.0)


def test_load_config_transaction_cost_robustness_inherits_global_overrides(tmp_path: Path):
    cfg = _load_from_blocks(
        tmp_path,
        validation_block={
            "result_consistency": _result_consistency_block(
                transaction_cost_robustness=_transaction_cost_robustness_block(
                    mode="enforce",
                    breakeven={
                        "enabled": True,
                        "min_multiplier": 1.0,
                        "max_multiplier": 5.0,
                        "max_iterations": 8,
                        "tolerance": 0.05,
                    },
                )
            )
        },
        collection_validation_block={
            "result_consistency": _result_consistency_block(
                transaction_cost_robustness={
                    "stress_multipliers": [3.0],
                    "breakeven": {"max_multiplier": 4.0},
                }
            )
        },
    )
    assert cfg.validation is not None
    assert cfg.validation.result_consistency is not None
    global_policy = cfg.validation.result_consistency.transaction_cost_robustness
    assert global_policy is not None
    assert global_policy.mode == "enforce"
    assert global_policy.stress_multipliers == [2.0, 5.0]
    assert global_policy.max_metric_drop_pct == pytest.approx(0.3)
    assert global_policy.breakeven is not None
    assert global_policy.breakeven.enabled is True
    assert global_policy.breakeven.min_multiplier == pytest.approx(1.0)
    assert global_policy.breakeven.max_multiplier == pytest.approx(5.0)
    assert cfg.collections[0].validation is not None
    collection_policy = cfg.collections[0].validation.result_consistency
    assert collection_policy is not None
    tc_policy = collection_policy.transaction_cost_robustness
    assert tc_policy is not None
    assert tc_policy.mode == "enforce"
    assert tc_policy.stress_multipliers == [3.0]
    assert tc_policy.max_metric_drop_pct == pytest.approx(0.3)
    assert tc_policy.breakeven is not None
    assert tc_policy.breakeven.enabled is True
    assert tc_policy.breakeven.min_multiplier == pytest.approx(1.0)
    assert tc_policy.breakeven.max_multiplier == pytest.approx(4.0)
    assert tc_policy.breakeven.max_iterations == 8
    assert tc_policy.breakeven.tolerance == pytest.approx(0.05)


def test_load_config_transaction_cost_robustness_requires_mode(tmp_path: Path):
    with pytest.raises(ValueError, match=r"validation\.result_consistency\.transaction_cost_robustness"):
        _load_from_blocks(
            tmp_path,
            validation_block={
                "result_consistency": _result_consistency_block(
                    transaction_cost_robustness={
                        "stress_multipliers": [2.0, 5.0],
                        "max_metric_drop_pct": 0.3,
                    }
                )
            },
        )


def test_load_config_transaction_cost_robustness_rejects_string_breakeven_enabled(
    tmp_path: Path,
):
    with pytest.raises(
        ValueError,
        match=r"validation\.result_consistency\.transaction_cost_robustness\.breakeven\.enabled",
    ):
        _load_from_blocks(
            tmp_path,
            validation_block={
                "result_consistency": _result_consistency_block(
                    transaction_cost_robustness=_transaction_cost_robustness_block(
                        breakeven={
                            "enabled": "true",
                            "min_multiplier": 1.0,
                            "max_multiplier": 5.0,
                            "max_iterations": 8,
                            "tolerance": 0.05,
                        },
                    )
                )
            },
        )


def test_load_config_transaction_cost_robustness_requires_breakeven_fields(tmp_path: Path):
    with pytest.raises(ValueError, match=r"validation\.result_consistency\.transaction_cost_robustness\.breakeven"):
        _load_from_blocks(
            tmp_path,
            validation_block={
                "result_consistency": _result_consistency_block(
                    transaction_cost_robustness=_transaction_cost_robustness_block(
                        breakeven={}
                    )
                )
            },
        )


def test_load_config_transaction_cost_robustness_rejects_nan_max_metric_drop_pct(tmp_path: Path):
    with pytest.raises(
        ValueError,
        match=r"validation\.result_consistency\.transaction_cost_robustness\.max_metric_drop_pct` must be finite",
    ):
        _load_from_blocks(
            tmp_path,
            validation_block={
                "result_consistency": _result_consistency_block(
                    transaction_cost_robustness=_transaction_cost_robustness_block(
                        max_metric_drop_pct=float("nan")
                    )
                )
            },
        )


def test_parse_optional_float_list_rejects_non_finite_values():
    with pytest.raises(ValueError, match=r"`sample.values\[0\]` must be finite"):
        parse_optional_float_list(
            {"values": [float("inf")]},
            "sample",
            "values",
        )


def test_parse_optional_float_list_rejects_non_numeric_values_with_field_context():
    with pytest.raises(ValueError, match=r"Invalid `sample.values\[0\]`: expected a number"):
        parse_optional_float_list(
            {"values": ["abc"]},
            "sample",
            "values",
        )


def test_load_config_rejects_string_int_fields(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  result_consistency:
    min_trades: "20"
    outlier_dependency:
      slices: 5
      profit_share_threshold: 0.6
      trade_share_threshold: 0.6
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError, match=r"validation\.result_consistency\.min_trades"):
        load_config(path)


def test_load_config_rejects_string_float_fields(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  result_consistency:
    min_trades: 20
    transaction_cost_robustness:
      mode: analytics
      stress_multipliers: [2.0, 5.0]
      max_metric_drop_pct: "0.3"
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(
        ValueError,
        match=r"validation\.result_consistency\.transaction_cost_robustness\.max_metric_drop_pct",
    ):
        load_config(path)


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
    cfg = _load_from_blocks(
        tmp_path,
        validation_block={
            "result_consistency": _result_consistency_block(
                outlier_dependency={
                    "slices": 6,
                    "profit_share_threshold": 0.80,
                    "trade_share_threshold": 0.05,
                },
                execution_price_variance={"price_tolerance_bps": 1.0},
            )
        },
    )
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
    cfg = _load_from_blocks(
        tmp_path,
        validation_block={
            "result_consistency": _result_consistency_block(
                outlier_dependency={
                    "slices": 8,
                    "profit_share_threshold": 0.80,
                    "trade_share_threshold": 0.05,
                },
                execution_price_variance={"price_tolerance_bps": 0.5},
            )
        },
        collection_validation_block={
            "result_consistency": _result_consistency_block(
                outlier_dependency={
                    "slices": 4,
                    "profit_share_threshold": 0.75,
                    "trade_share_threshold": 0.10,
                }
            )
        },
    )
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
    with pytest.raises(ValueError):
        _load_from_blocks(
            tmp_path,
            validation_block={
                "result_consistency": _result_consistency_block(
                    outlier_dependency={
                        "slices": 1,
                        "profit_share_threshold": 0.80,
                        "trade_share_threshold": 0.05,
                    }
                )
            },
        )


def test_load_config_result_consistency_requires_profit_share_threshold(tmp_path: Path):
    with pytest.raises(ValueError):
        _load_from_blocks(
            tmp_path,
            validation_block={
                "result_consistency": _result_consistency_block(
                    outlier_dependency={"slices": 5, "trade_share_threshold": 0.05}
                )
            },
        )


def test_load_config_result_consistency_requires_trade_share_threshold(tmp_path: Path):
    with pytest.raises(ValueError):
        _load_from_blocks(
            tmp_path,
            validation_block={
                "result_consistency": _result_consistency_block(
                    outlier_dependency={"slices": 5, "profit_share_threshold": 0.80}
                )
            },
        )


def test_load_config_result_consistency_threshold_out_of_range(tmp_path: Path):
    with pytest.raises(ValueError):
        _load_from_blocks(
            tmp_path,
            validation_block={
                "result_consistency": _result_consistency_block(
                    outlier_dependency={
                        "slices": 5,
                        "profit_share_threshold": 1.2,
                        "trade_share_threshold": 0.05,
                    }
                )
            },
        )


def test_load_config_result_consistency_execution_price_variance_requires_tolerance(tmp_path: Path):
    with pytest.raises(ValueError):
        _load_from_blocks(
            tmp_path,
            validation_block={
                "result_consistency": _result_consistency_block(execution_price_variance={})
            },
        )


def test_load_config_result_consistency_execution_price_variance_tolerance_non_negative(tmp_path: Path):
    with pytest.raises(ValueError):
        _load_from_blocks(
            tmp_path,
            validation_block={
                "result_consistency": _result_consistency_block(
                    execution_price_variance={"price_tolerance_bps": -0.1}
                )
            },
        )


def test_load_config_result_consistency_allows_missing_min_metric(tmp_path: Path):
    cfg = _load_from_blocks(
        tmp_path,
        validation_block={
            "result_consistency": _result_consistency_block(
                min_metric=None,
                execution_price_variance={"price_tolerance_bps": 1.0},
            ),
        },
    )
    assert cfg.validation is not None
    assert cfg.validation.result_consistency is not None
    assert cfg.validation.result_consistency.min_metric is None


def test_load_config_result_consistency_allows_missing_min_trades(tmp_path: Path):
    cfg = _load_from_blocks(
        tmp_path,
        validation_block={
            "result_consistency": _result_consistency_block(
                min_trades=None,
                execution_price_variance={"price_tolerance_bps": 1.0},
            ),
        },
    )
    assert cfg.validation is not None
    assert cfg.validation.result_consistency is not None
    assert cfg.validation.result_consistency.min_trades is None


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
    assert cfg.validation.data_quality.calendar is not None
    assert cfg.validation.data_quality.calendar.kind == "exchange"
    assert cfg.validation.data_quality.calendar.exchange == "XNYS"
    assert cfg.validation.data_quality.calendar.timezone == "UTC-05:00"


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
    calendar:
      kind: exchange
      timezone: America/New_York
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_data_quality_rejects_legacy_continuity_calendar_location(tmp_path: Path):
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
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError, match=r"validation\.data_quality\.continuity\.calendar"):
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
    assert cfg.validation.data_quality.calendar is None


def test_load_config_data_quality_ohlc_integrity_defaults(tmp_path: Path):
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
    ohlc_integrity: {}
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.validation is not None
    assert cfg.validation.data_quality is not None
    assert cfg.collections[0].validation is not None
    effective_dq = cfg.collections[0].validation.data_quality
    assert effective_dq is not None
    assert effective_dq.ohlc_integrity is not None
    assert effective_dq.ohlc_integrity.max_invalid_bar_pct == pytest.approx(0.0)
    assert effective_dq.ohlc_integrity.allow_negative_price is False
    assert effective_dq.ohlc_integrity.allow_negative_volume is False


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
    calendar: {}
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.validation is not None
    assert cfg.validation.data_quality is not None
    assert cfg.validation.data_quality.calendar is not None
    assert cfg.validation.data_quality.calendar.kind is None
    assert cfg.collections[0].validation is not None
    assert cfg.collections[0].validation.data_quality is not None
    assert cfg.collections[0].validation.data_quality.calendar is not None
    assert cfg.collections[0].validation.data_quality.calendar.kind == "auto"


def test_load_config_data_quality_ohlc_integrity_settings(tmp_path: Path):
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
    ohlc_integrity:
      max_invalid_bar_pct: 1.5
      allow_negative_price: true
      allow_negative_volume: false
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.validation is not None
    assert cfg.validation.data_quality is not None
    assert cfg.validation.data_quality.ohlc_integrity is not None
    assert cfg.validation.data_quality.ohlc_integrity.max_invalid_bar_pct == pytest.approx(1.5)
    assert cfg.validation.data_quality.ohlc_integrity.allow_negative_price is True
    assert cfg.validation.data_quality.ohlc_integrity.allow_negative_volume is False


def test_load_config_data_quality_ohlc_integrity_rejects_string_booleans(tmp_path: Path):
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
    ohlc_integrity:
      allow_negative_price: "false"
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError, match=r"validation\.data_quality\.ohlc_integrity\.allow_negative_price"):
        load_config(path)


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
    cfg = _load_from_blocks(
        tmp_path,
        validation_block={
            "result_consistency": _result_consistency_block(
                lookahead_shuffle_test={"permutations": 100, "pvalue_max": 0.10, "seed": 1337}
            )
        },
        collection_validation_block={
            "result_consistency": _result_consistency_block(
                lookahead_shuffle_test={"permutations": 100, "seed": 7, "max_failed_permutations": 2}
            )
        },
    )
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
    cfg = _load_from_blocks(
        tmp_path,
        validation_block={
            "result_consistency": _result_consistency_block(
                lookahead_shuffle_test={"permutations": 100, "pvalue_max": 0.5, "seed": 17}
            )
        },
        collection_validation_block={
            "result_consistency": _result_consistency_block(
                lookahead_shuffle_test={"permutations": 100, "pvalue_max": 0.25}
            )
        },
    )
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
    with pytest.raises(ValueError, match=r"validation\.result_consistency\.lookahead_shuffle_test\.permutations"):
        _load_from_blocks(
            tmp_path,
            validation_block={
                "result_consistency": _result_consistency_block(
                    lookahead_shuffle_test={"permutations": 99}
                )
            },
        )


def test_load_config_lookahead_shuffle_test_invalid_max_failed_permutations(tmp_path: Path):
    with pytest.raises(
        ValueError,
        match=r"validation\.result_consistency\.lookahead_shuffle_test\.max_failed_permutations",
    ):
        _load_from_blocks(
            tmp_path,
            validation_block={
                "result_consistency": _result_consistency_block(
                    lookahead_shuffle_test={"permutations": 100, "max_failed_permutations": 101}
                )
            },
        )


def test_load_config_lookahead_shuffle_test_invalid_max_failed_permutations_default_permutations(
    tmp_path: Path,
):
    with pytest.raises(
        ValueError,
        match=r"validation\.result_consistency\.lookahead_shuffle_test\.max_failed_permutations",
    ):
        _load_from_blocks(
            tmp_path,
            validation_block={
                "result_consistency": _result_consistency_block(
                    lookahead_shuffle_test={"max_failed_permutations": 101}
                )
            },
        )


def test_load_config_lookahead_shuffle_test_requires_pvalue_max(tmp_path: Path):
    with pytest.raises(ValueError, match=r"missing required field\(s\): pvalue_max"):
        _load_from_blocks(
            tmp_path,
            validation_block={
                "result_consistency": _result_consistency_block(
                    lookahead_shuffle_test={"permutations": 100}
                )
            },
        )


def test_load_config_lookahead_shuffle_test_rejects_threshold_key(tmp_path: Path):
    with pytest.raises(ValueError, match=r"lookahead_shuffle_test\.threshold"):
        _load_from_blocks(
            tmp_path,
            validation_block={
                "result_consistency": _result_consistency_block(
                    lookahead_shuffle_test={
                        "permutations": 100,
                        "pvalue_max": 0.05,
                        "threshold": 0.2,
                    }
                )
            },
        )


def test_load_config_lookahead_shuffle_test_invalid_pvalue_max(tmp_path: Path):
    with pytest.raises(
        ValueError,
        match=r"validation\.result_consistency\.lookahead_shuffle_test\.pvalue_max",
    ):
        _load_from_blocks(
            tmp_path,
            validation_block={
                "result_consistency": _result_consistency_block(
                    lookahead_shuffle_test={"permutations": 100, "pvalue_max": 1.2}
                )
            },
        )


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


def test_resolve_validation_overrides_rejects_programmatic_transaction_cost_bool_numeric(
    tmp_path: Path,
):
    cfg = _load_from_blocks(
        tmp_path,
        validation_block={
            "result_consistency": _result_consistency_block(
                transaction_cost_robustness=_transaction_cost_robustness_block()
            )
        },
    )
    assert cfg.collections[0].validation is not None
    assert cfg.collections[0].validation.result_consistency is not None
    assert cfg.collections[0].validation.result_consistency.transaction_cost_robustness is not None

    cfg.collections[0].validation.result_consistency.transaction_cost_robustness.max_metric_drop_pct = True

    with pytest.raises(
        ValueError,
        match=r"validation\.result_consistency\.transaction_cost_robustness\.max_metric_drop_pct",
    ):
        resolve_validation_overrides(cfg)


def test_resolve_validation_overrides_rejects_programmatic_transaction_cost_breakeven_bool_int(
    tmp_path: Path,
):
    cfg = _load_from_blocks(
        tmp_path,
        validation_block={
            "result_consistency": _result_consistency_block(
                transaction_cost_robustness=_transaction_cost_robustness_block(
                    breakeven={
                        "enabled": True,
                        "min_multiplier": 1.0,
                        "max_multiplier": 5.0,
                        "max_iterations": 8,
                        "tolerance": 0.05,
                    }
                )
            )
        },
    )
    assert cfg.collections[0].validation is not None
    assert cfg.collections[0].validation.result_consistency is not None
    assert cfg.collections[0].validation.result_consistency.transaction_cost_robustness is not None
    assert cfg.collections[0].validation.result_consistency.transaction_cost_robustness.breakeven is not None

    cfg.collections[0].validation.result_consistency.transaction_cost_robustness.breakeven.max_iterations = True

    with pytest.raises(
        ValueError,
        match=r"validation\.result_consistency\.transaction_cost_robustness\.breakeven\.max_iterations",
    ):
        resolve_validation_overrides(cfg)


def test_load_config_rejects_fractional_min_trades(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  result_consistency:
    min_trades: 1.9
    outlier_dependency:
      slices: 5
      profit_share_threshold: 0.6
      trade_share_threshold: 0.6
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError, match=r"validation\.result_consistency\.min_trades"):
        load_config(path)


def test_resolve_validation_overrides_rejects_programmatic_fractional_int(tmp_path: Path):
    cfg = _load_from_blocks(
        tmp_path,
        validation_block={
            "result_consistency": _result_consistency_block(
                outlier_dependency={
                    "slices": 5,
                    "profit_share_threshold": 0.6,
                    "trade_share_threshold": 0.6,
                }
            )
        },
    )
    assert cfg.collections[0].validation is not None
    assert cfg.collections[0].validation.result_consistency is not None

    cfg.collections[0].validation.result_consistency.min_trades = 1.9

    with pytest.raises(ValueError, match=r"validation\.result_consistency\.min_trades"):
        resolve_validation_overrides(cfg)
