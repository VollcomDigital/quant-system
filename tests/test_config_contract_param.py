from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.config import load_config


def _get_attr_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def _load_from_blocks(
    tmp_path: Path,
    *,
    validation_block: dict[str, Any],
    collection_validation_block: dict[str, Any] | None = None,
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
        "validation": deepcopy(validation_block),
    }
    if collection_validation_block is not None:
        raw["collections"][0]["validation"] = deepcopy(collection_validation_block)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return load_config(path)


CONTRACT_CASES = [
    {
        "id": "optimization.runtime_error_max_per_tuple",
        "parse_path": "optimization.runtime_error_max_per_tuple",
        "effective_path": "optimization.runtime_error_max_per_tuple",
        "global_for_default": {"optimization": {"on_fail": "skip_job", "min_bars": 123, "dof_multiplier": 7}},
        "default_value": 1,
        "global_for_inherit": {
            "optimization": {
                "on_fail": "skip_job",
                "min_bars": 123,
                "dof_multiplier": 7,
                "runtime_error_max_per_tuple": 4,
            }
        },
        "collection_for_inherit": {
            "optimization": {
                "on_fail": "skip_job",
                "min_bars": 200,
                "dof_multiplier": 9,
            }
        },
        "inherit_value": 4,
    },
    {
        "id": "stationarity.min_points",
        "parse_path": "data_quality.stationarity.min_points",
        "effective_path": "data_quality.stationarity.min_points",
        "global_for_default": {
            "data_quality": {
                "on_fail": "skip_job",
                "stationarity": {"adf_pvalue_max": 0.05},
            }
        },
        "default_value": 30,
        "global_for_inherit": {
            "data_quality": {
                "on_fail": "skip_job",
                "stationarity": {"adf_pvalue_max": 0.05, "min_points": 50},
            }
        },
        "collection_for_inherit": {
            "data_quality": {
                "on_fail": "skip_job",
                "stationarity": {"adf_pvalue_max": 0.1},
            }
        },
        "inherit_value": 50,
    },
    {
        "id": "lookahead.permutations",
        "parse_path": "result_consistency.lookahead_shuffle_test.permutations",
        "effective_path": "result_consistency.lookahead_shuffle_test.permutations",
        "global_for_default": {"result_consistency": {"lookahead_shuffle_test": {}}},
        "default_value": 100,
        "global_for_inherit": {"result_consistency": {"lookahead_shuffle_test": {"permutations": 133}}},
        "collection_for_inherit": {"result_consistency": {"lookahead_shuffle_test": {"threshold": 0.2}}},
        "inherit_value": 133,
    },
    {
        "id": "lookahead.threshold",
        "parse_path": "result_consistency.lookahead_shuffle_test.threshold",
        "effective_path": "result_consistency.lookahead_shuffle_test.threshold",
        "global_for_default": {"result_consistency": {"lookahead_shuffle_test": {}}},
        "default_value": 0.0,
        "global_for_inherit": {"result_consistency": {"lookahead_shuffle_test": {"threshold": 0.5}}},
        "collection_for_inherit": {"result_consistency": {"lookahead_shuffle_test": {"permutations": 100}}},
        "inherit_value": 0.5,
    },
    {
        "id": "lookahead.seed",
        "parse_path": "result_consistency.lookahead_shuffle_test.seed",
        "effective_path": "result_consistency.lookahead_shuffle_test.seed",
        "global_for_default": {"result_consistency": {"lookahead_shuffle_test": {}}},
        "default_value": 1337,
        "global_for_inherit": {"result_consistency": {"lookahead_shuffle_test": {"seed": 17}}},
        "collection_for_inherit": {"result_consistency": {"lookahead_shuffle_test": {"permutations": 100}}},
        "inherit_value": 17,
    },
    {
        "id": "calendar.kind",
        "parse_path": "data_quality.continuity.calendar.kind",
        "effective_path": "data_quality.continuity.calendar.kind",
        "global_for_default": {
            "data_quality": {
                "on_fail": "skip_job",
                "continuity": {"calendar": {}},
            }
        },
        "default_value": "auto",
        "global_for_inherit": {
            "data_quality": {
                "on_fail": "skip_job",
                "continuity": {"calendar": {"kind": "exchange", "exchange": "XNYS"}},
            }
        },
        "collection_for_inherit": {
            "data_quality": {
                "on_fail": "skip_job",
                "continuity": {"min_score": 0.95, "calendar": {}},
            }
        },
        "inherit_value": "exchange",
    },
]


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=[c["id"] for c in CONTRACT_CASES])
def test_validation_contract_parse_keeps_inheritance_sensitive_fields_unset(tmp_path: Path, case: dict[str, Any]):
    cfg = _load_from_blocks(tmp_path, validation_block=case["global_for_default"])
    assert cfg.validation is not None
    assert _get_attr_path(cfg.validation, case["parse_path"]) is None


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=[c["id"] for c in CONTRACT_CASES])
def test_validation_contract_effective_defaults_are_materialized(tmp_path: Path, case: dict[str, Any]):
    cfg = _load_from_blocks(tmp_path, validation_block=case["global_for_default"])
    assert cfg.collections[0].validation is not None
    assert (
        _get_attr_path(cfg.collections[0].validation, case["effective_path"]) == case["default_value"]
    )


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=[c["id"] for c in CONTRACT_CASES])
def test_validation_contract_collection_partial_override_inherits_global_values(
    tmp_path: Path, case: dict[str, Any]
):
    cfg = _load_from_blocks(
        tmp_path,
        validation_block=case["global_for_inherit"],
        collection_validation_block=case["collection_for_inherit"],
    )
    assert cfg.collections[0].validation is not None
    assert (
        _get_attr_path(cfg.collections[0].validation, case["effective_path"]) == case["inherit_value"]
    )
