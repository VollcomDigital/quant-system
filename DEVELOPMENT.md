# Development Notes

This document captures implementation details that are useful when maintaining
the backtest engine.

## Backtest Runner Overview

Source: `BacktestRunner.run_all` in `src/backtest/runner.py`.

### High-level flow

1. Build executable jobs from collections/symbols/timeframes.
2. For each job, run gate stages in order:
   - collection validation
   - data fetch
   - data validation
   - execution-context preparation
3. For each discovered strategy:
   - create strategy plan (fixed params + search space)
   - apply job-level optimization constraints
   - validate strategy plan
   - run strategy evaluation (baseline/grid/optuna path)
   - validate strategy outcome
4. Assemble `BestResult`, persist stores/caches, and return run results.

### Gate model

- Each stage returns a `GateDecision` with:
  - `passed`
  - `action` (`continue`, `skip_optimization`, `baseline_only`, `skip_job`, `skip_collection`, `reject_result`)
  - `reasons`
- Stage decisions are composed by `_compose_gate_decisions`.
- `skip_optimization` is a job-level signal; strategy execution falls back to
  baseline-only evaluation.

### Evaluation model

- Runner orchestration owns job/strategy loops and gating.
- Evaluator owns simulation + metric computation.
- Cache/store writes happen after evaluation results are enriched.

### Flow diagram (Mermaid)

```mermaid
flowchart TD
  A([run_all]) --> B[_create_job_list]
  B --> C{{for each JobContext}}
  C --> D[_collection_validation]
  D --> GD{collection_validation gate}
  GD -->|fail| C
  GD --> E[_data_fetch]
  E --> GE{data_fetch gate}
  GE -->|fail| C
  GE --> F[_data_validation_common + _data_validation]
  F --> GF{data_validation gate}
  GF -->|fail| C
  GF --> G[_execution_context_prepare_common + _execution_context_prepare]
  G --> GG{data_preparation gate}
  GG -->|fail| C
  GG --> H{{for each strategy in external_index}}
  H --> I[_strategy_create_plan]
  I --> J[_apply_policy_constraints_to_plan]
  J --> K[_strategy_validate_plan_common + _strategy_validate_plan]
  K --> GK{strategy_optimization gate}
  GK -->|fail/reject| H
  GK --> L[_strategy_run + _strategy_evaluation + evaluator.evaluate]
  L --> M[_strategy_validate_results_common + _strategy_validate_results]
  M --> GM{strategy_validation gate}
  GM -->|reject| H
  GM --> N[_build_result_record + _result_store_insert]
  N --> H
  H --> C
  C --> Z([return BestResult list])
```

## Metadata By Stage

Source: `BacktestRunner.run_all` and helper methods in `src/backtest/runner.py`.

### 1) Final stats model (single source of truth)

`result_store.result_records.stats_json` stores `BestResult.stats` for each
selected strategy result.

Top-level keys in `BestResult.stats` (full current overview):

| Parent key | Child keys (full/current) | Notes |
|---|---|---|
| core performance keys | `sharpe`, `sortino`, `omega`, `tail_ratio`, `profit`, `pain_index`, `trades`, `max_drawdown`, `cagr`, `calmar`, `equity_curve`, `drawdown_curve`, `trades_log` | base evaluator/runner stats payload |
| `trade_meta` | `outlier_dependency`, `execution_price_variance` | produced by evaluator |
| `trade_meta.outlier_dependency` | `is_complete`, `analyzed_trades_count`, `total_trades`, `trade_count_with_pnl`, `total_positive_profit`, `profit_share_threshold_used`, `dominant_trade_count_for_profit_share`, `dominant_trade_share_for_profit_share`, `slices_used`, `max_slice_profit_share`, `reason`, `expected_trades` | `reason`/`expected_trades` appear on incomplete paths |
| `trade_meta.execution_price_variance` | `is_complete`, `checked_fills`, `violations`, `violation_ratio`, `reason`, `price_tolerance_bps_used` | complete or incomplete path |
| `data_reliability` | `continuity` (+ pre-existing reliability keys if present) | merged in `_enrich_evaluation_stats` |
| `optimization` (optional) | `skipped`, `reason`, `reasons`, `min_bars_required`, `bars_available`, `reliability_reasons`, `runtime_error_threshold` | present when optimization details exist |
| `post_run_meta` (optional) | `lookahead_shuffle_test` | added during strategy-validation post check |
| `post_run_meta.lookahead_shuffle_test` | `is_complete`, `reason`, `reason_detail`, `metric_name`, `permutations`, `seed`, `failed_permutations`, `max_failed_permutations`, `threshold`, `finite_permutations`, `median_shuffled_metric`, `min_shuffled_metric`, `max_shuffled_metric` | fields vary by complete/incomplete path |

### 2) Where each parent key lands

Legend: `[x]` = lands there, `[ ]` = does not land there.

| Metadata key/group | Gate logs | `runner.failures` | `evaluation_cache.sqlite` | `results.sqlite` | `result_store.sqlite` (`result_records`) | `summary.json` |
|---|---|---|---|---|---|---|
| Gate decision (`passed/action/reasons`) | [x] | [x] (flattened error on fail only) | [ ] | [ ] | [ ] | [x] (via failures list) |
| Gate context `strategy` | [x] | [x] | [ ] | [ ] | [ ] | [x] (inside failures) |
| Gate context `search_method` | [x] (`strategy_optimization_gate`) | [ ] | [ ] | [ ] | [ ] | [ ] |
| Core performance keys | [ ] | [ ] | [x] | [x] | [x] | [ ] |
| `trade_meta` | [ ] | [ ] | [x] | [x] | [x] | [ ] |
| `data_reliability` | [ ] | [ ] | [ ] | [x] | [x] | [ ] |
| `optimization` | [ ] | [ ] | [ ] | [x] | [x] | [ ] |
| `post_run_meta` | [ ] | [ ] | [ ] | [ ] | [x] | [ ] |
| Run-level validation profile (`active_gates`, etc.) | [ ] | [ ] | [ ] | [ ] | [x] (`run_metadata`) | [x] (`validation`) |
| Run counters (`result_cache_hits`, etc.) | [ ] | [ ] | [ ] | [ ] | [ ] | [x] (`metrics`) + `metrics.prom` |

### 3) Short rule

- Research/debug metadata that must survive belongs in `BestResult.stats`
  (result store).
- Operational gate signals belong in gate logs/failures and should stay small.

## Continuity Score Calendar Behavior

Source: `BacktestRunner.compute_continuity_score` in
`src/backtest/runner.py`.

### Timeframe shape terms

- `daily`: timeframe unit is `d/day/days`.
- `eod-like`: broader end-of-day family (`daily`, `weekly`, `monthly`).

### Calendar decision flow

1. `calendar_kind=exchange` + daily + exchange code set:
   - Uses `exchange_calendars` sessions for expected dates.
2. `calendar_kind in {weekday, exchange}` + daily:
   - Uses weekday expected dates (Mon-Fri).
3. All other cases (including weekly/monthly):
   - Uses fixed-delta gap counting from actual timestamps.

### Missing-gap implementation

- Missing bars against an expected index are computed via boolean membership.
- Largest consecutive missing gap is computed with vectorized NumPy transition
  detection.

### Related tests

- `tests/test_backtest_runner.py::test_compute_continuity_score_weekend_gap_not_missing_for_weekday_calendar`
- `tests/test_backtest_runner.py::test_compute_continuity_score_exchange_calendar_ignores_market_holiday`
- `tests/test_backtest_runner.py::test_compute_continuity_score_weekday_calendar_non_daily_uses_fixed_delta`

## Validation Policy Resolution

Source: `resolve_validation_overrides` in `src/config.py`.

- Resolution ownership is in config loading, not in runner runtime.
- Effective policy is materialized on each collection under `collection.validation`.
- Runner then only reads collection-level effective policies.

### Resolution rules (per module)

- Modules: `validation.data_quality`, `validation.optimization`, `validation.result_consistency`.
- If neither global nor collection policy is set: module stays disabled (`None`).
- If only global is set: collection inherits global policy.
- If only collection is set: collection policy is normalized and used.
- If both are set: collection override takes precedence.
- The persisted validation profile stores only effective collection-level policies
  (`profile.collections`), not top-level global input blocks.

### Why this matters

- Runner code stays lean: no global-vs-collection merge branches in gate execution.
- Hashing/job profiles use effective collection policy consistently.
- Behavior is deterministic across tests and real runs after `load_config`.

## Config Parse-To-Effective Pattern

Source: `load_config`, parser helpers, and merge helpers in `src/config.py`.

This is the implementation contract for validation/policy config work.

### Naming and phase contract

Use this lifecycle for each policy module:

1. `_parse_*`
   - Input: raw YAML value (`Any`).
   - Output: typed dataclass or `None`.
   - Job:
     - enforce mapping shape (`require_mapping`)
     - parse primitive fields (`parse_required_*`, `parse_optional_*`)
     - validate required keys for that module
   - Rule: do not apply inheritance-sensitive defaults here.

2. `_normalize_*`
   - Input: typed dataclass (possibly from parser or merge result).
   - Output: validated, normalized dataclass.
   - Job:
     - value range checks
     - string normalization
     - nested block normalization
   - Rule: do not inject defaults in normalize.

3. `_apply_*_defaults`
   - Input: normalized effective config dataclass.
   - Output: normalized dataclass with all effective defaults materialized.
   - Job:
     - fill inheritance-sensitive defaults after merge.
   - Rule: defaults must have a single owner function per module.

4. `_merge_*_config`
   - Input: `base` (global), `override` (collection).
   - Output: effective module config or `None`.
   - Job:
     - resolve fields via `_merged_field` (override wins only when non-`None`)
     - for parent module merges: call `_normalize_*` then `_apply_*_defaults`
     - for nested sub-merges: only merge fields; do not normalize/apply at sub-level
   - Rule: merge is where effective defaults are materialized.

5. `resolve_validation_overrides`
   - Build normalized global runtime policy snapshots.
   - Merge each collection override onto global.
   - Write final effective policy to `collection.validation`.

6. Runner consumption
   - Runner reads effective collection policy only.
   - Runner does not perform global/collection merge logic.

### Style Contract (Strict)

All validation modules must use the same implementation style.

Canonical merge shape:

```python
def _merge_<module>_config(base, override):
    if base is None and override is None:
        return None
    normalized = _normalize_<module>_config(
        <ModuleConfig>(
            field_a=_merged_field(base, override, "field_a"),
            field_b=_merged_field(base, override, "field_b"),
            nested=_merge_<nested>_config(...),
        ),
        "validation.<module_path>",
    )
    return _apply_<module>_defaults(
        _require_normalized(normalized, "validation.<module_path>")
    )
```

Required conventions:

- `_merge_*_config` contains no inline validation branches beyond the `None/None` short-circuit.
- `_merge_*_config` does not call `_parse_*`.
- Nested `_merge_*` helpers under a parent module should be raw field merges only; parent module merge is the single owner for normalize + apply-defaults.
- `_normalize_*` is the only place for validation/canonicalization rules.
- `_apply_*_defaults` is the only place where module defaults are injected.
- Even when a module currently has no defaults, keep `_apply_*_defaults` as an identity function for consistency.
- `_parse_*` returns normalized parsed objects and never injects inheritance-sensitive defaults.
- Error prefixes must match full config paths (for example `validation.result_consistency...`) for stable tests and UX.
- Production code should not rely on `assert` for runtime invariants; use explicit guards (`if ...: raise/return`) so behavior is stable when Python is run with `-O`.
- In config merge helpers, use `_require_normalized(...)` for post-normalize non-`None` guarantees instead of `assert`.

### Defaulting rules (strict)

- Parse-stage defaults:
  - allowed for top-level runtime config (non-override policy), e.g. `metric`, `engine`, `fees`.
  - avoid for override-sensitive policy fields.

- Merge-stage defaults:
  - preferred for policy fields that participate in global+collection inheritance.
  - implemented in `_apply_*_defaults` (called from merge path).
  - example: `stationarity.min_points`
    - parse: may remain `None`
    - merge effective config: `_apply_stationarity_defaults` sets `30` when still `None`
    - collection `null`/omission inherits global value if global is set.

### Practical implementation template for new policy module

When adding a new module under `validation.*`:

1. Add dataclass fields (global + collection paths).
2. Add `_parse_new_module(...)` with strict shape/key/range validation.
3. Add `_normalize_new_module(...)` for validation/normalization only.
4. Add `_apply_new_module_defaults(...)` for effective default injection only.
5. Add `_merge_new_module_config(base, override)` and call normalize + apply-defaults there.
   - For nested sub-blocks, implement raw `_merge_<sub>_config(...)` helpers and let the parent module merge own normalize + apply-defaults.
6. Wire module into:
   - `_parse_validation_*` tree
   - `_merge_data_quality_config` / relevant parent merge
   - `resolve_validation_overrides`
7. Update runner to read only `collection.validation...` effective values.
8. Add tests for:
   - parse happy path
   - missing required keys
   - range/type errors
   - global-only, collection-only, global+override merge behavior
   - explicit `None` inheritance behavior
   - parse-stage object keeps inheritance-sensitive fields as `None`
   - effective collection policy has defaults materialized

### Existing examples in codebase

- Data quality / optimization / result consistency:
  - all follow the same `merge -> normalize -> apply_defaults` style.
- Stationarity:
  - final `min_points` default injected in `_apply_stationarity_defaults`.
- Result consistency:
  - nested modules merged and enabled independently; module disabled when `None`.

### Responsibility split

- `config.py`: parse, normalize, apply defaults, merge, and materialize effective policies.
- `runner.py`: gate enforcement and diagnostics only.
