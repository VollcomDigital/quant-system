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

This table tracks which metadata is created at each stage/check and where it
lands (logs/caches/stores/reports).

| Stage / check | Metadata created or updated | Lands in logs | Lands in caches/stores | Lands in reports |
|---|---|---|---|---|
| `collection_validation` gate | `GateDecision` (`passed/action/reasons`), `JobState.decisions`/`reasons_by_stage` | `collection_validation_gate` event (when non-continue) | in-memory `JobState` only | only if failure copied to `runner.failures` |
| `data_fetch` gate | same gate metadata as above | `data_fetch_gate` | in-memory `JobState` only | only if failure copied to `runner.failures` |
| `data_validation` gate | continuity + reliability reasons in `ValidatedData`; gate decision in state | `data_validation_gate` | in-memory `ValidatedData`, `JobState` | failure entries in `summary.json.failures` / `health.md` when blocked |
| `data_preparation` gate | prepared execution context (`ExecutionPreparedData`), gate decision in state | `data_preparation_gate` | in-memory only | only if failure copied to `runner.failures` |
| `strategy_optimization` gate | plan skip reasons/details, gate decision; optional runtime-threshold detail in `plan.optimization_details` | `strategy_optimization_gate` with `strategy` + `search_method` | in-memory plan/state; runtime cache counters | failures in `summary.json.failures` when rejected/skipped |
| Strategy evaluation (cache lookup) | `EvaluationRequest` including `validation_config_hash` and `strategy_fingerprint`; cache hit/miss counters | no gate event; normal process logs only | read/write `evaluation_cache.sqlite` and `results.sqlite` | aggregate counters in `summary.json.metrics` / `metrics.prom` |
| Strategy evaluation (fresh compute) | evaluator `stats` (`trade_meta`, metrics), best-candidate state, cached metric/stats payloads | no gate event | writes to `evaluation_cache.sqlite` (raw stats) and `results.sqlite` (enriched stats) | reflected in final result stats if candidate selected |
| `strategy_validation` checks | result-consistency reasons; lookahead diagnostics attached to `best_stats.post_run_meta.lookahead_shuffle_test` | `strategy_validation_gate` with `strategy` | `BestResult.stats` persisted to `result_store.sqlite`; no `post_run_meta` in eval cache writes | failures in `summary.json.failures`; selected stats exported in run outputs |
| Result finalize | `BestResult` + `ResultRecord` write | no gate event | `result_store.sqlite` (`result_records`) | `all_results.csv`, `summary.csv`, `report.md`, `summary.json` |
| Run finalize | validation profile metadata, active/inactive gates, run counters, failures list | summary lines in CLI output | `result_store.sqlite` (`run_metadata`) | `summary.json`, `metrics.prom`, `health.md` |

### Sink Matrix

| Sink | What is stored |
|---|---|
| Structured gate logs (`log_json`) | Stage gate verdict + job context + small `context_extra` (`strategy`, `search_method` where applicable). |
| `runner.failures` (in-memory) | Normalized failure payloads (`collection/symbol/timeframe/source/stage/error` + optional `strategy`). |
| `.cache/evaluation/evaluation_cache.sqlite` | Per-evaluation cache rows keyed by request identity + `mode_config_hash` + `validation_config_hash` + `strategy_fingerprint`; stores `metric_value` + raw `stats`. |
| `.cache/results/results.sqlite` | Run-scoped result cache rows (`run_id`, params, metric, enriched `stats`) for reporting/backward-compatible retrieval. |
| `.cache/evaluation/result_store.sqlite` (`result_records`) | Final selected `BestResult` rows for the run, including persisted `stats` (for example `trade_meta`, `post_run_meta`). |
| `.cache/evaluation/result_store.sqlite` (`run_metadata`) | Effective validation profile + active/inactive gate ids for the run. |
| `reports/<run>/summary.json` | run timing, counters, failures list, dashboard summary, validation metadata snapshot. |
| `reports/<run>/health.md` | Human-readable table of failures derived from `runner.failures`. |
| `reports/<run>/metrics.prom` | Prometheus counters (cache hits/misses, eval counts, duration, etc.). |

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
