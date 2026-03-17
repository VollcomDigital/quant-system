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

- Modules: `validation.data_quality`, `validation.optimization`.
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
