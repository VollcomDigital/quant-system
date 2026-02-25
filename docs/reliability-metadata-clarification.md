# Reliability Metadata Clarification (VD-4344)

This document addresses clarification questions raised for **VD-4344 – Define "Reliability" Metadata Schema** within the broader User Story VD-3919 (Guardrail – Optimization Only on Reliable Collections).

---

## 1. Are `is_verified`, `min_data_points`, `last_updated` intended as flags set on the Collection YAML level for approval?

**Answer: Partially.**

| Field             | YAML Level | Purpose |
|-------------------|------------|---------|
| **`is_verified`** | ✅ Yes     | **Manual approval flag** set per collection. When `false`, the guardrail blocks optimization for that collection (per VD-3919 checklist: "Manual override for unreliable sources"). Acts as an explicit opt-in/opt-out for optimization eligibility. |
| **`min_data_points`** | ✅ Yes (recommended) | **Per-collection threshold** override. Can mirror the global `param_min_bars` behavior but allows asset-class-specific values (e.g., stricter for crypto, looser for established indices). Falls back to global `param_min_bars` if absent. |
| **`last_updated`** | ❌ No (computed) | **Runtime/computed metadata**, not a static YAML flag. Populated when data is fetched or when reliability checks run. Used for auditing and freshness visibility (e.g., in a Reliability Status Dashboard) rather than manual configuration. |

**Recommendation:** Expose `is_verified` and `min_data_points` on each collection in YAML. Keep `last_updated` as a computed field stored in run metadata or a reliability status file.

---

## 2. Should `min_data_points` in `example.yaml` skip an entire backtest run if too few candles are available?

**Answer: No – it should gate optimization, not the entire backtest.**

Current behavior (see `BacktestRunner`):

- If `len(df) < min_bars_for_optimization`, **optimization is skipped** for that symbol/timeframe/strategy combo.
- The backtest **still runs** with default/fixed parameters.
- The run produces results; stats include `optimization.skipped: true` and the reason.

**Proposed behavior for reliability metadata:**

- `min_data_points` (or `param_min_bars`) should **not** skip the whole run.
- It should **skip optimization** (Optuna/grid) when insufficient data is available.
- Optionally, a **separate flag** (e.g. `reliability.skip_collection_on_failure`) could be introduced to skip a collection entirely when reliability checks fail; this would be a stronger gate and should be explicitly configurable.

**Summary:**

| Scenario                              | Backtest runs? | Optimization runs? |
|---------------------------------------|----------------|--------------------|
| `min_data_points` not met              | ✅ Yes         | ❌ No (skipped)    |
| `is_verified: false`                  | ✅ Yes (optional) | ❌ No (blocked) |
| Explicit "skip collection" threshold  | ❌ No          | ❌ No              |

---

## 3. Is the Data Continuity Score purely a metric for evaluating OHLC data quality?

**Answer: Yes – it is a diagnostic metric, not an enforcement mechanism by default.**

The **Data Continuity Score** should:

- Measure **missing bars** or **structural gaps** in OHLC data (as described in VD-4344).
- Be computed at runtime (e.g. from expected vs. actual bar counts, gap detection).
- Be stored as a metric for dashboards and researcher visibility.
- **Not** automatically cause skips or blocks unless paired with a **threshold** (see Q4).

Separation of concerns:

- **Metric**: Data Continuity Score = diagnostic value (e.g. 0.0–1.0 or percentage of non-missing bars).
- **Enforcement**: A separate YAML threshold (e.g. `max_gaps_pct: 0.1`) determines when the score is considered "fail" and triggers guardrail actions.

---

## 4. Should there be a YAML-defined threshold that determines when a collection is skipped?

**Answer: Yes – but as a separate, explicit configuration.**

Per VD-4349 (Configuration-Based Thresholds) and VD-3919, reliability thresholds should be configurable via YAML. The schema should distinguish:

### A. Global reliability thresholds (config root)

```yaml
reliability:
  max_gaps_pct: 0.1        # > 0.1% missing bars = fail
  max_kurtosis: 10          # kurtosis > 10 = fail
  min_data_points: 2000     # global floor (can override per collection)
  skip_collection_on_fail: false  # if true, skip entire collection when checks fail
```

### B. Per-collection overrides

```yaml
collections:
  - name: crypto
    source: binance
    symbols: [...]
    reliability:
      is_verified: false
      min_data_points: 5000
      max_gaps_pct: 0.05
```

### Logic summary

| Config                           | Role |
|----------------------------------|------|
| `reliability.min_data_points`    | Threshold for sample-size check (N vs Parameters × multiplier) |
| `reliability.max_gaps_pct`       | Threshold for Data Continuity Score / gap check |
| `reliability.max_kurtosis`       | Threshold for distribution spikiness |
| `reliability.skip_collection_on_fail` | If `true`, skip collection entirely when any check fails |
| `collections[].reliability.is_verified` | Manual override; `false` = block optimization |

**Metric vs. threshold:**

- **Metric**: Data Continuity Score, kurtosis, N, etc. – computed values.
- **Threshold**: YAML-defined limits that turn metrics into pass/fail.

---

## Summary: YAML schema proposal for VD-4344

```yaml
reliability:
  min_data_points: 2000
  max_gaps_pct: 0.1
  max_kurtosis: 10
  skip_collection_on_fail: false

collections:
  - name: crypto
    source: binance
    symbols: [...]
    reliability:
      is_verified: true
      min_data_points: 5000
```

Computed at runtime (not in YAML):

- `last_updated`: timestamp when data/reliability was last checked
- `data_continuity_score`: metric (0.0–1.0 or equivalent)
- `kurtosis`, `n_bars`, `n_params`: diagnostic values

---

## Dependency alignment

| Sub-Task   | Clarification impact |
|------------|----------------------|
| VD-4344    | Defines schema and semantics of `is_verified`, `min_data_points`, `last_updated`, Data Continuity Score |
| VD-4345    | Gatekeeper consumes these fields and thresholds; blocks optimization (and optionally collection) on fail |
| VD-4349    | Provides YAML structure for thresholds (`reliability.*`, `collections[].reliability.*`) |
