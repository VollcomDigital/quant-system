# VD-4344 — Define "Reliability" Metadata Schema

**Status:** In Progress  
**Parent Story:** VD-3919 — Guardrail – Optimization Only on Reliable Collections

---

## Clarification Questions & Answers

### Q1: Are `is_verified`, `min_data_points`, `last_updated` intended as flags set on the Collection YAML level for approval?

**Yes.** These are per-collection reliability metadata fields declared in the collection YAML under a `reliability:` block nested inside each collection entry.

```yaml
collections:
  - name: crypto
    source: binance
    symbols: ["BTC/USDT", "ETH/USDT"]
    reliability:
      is_verified: true
      min_data_points: 3000
      last_updated: "2026-02-25T00:00:00"
```

Each field serves a distinct purpose:

| Field | Type | Purpose |
|---|---|---|
| `is_verified` | `bool` (default `true`) | Manual approval flag. When `false`, the collection is treated as unreliable and skipped if `require_verified` is enabled globally. Maps to the "Status" row in VD-3919's guardrail table. |
| `min_data_points` | `int \| null` | Per-collection override for the minimum bar count. Allows asset-class tuning (e.g., 3000 for crypto, 1000 for bonds). Falls back to the global `reliability.min_data_points`. |
| `last_updated` | `datetime \| null` | ISO-8601 timestamp of the last human review or automated quality check. Informational — enables staleness detection in dashboards or future automated checks. |
| `min_continuity_score` | `float \| null` | Per-collection override for the Data Continuity Score threshold (0.0–1.0). Falls back to the global `reliability.min_continuity_score`. |

These belong at collection-level because different asset classes have fundamentally different reliability profiles. Crypto markets trade 24/7 (expecting near-perfect continuity), while bonds may have legitimate gaps due to thin trading.

**Implementation:** `CollectionReliability` Pydantic model in `src/reliability/schema.py`, parsed from YAML in `src/config.py`.

---

### Q2: Should `min_data_points` in example.yaml skip an entire backtest run if too few candles are available?

**Yes — but at the symbol level, not the collection level.** The check is per (symbol, timeframe) pair.

There are now **two tiers** of bar-count gating:

| Tier | Field | Scope | Behavior |
|---|---|---|---|
| **Data quality gate** | `reliability.min_data_points` | Per symbol | Skips the *entire backtest* (including fixed-param evaluation) if the symbol has fewer bars than the threshold. |
| **Optimization gate** | `param_min_bars` / `param_dof_multiplier` | Per param search | Skips *only the parameter search* (Optuna, grid) and falls back to default params. Already implemented in `runner.py:517-538`. |

The rationale: if a symbol has too few candles, even a fixed-parameter backtest is statistically meaningless. The existing `param_min_bars` guard only skips optimization but still evaluates the default params — which can produce misleading results on tiny datasets.

**Separation of concerns:**
- `min_data_points` = "Is there enough data to evaluate *at all*?"
- `param_min_bars` = "Is there enough data to *search parameters*?"

The effective threshold for a given collection is:
```
effective_min_data_points = collection.reliability.min_data_points
                           ?? config.reliability.min_data_points
                           ?? 2000  (hardcoded default)
```

---

### Q3: Is the Data Continuity Score purely a metric for evaluating OHLC data quality?

**Primarily yes — it is a computed metric.** The Data Continuity Score is defined as:

```
score = present_bars / expected_bars
```

where `expected_bars` is derived from a regular frequency grid spanning the data's date range, adjusted for market calendars (business days for equities, full calendar for crypto).

It quantifies:
- **Missing bars** — gaps in the time series
- **Structural gaps** — extended periods with no data (e.g., exchange outage, delisted asset)
- **Largest gap** — the longest contiguous gap in bars

The score is encapsulated in `SymbolContinuityReport` (Pydantic model) and can be stored, logged, or displayed in dashboards independently of any enforcement decision.

**However, it is not *only* a passive metric.** Per VD-3919's guardrail table ("`> 0.1%` missing bars → Fail"), the score is compared against a configurable threshold (`min_continuity_score`) to gate execution. This brings us to Q4.

**Implementation:** `compute_continuity_score()` in `src/reliability/continuity.py`.

---

### Q4: Should there be a YAML-defined threshold that determines when a collection is skipped?

**Yes.** This is the core of VD-4349 ("Configuration-Based Thresholds") and integral to VD-4344's schema.

Thresholds live at two levels:

#### Global (YAML root level)

```yaml
reliability:
  min_data_points: 2000
  min_continuity_score: 0.999
  max_gap_percentage: 0.001
  max_kurtosis: 10.0
  require_verified: true
```

#### Per-collection override

```yaml
collections:
  - name: crypto
    reliability:
      min_data_points: 3000
      min_continuity_score: 0.9999  # stricter for 24/7 market
```

The resolution order for each threshold is:
1. Per-collection value (if set and not `null`)
2. Global `reliability:` value
3. Hardcoded default

This cleanly separates:
- **Metric** — Data Continuity Score, kurtosis, bar count (computed, informational, storable)
- **Threshold** — `min_continuity_score`, `min_data_points`, `max_kurtosis` (configurable, enforcement)

When a symbol/collection fails a threshold check, the backtest is skipped with a structured log message (VD-4348):
```
"Backtest Skipped: Collection 'crypto' symbol 'DOGE/USDT' failed continuity check (score=0.982, threshold=0.999)"
```

**Implementation:** `ReliabilityThresholds` Pydantic model in `src/reliability/schema.py`, parsed from YAML in `src/config.py`.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────┐
│                     YAML Config                         │
│                                                         │
│  reliability:              # Global thresholds          │
│    min_data_points: 2000                                │
│    min_continuity_score: 0.999                          │
│    ...                                                  │
│                                                         │
│  collections:                                           │
│    - name: crypto                                       │
│      reliability:          # Per-collection overrides   │
│        is_verified: true                                │
│        min_data_points: 3000                            │
│        min_continuity_score: 0.9999                     │
│        last_updated: "2026-02-25"                       │
└────────────────────┬────────────────────────────────────┘
                     │ load_config()
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Config + CollectionConfig                   │
│                                                         │
│  Config.reliability: ReliabilityThresholds              │
│  CollectionConfig.reliability: CollectionReliability     │
└────────────────────┬────────────────────────────────────┘
                     │ run_all()
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Pre-Execution Gate (future VD-4345)         │
│                                                         │
│  1. Check is_verified (if require_verified=true)        │
│  2. Check bar count ≥ effective min_data_points         │
│  3. Compute continuity score → SymbolContinuityReport   │
│  4. Check score ≥ effective min_continuity_score        │
│  5. (Future) Check kurtosis ≤ max_kurtosis             │
│                                                         │
│  → PASS: proceed to backtest / optimization             │
│  → FAIL: skip with structured log + failure record      │
└─────────────────────────────────────────────────────────┘
```

---

## File Changes

| File | Change |
|---|---|
| `src/reliability/__init__.py` | New module entry point |
| `src/reliability/schema.py` | `CollectionReliability`, `ReliabilityThresholds`, `SymbolContinuityReport` Pydantic models |
| `src/reliability/continuity.py` | `compute_continuity_score()` implementation |
| `src/config.py` | Added `reliability` field to `CollectionConfig` and `Config`; new YAML parsing helpers |
| `config/example.yaml` | Added `reliability:` blocks (global + per-collection examples) |
| `tests/test_reliability.py` | Unit tests for schema, continuity score, config parsing |

---

## Relationship to Other Sub-Tasks

| Sub-Task | Relationship |
|---|---|
| **VD-4345** (Gatekeeper) | Consumes the metadata and thresholds defined here to gate optimization. The `min_data_points` and `min_continuity_score` fields become inputs to the gatekeeper logic. |
| **VD-4346** (Stationarity) | Will add computed metrics (ADF p-value, regime flags) alongside the continuity score in `SymbolContinuityReport` or a sibling model. |
| **VD-4347** (Outlier Detection) | May add `outlier_percentage` to the continuity report; uses `max_kurtosis` from thresholds. |
| **VD-4348** (Logging) | Consumes skip reasons and `SymbolContinuityReport` to produce structured rejection messages. |
| **VD-4349** (Config Thresholds) | The `ReliabilityThresholds` model and per-collection overrides directly implement this sub-task. |
