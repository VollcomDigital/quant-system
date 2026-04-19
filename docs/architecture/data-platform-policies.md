# Data Platform Policies

- Status: Accepted (Phase 2)
- Source ADRs: ADR-0002 (orchestration + immutability), ADR-0003
  (two-speed runtime), ADR-0005 (gateways)
- Applies to: every package that ingests, stores, or serves market data
  and features.

## Purpose

Phase 2 turns the data-platform roadmap items into binding rules so that
Phase 3 (research), Phase 4 (backtest), Phase 6 (mid-frequency trading),
and Phase 8 (HFT) all consume the same foundations. These rules are
tested in `tests/phase_2/test_data_platform_policies_doc.py` and must
not drift silently.

## Ingestion Split (HFT vs Mid-Frequency)

Ingestion diverges along the two-speed boundary:

- **HFT lane** — `trading_system/native/hft_engine/network/` and the
  future PCAP-replay pipeline:
  - native parsers for binary exchange feeds (ITCH/OUCH, FIX-FAST,
    exchange-native binary);
  - L3 order-book reconstruction from raw packets;
  - PCAP replay for deterministic back-tests at the microstructure
    level;
  - storage optimized for very large binary datasets (native formats,
    not Parquet).
- **Mid-frequency lane** — `data_platform/`:
  - Parquet-backed bars, fundamentals, and alternative data;
  - Polars-native query patterns;
  - tolerated latencies are measured in seconds, not microseconds;
  - all raw and normalized datasets flow through `data_platform.storage`
    snapshots so research, backtest, and live consumers get identical
    frames.

The two lanes never cross-pollinate raw storage. HFT binary dumps stay
inside `trading_system/native/`; mid-frequency Parquet datasets stay
inside `data_platform/`. Feature-level bridges (e.g. microstructure-
derived features exposed as columns in the feature store) are allowed
through explicit pipelines.

## Vendor Routing

Broker and exchange APIs are **not** the historical training data
source:

- **Alpaca** and **IBKR** are execution and live-connectivity providers
  (ADR-0005). They are not used as the primary historical training data
  source. Using them for backfills or training data is forbidden.
- **Historical training + backtest data** flows through:
  - **Polygon.io** — equities and options history;
  - **Databento** — microstructure, options, futures;
  - **Tiingo** — fundamentals + daily history;
  - plus the Phase-0 registry entries (yfinance, finnhub, twelvedata,
    alphavantage) for research use, with vendor quality ranked in
    `data_platform.connectors.PROVIDER_REGISTRY`.
- **On-chain data** flows through either The Graph subgraph ingestion
  or our custom EVM ETL, both producing the protocol-normalized event
  schemas in `data_platform.indexing`.

The connector registry records `role` per provider
(`historical` / `live` / `mixed`) so factor code cannot silently
consume a live-only broker endpoint during research.

## Polars Default

Polars is the **default** data-manipulation engine for every new
domain package:

- `data_platform`, `alpha_research`, `backtest_engine`, `ai_agents`,
  `trading_system/mid_freq_engine` — default to Polars.
- pandas is tolerated **only** inside legacy `src/*` code and
  compatibility shims; new domain code must not introduce pandas as a
  dependency.
- `shared_lib.math_utils` stays pure NumPy so it can be consumed by any
  engine (Polars, pandas, or raw NumPy arrays).

## Orchestration Baseline

Apache **Airflow** is the primary orchestrator (ADR-0002). Pipeline
definitions live in `data_platform.pipelines` and are orchestrator-
agnostic; an Airflow translator is a Phase 9 (infrastructure)
deliverable. Prefect and Dagster remain evaluation candidates but must
not be adopted without an ADR superseding ADR-0002.

## Feature Store as Source of Truth

`data_platform.feature_store.FeatureStore` is the single read/write
surface for factor data across research, backtest, and live trading:

- factor definitions are immutable per `(factor_id, version)` —
  `FactorRegistry` enforces this;
- writes require a registered definition; unknown factor or unknown
  version is a `LookupError`;
- reads accept `factor_id`, optional `version`, optional `symbol`, and
  an `[start, end)` window — same API for every caller.

Research notebooks, backtest runners, and live strategies all go
through the same surface. Parallel "research-only" factor stores are
forbidden.

## Enforcement

- Phase 2 invariant suite (`tests/phase_2/`) covers every policy named
  above.
- Changes to this document require an ADR update (ADR-0002 or a
  superseder).
- The Phase 5 `code_reviewer` agent treats deviations — introducing a
  broker as a historical source, importing pandas into a new domain
  package, bypassing the feature store — as blocking review findings.
