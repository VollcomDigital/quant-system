# ADR 0002: Data Platform Orchestration and Immutability

- Status: Proposed
- Owners: Data Platform, Research
- Target phase: Phase 2

## Context

The data platform must ingest TradFi and on-chain data, support reproducible
feature definitions, and guarantee research/backtest/live parity. Workflow
orchestration and storage immutability choices define the operational baseline.

## Decision Drivers

- deterministic replay and backfills
- mixed TradFi and on-chain dependency graphs
- immutable datasets and versioned factor definitions
- feature-store parity across research and live execution
- operational observability

## Options to Evaluate

1. Apache Airflow as the primary orchestrator
2. Prefect for Python-native orchestration ergonomics
3. Dagster for asset-centric dependency management

## Proposed Direction

Use Apache Airflow as the default orchestrator for Phase 2 because it best fits
scheduled ETL, dependency-heavy DAGs, long-running backfills, and shared
platform observability. Store analytical datasets in Apache Parquet and make
Polars the default high-volume transformation engine. Use dbt for declarative
transformation layers where SQL-modeling is a better fit than imperative ETL.

## Questions to Resolve

- Which datasets need immutable snapshots versus rolling partitions?
- Which transformations belong in dbt versus Python/Polars jobs?
- What feature-store backend should be the Phase 2 default?

## Exit Criteria

- orchestrator decision accepted
- immutable storage/versioning policy approved
- Parquet/Polars/dbt usage boundaries documented
