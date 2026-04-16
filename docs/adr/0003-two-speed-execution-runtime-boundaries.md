# ADR 0003: Two-Speed Execution Runtime Boundaries

- Status: Proposed
- Owners: Execution, Platform, Research
- Target phase: Phase 0 / Phase 6 / Phase 8

## Context

The repository must support both mid-frequency and HFT workloads. These paths
share data and research foundations but diverge sharply on runtime, latency,
deployment, and implementation language.

## Decision Drivers

- deterministic low-latency HFT path
- cloud-native mid-frequency path
- strict Python-to-native handoff boundaries
- reusable shared gateway contracts
- independent deployment models

## Options to Evaluate

1. Unified execution service with runtime flags
2. Two dedicated engines with shared gateway contracts
3. Multiple execution services split by venue and latency class

## Proposed Direction

Adopt a two-speed architecture:

- `trading_system/mid_freq_engine/` for minute-to-week execution
- `trading_system/hft_engine/` for sub-millisecond execution
- `trading_system/shared_gateways/` for common protocol contracts

Python remains valid in research and model-serving paths, but is removed from
the HFT critical path. Native services own HFT event loops, order entry, and
network-path control.

## Questions to Resolve

- Which control-plane operations remain shared?
- What is the standard signal handoff contract from Python to native services?
- Which latency budgets define when a strategy must move from mid-frequency to
  HFT infrastructure?

## Exit Criteria

- runtime ownership boundaries approved
- shared gateway contract scope defined
- Python exclusion boundary for HFT documented
