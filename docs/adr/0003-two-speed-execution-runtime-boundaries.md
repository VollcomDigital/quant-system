# ADR 0003: Two-Speed Execution Runtime Boundaries

- Status: Accepted
- Owners: Execution, Platform, Research
- Implementation Owner: Execution (mid-frequency), Platform (shared), Native Execution (HFT)
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

## Options Considered

1. Unified execution service with runtime flags
2. Two dedicated engines with shared gateway contracts
3. Multiple execution services split by venue and latency class

## Decision

Adopt **Option 2 — two dedicated engines with shared gateway contracts** as
the binding Phase 0 runtime decision:

- `trading_system/mid_freq_engine/` — cloud-native, Python + model-serving
  services for minute-to-week horizons (Phase 6).
- `trading_system/native/hft_engine/` — Rust/C++ core for sub-millisecond
  execution (Phase 8). Python is excluded from the tick-to-trade critical
  path.
- `trading_system/shared_gateways/` — shared protocol contracts (FIX, binary
  market-data, Web3 adapters) consumed by both engines.

The Python-to-native handoff is explicit:

- Python is allowed everywhere in research, feature generation, model
  training, model serving, mid-frequency OMS/EMS/RMS orchestration, and the
  control-plane web backend.
- Python is forbidden on the HFT critical path once Phase 8 ships.
- The handoff boundary is the gRPC/Kafka/ZeroMQ transport layer defined by
  `shared_lib.transport` and ADR-0001 import conventions.

Rejected alternatives:

- Option 1 (unified service) cannot deliver both sub-millisecond latency and
  cloud-native model serving from a single runtime.
- Option 3 (per-venue service fragmentation) duplicates gateway and risk
  logic and blocks shared OMS/EMS contracts.

## Consequences

- Phases 6, 7, and 8 can each progress on their own timelines as long as
  they honour the shared gateway contracts.
- FPGA exploration remains in scope but is gated behind Phase 8 exit
  criteria.
- Any future "unified execution" proposal must supersede this ADR first.

## Exit Criteria

- runtime ownership boundaries approved (done)
- shared gateway contract scope defined (done: `trading_system/shared_gateways/`)
- Python exclusion boundary for HFT documented (done: above)
