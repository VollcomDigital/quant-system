# Python / HFT Latency Boundary

- Status: Accepted (Phase 0)
- Source ADRs: ADR-0003 (two-speed runtime), ADR-0001 (package boundaries)
- Applies to: every package that participates in the trading critical path.

## Purpose

ADR-0003 commits the platform to a two-speed architecture with a sharp
Python/native boundary for HFT. This document turns that decision into an
operational rule that engineers, reviewers, and agents can apply without
re-opening the architecture debate on every PR.

It answers four questions:

1. Where is Python allowed?
2. Where is Python explicitly forbidden?
3. What latency budget forces a strategy across the boundary?
4. What is the handoff contract when a signal must cross it?

## Python-Allowed Zones

Python (3.12+) remains the default language for every non-critical path:

- `alpha_research/` — research notebooks, factor libraries, ML training
  loops, walk-forward evaluation, RL experiments.
- `data_platform/` — connectors, ingestion DAGs (Airflow), dbt transforms,
  feature-store read/write logic, on-chain indexing.
- `backtest_engine/` — simulator, analytics, market mechanics, and
  tear-sheet generation.
- `ai_agents/` — agent runtime, prompt registry, research/review/risk
  agents, memory retrieval.
- `web_control_plane/backend/` — REST APIs, approval workflows, audit,
  operator views, authentication.
- `trading_system/oms/`, `trading_system/ems/`, `trading_system/mid_freq_engine/` —
  mid-frequency order, execution, and portfolio management.
- `trading_system/shared_gateways/` adapters for TradFi brokers (Alpaca,
  IBKR) that operate at mid-frequency latencies.

Latency expectations in these zones sit between 10 ms and several seconds
at the 99th percentile. They are cloud-native and can tolerate Python
runtime characteristics (GIL, GC pauses, cold imports).

## Python-Forbidden Zones

Python is forbidden on the HFT critical path once Phase 8 ships. Explicitly
forbidden surfaces:

- **Market data decoding** of binary exchange feeds (ITCH/OUCH, FIX-FAST,
  exchange-native binary). These live under
  `trading_system/native/hft_engine/network/` and are implemented in Rust
  or C++.
- **Order entry** onto HFT venues. The tick-to-trade path from signal to
  NIC must not cross into Python. This lives under
  `trading_system/native/hft_engine/core/` and `.../network/`.
- **HFT inference**. Model inference on the critical path runs through
  ONNX Runtime, TensorRT, or hand-rolled C++ kernels under
  `trading_system/native/hft_engine/fast_inference/`. Research weights
  must be compiled to one of these targets before live eligibility.
- **FPGA control-plane traffic** once `trading_system/native/hft_engine/fpga/`
  lands. Python may configure the board offline; it may not sit in the
  hot path.

The rule is non-negotiable: **no Python bytecode executes between a
market-data tick arriving at the NIC and an order leaving the NIC in HFT
mode.**

## Latency Budgets

These budgets are normative. A strategy that needs better than the
mid-frequency budget must graduate to the HFT path or be rejected.

| Zone                               | Allowed runtime | p99 budget       | Notes                                              |
|------------------------------------|------------------|------------------|----------------------------------------------------|
| Research / backtest                | Python           | minutes          | Not on any live critical path.                     |
| Mid-frequency OMS/EMS decision     | Python           | ≤ 50 ms          | From signal ingestion to broker submission.        |
| Mid-frequency broker round-trip    | Python           | ≤ 500 ms         | Depends on broker; measured end-to-end.            |
| HFT tick-to-trade                  | Native (Rust/C++)| ≤ 50 microseconds| Co-located, kernel bypass NIC, no Python.          |
| HFT ML inference                   | ONNX / C++       | ≤ 10 microseconds| Compiled artifact only.                            |
| HFT FPGA inference (late stage)    | FPGA             | ≤ 1 microsecond  | Configuration only from Python, offline.           |

Anything requesting a p99 below 10 milliseconds for live execution must
justify it against this table in the associated ADR or design review.

## Handoff Contract

When research produces a signal that must cross the boundary, it crosses
exactly one transport:

- **Mid-frequency → HFT native**: ZeroMQ PUSH or a shared memory ring
  buffer between a Python producer process and a native consumer process
  on the same co-located host. Payloads are protobuf-encoded and carry
  `idempotency_key`, `trace_id`, and a bounded size.
- **Research → production model**: the research side emits a compiled
  artifact (ONNX graph, C++ kernel, or FPGA bitstream) plus a model card.
  No Python pickle crosses into `trading_system/native/`.
- **Native → control plane**: HFT telemetry (p50/p99 latency, fill counts,
  rejects) is published to Kafka from a dedicated native exporter. Python
  consumers read the Kafka topic; they do not open sockets to the native
  core.

Every handoff is versioned via `shared_lib.contracts` and governed by the
Service-to-Service Communication Standards.

## Escalation Rules

A mid-frequency strategy must be escalated to the HFT path when **any** of
the following is true:

1. Its target holding horizon drops below 1 second.
2. Its alpha decay is documented under 50 milliseconds.
3. Its competitors are known to co-locate and it cannot meet the
   mid-frequency 50 ms budget at the 99th percentile.
4. Risk or execution review flags that Python GC or GIL jitter is the
   dominant source of slippage.

Escalation requirements:

- The strategy's inference graph must be convertible to ONNX, compilable
  to a C++ kernel, or representable as FPGA logic. Strategies that cannot
  be compiled are rejected for HFT deployment.
- The strategy registers a new signal contract under
  `shared_lib/contracts/` and a native consumer under
  `trading_system/native/hft_engine/`.
- Deployment is gated by the Phase 8 replay and latency benchmark
  harnesses.

Demotion back to mid-frequency follows the same review: if native jitter
or vendor drift removes the HFT justification, the strategy returns to
Python on the mid-frequency path.

## Enforcement

- Phase 5 `code_reviewer` agent treats any `import` of a Python module
  from `trading_system/native/` source as a blocking review finding.
- Phase 8 CI refuses builds where a Python shim appears in the
  tick-to-trade control flow.
- Phase 0 invariant tests (`tests/phase_0/test_hft_latency_boundary.py`)
  ensure this document remains concrete.
- Any change to the latency budgets requires an ADR update.
