# HFT Engine Interface (Phase 8)

- Status: Accepted (Phase 8)
- Source ADRs: ADR-0003 (two-speed runtime), ADR-0001 (package
  boundaries), ADR-0006 (signing / custody)
- Applies to: `trading_system/native/hft_engine/*` and every Python
  domain package that talks about the HFT path.

## Purpose

Phase 8 turns the Phase-0 HFT latency boundary from a doc into a set
of testable interfaces. The Python-exclusion rule is enforced
statically, the native crate layout is frozen, and every model that
reaches the HFT critical path must have an `HFTModelCard` whose p99
budget has been honoured against a `LatencyReport`.

This doc extends `docs/architecture/hft-latency-boundary.md` (the
normative latency budget table) with the interface-level contracts
Phase 8 introduced.

## Python Exclusion Boundary

Python is forbidden on the tick-to-trade HFT critical path:

- **No `.py` files** may live under `trading_system/native/**`.
  Enforced by `tests/phase_8/test_native_scaffold.py`.
- **No Python domain package may `import trading_system.native.*`**
  at runtime. Enforced by
  `tests/phase_8/test_python_exclusion_and_interface.py`.
- Python **is** allowed, and expected, in `trading_system.hft_engine`
  (Python-side contracts: model cards, benchmark harness). Python
  here is for test + paper-trading scaffolding only; it never
  participates in the live tick-to-trade loop.

See also `docs/architecture/hft-latency-boundary.md` (ADR-0003) for
the latency budget table that drives this exclusion.

## Native Crate Boundaries

Three mandatory crates under `trading_system/native/hft_engine/`:

- `core/` — lock-free queues, ring buffers, deterministic event
  dispatch, memory layout guarantees. Pure Rust, no FFI.
- `network/` — kernel bypass (DPDK / io_uring / AF_XDP / NIC-specific
  integrations), binary market-data parsers (ITCH/OUCH, FIX-FAST,
  exchange-binary), order-entry path.
- `fast_inference/` — ONNX Runtime / TensorRT / hand-written C++
  kernel wrappers. Talks only to `core` via shared ring buffers;
  never to Python.

One reserved subtree:

- `fpga/` — FPGA bitstream sources. Only filled in after the software
  baselines (core + network + fast_inference) exist and show a
  measurable latency headroom that justifies the hardware track.

Each crate declares its own `Cargo.toml` and is wired into the
workspace `Cargo.toml` under `trading_system/native/Cargo.toml`. The
workspace pins a latency-critical `[profile.release]` (`lto = "fat"`,
`codegen-units = 1`, `opt-level = 3`, `panic = "abort"`, `strip =
true`).

## Model Compile Requirement

A research model is **not** live-eligible on the HFT path until it
has a Python-side `HFTModelCard` with:

- `compiled_target ∈ {onnx, cpp_kernel, fpga}` (raw PyTorch / sklearn
  weights are refused at construction).
- Numeric `p99_inference_budget_us` (must be > 0).
- Explicit `input_shape` / `output_shape` tuples.
- A `training_data_snapshot_id` so weights trace back to a
  reproducible Phase 2 `SnapshotIndex` snapshot.

`is_live_eligible(card, measured_p99_us)` enforces the budget gate
by returning a `shared_lib.contracts.ValidationResult`.

## Replay + Benchmark Prerequisites

No HFT code reaches live trading until the Phase 8 harnesses show
the declared budget is met:

- `summarise_latency(samples_us)` → `LatencyReport(n, p50, p95, p99,
  max)`.
- `LatencyBudget(p50_us, p95_us, p99_us)` rejects non-monotonic
  limits.
- `enforce_budget(report, budget)` returns a `ValidationResult`
  suitable for CI + the approval queue.

Both replay (Phase 7 `shared_gateways.replay`) and benchmark harnesses
must pass against a co-located simulated venue before any production
deployment approval is granted.

## Interfaces Between HFT Core and Subsystems

The tick-to-trade path crosses exactly these boundaries, and each
boundary is a concrete interface rather than a loose function call:

- **Risk controls** — Phase 6 `trading_system.rms` produces the
  `RiskLimits` + `TRADING_HALTED` flag. HFT core consumes a flat
  snapshot of those limits at startup; live mutations require a
  restart (no hot reload on the critical path).
- **Market data decoding** — `hft_engine.network` parses frames from
  the NIC and hands raw `ring_buffer` slots to `hft_engine.core`.
- **Order entry** — `hft_engine.core` writes to a second ring buffer
  that `hft_engine.network` drains to the NIC.
- **Model inference** — `hft_engine.fast_inference` reads features
  from `core`'s ring buffer, runs an ONNX / TensorRT model, and
  writes scores back into a results ring buffer.
- **Telemetry** — `hft_engine.core` emits latency and fill events to
  a Kafka topic consumed by the Phase 5 HFT Latency Agent (which
  runs in Python, off the critical path).

None of these interfaces permit Python on the hot path. Python is
allowed **only** for:

- The Python-side `HFTModelCard` + benchmark harness.
- Offline FPGA configuration scripts.
- The Phase 5 HFT Latency Agent consuming Kafka telemetry.

## Enforcement

- `tests/phase_8/` enforces the native scaffold, model-card contract,
  benchmark behaviour, and Python-exclusion rule.
- Phase 5 `code_reviewer` treats any PR that puts a Python handler on
  the HFT critical path as a blocking review finding.
- `deny(unsafe_op_in_unsafe_fn)` + `warn(missing_docs)` are set in
  every native crate; CI will build-gate these when the Phase 9 Rust
  toolchain lands.
- Changes to this document require an ADR update (ADR-0003 or a
  superseder).
