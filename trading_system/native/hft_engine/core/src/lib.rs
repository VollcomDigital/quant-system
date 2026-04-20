//! HFT engine core — lock-free queues, ring buffers, deterministic dispatch.
//!
//! Phase 8 ships the crate skeleton only. The concrete lock-free queue,
//! ring buffer, and event-dispatch implementations land after the
//! replay + latency benchmark harnesses (see
//! docs/architecture/hft-latency-boundary.md).

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

/// Marker returned by the tick-to-trade path to confirm that the engine
/// is wired. Real trading code replaces this in later phases.
pub fn engine_marker() -> &'static str {
    "hft_engine.core"
}
