//! HFT engine network — kernel-bypass + NIC-specific integrations.
//!
//! Phase 8 ships the crate skeleton only. The concrete ITCH/OUCH,
//! FIX-FAST, and exchange-binary parsers land alongside replay
//! harnesses in a later infrastructure phase.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

/// Marker returned by the market-data path to confirm the crate is linked.
pub fn network_marker() -> &'static str {
    "hft_engine.network"
}
