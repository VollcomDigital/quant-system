//! HFT engine fast-inference — ONNX Runtime / TensorRT wrappers.
//!
//! Phase 8 ships the crate skeleton only. Concrete ONNX / TensorRT
//! wrappers land after the Phase 8 replay + benchmark harnesses
//! demonstrate the Python-side contracts can be honoured.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

/// Marker returned by the inference path to confirm the crate is linked.
pub fn fast_inference_marker() -> &'static str {
    "hft_engine.fast_inference"
}
