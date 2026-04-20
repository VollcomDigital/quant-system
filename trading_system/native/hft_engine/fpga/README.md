# HFT FPGA subtree

Reserved for FPGA bitstream sources and Verilog/VHDL projects. Phase 8
does **not** ship FPGA implementations: software baselines (Rust/C++
core + ONNX fast_inference) must exist first (see
`docs/architecture/hft-latency-boundary.md`).

A real FPGA project will land here once the software replay and
latency-benchmark harnesses show a measurable headroom that justifies
the hardware track.
