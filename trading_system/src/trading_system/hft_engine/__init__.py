"""Python-side HFT engine contracts.

The actual HFT runtime lives under `trading_system/native/hft_engine/`
(Rust). This Python package ships only the contracts that the
research, backtest, and ops surfaces need to talk about the HFT
engine: model cards, latency benchmarks, and the Python-exclusion
rule.

Phase 0's `docs/architecture/hft-latency-boundary.md` is the
normative source on what belongs here vs under `native/`.
"""

from __future__ import annotations
