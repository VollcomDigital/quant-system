# ADR 0005: TradFi and Web3 Gateway Architecture

- Status: Proposed
- Owners: Execution, Platform, Crypto
- Target phase: Phase 7

## Context

The execution stack must support broker APIs and on-chain transaction execution.
These are not variations of the same gateway problem: TradFi requires sessioned
broker connectivity and reconciliation, while Web3 requires transaction
construction, signing, ABI awareness, and mempool-aware behavior.

## Decision Drivers

- IBKR and Alpaca operational differences
- local IB Gateway runtime requirements
- Web3 transaction lifecycle complexity
- ABI/version control needs
- paper-trading parity across both paradigms

## Options to Evaluate

1. One generic gateway abstraction with adapter-specific branching
2. Separate TradFi and Web3 gateway families with shared contracts only where
   possible
3. Venue-specific services without a shared gateway layer

## Proposed Direction

Use two gateway families:

- TradFi gateways for IBKR and Alpaca
- Web3/DeFi gateways for RPC-driven execution via Alchemy/Infura and indexed
  data access via The Graph

IBKR should run through a local IB Gateway container, managed with IBC/IB
Controller automation. Short-term Python control-plane integration may use
`ib_insync`, but the long-term contract should remain transport-agnostic.

Web3 gateways should include a version-controlled ABI registry, transaction
simulation, gas estimation, and broadcast lifecycle handling.

## Questions to Resolve

- Which paper-trading semantics must be identical across broker and chain paths?
- Which DeFi protocols belong in the initial allowlist?
- Which gateway operations are synchronous versus queued?

## Exit Criteria

- TradFi vs Web3 split accepted
- IB Gateway operational model approved
- ABI registry and on-chain lifecycle boundaries documented
