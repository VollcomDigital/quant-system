# ADR 0006: Execution Signing, Custody, and Kill Switches

## Status

Proposed

## Context

The target platform spans:

- TradFi brokers and broker-like APIs
- Web3 RPC providers and smart-contract execution
- autonomous AI model outputs
- institutional requirements for custody separation and emergency controls

The architecture must decide:

- how transactions are signed
- where treasury assets are held
- what automated and human kill-switch powers exist
- how the system fails closed during model drift, hallucination, broker outages, or on-chain exploit risk

## Decision

Decision required on:

1. default programmatic signing model:
   - AWS KMS-backed signing
   - HashiCorp Vault-backed signing
   - MPC custody provider integration
2. treasury custody model:
   - Safe-based smart-contract treasury
   - institutional MPC custodian
   - broker-native settlement accounts where applicable
3. allowed execution signer permissions:
   - trading-only signer
   - protocol-restricted signer
   - capped notional/volume signer
4. layered kill-switch architecture:
   - Layer 1: pre-generation AI guardrails
   - Layer 2: deterministic pre-trade risk engine
   - Layer 3: execution-phase automated panic-button actions
   - Layer 4: DeFi-specific pause and allowance-revocation controls
   - Layer 5: out-of-band IAM and infrastructure isolation

## Options Considered

### Option A: AWS KMS signing + Safe treasury + human multisig withdrawals

Pros:

- strong institutional baseline
- compatible with automated execution
- treasury and bot signer are separated

Cons:

- requires custom transaction-building/signing integration
- DeFi protocol coverage must be engineered internally

### Option B: MPC provider for signing and treasury

Pros:

- strong operational controls and approvals
- simpler external due-diligence story

Cons:

- higher dependency on vendor policy engines and APIs
- may increase cost and latency

### Option C: mixed model

Pros:

- KMS for fast execution
- Safe or MPC for treasury
- adaptable by strategy and asset class

Cons:

- more complex governance and implementation

## Consequences

If accepted, follow-on design work must define:

- permitted signer scopes by venue and strategy
- emergency shutdown state machine
- protocol denylist and approval revocation flows
- reconciliation and restart flows after kill-switch activation

## Follow-Up Tasks

- define exact signer and treasury boundaries
- define panic-button and recovery sequence diagrams
- define human escalation paths and approvals
- define audit logging requirements for every kill-switch action
