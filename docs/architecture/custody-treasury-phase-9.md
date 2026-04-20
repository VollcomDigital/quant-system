# Custody, Treasury, and Multisig (Phase 9)

- Status: Accepted (Phase 9)
- Source ADRs: ADR-0006 (signing / custody / kill switches),
  ADR-0004 (control plane permissions), ADR-0005 (gateways)
- Applies to: every signer the platform uses — TradFi broker auth,
  Web3 EVM transactions, treasury movements, kill-switch-reset
  signing.

## Purpose

Phase 9 fills in ADR-0006's custody decision with a concrete
signer-permission tiering, a treasury-custody model, and a multisig
requirement for anything that leaves the vault. Bots on the trading
path can sign inside their tier; nothing more.

## Signer Permission Tiers

Three signer roles, each backed by an AWS KMS key (Terraform module
`kms_signing` provisions one key per role):

- **trading_signer** — signs EVM trading transactions only. Cannot
  transfer assets out of the Safe treasury. Cannot disable other KMS
  keys. Used by `trading_system.gateways.web3.Web3Gateway` via the
  `SigningClient` protocol.
- **treasury_signer** — signs treasury moves that the Safe
  smart-contract vault cannot execute unilaterally. Requires an
  `ApprovalRequest(subject="treasury_transfer")` that at least two
  human approvers have decided.
- **kill_switch_signer** — signs the `pause()` / allowance-revoke
  transactions emitted by the Phase 7 DeFi kill-switch helpers. Used
  by `trading_system.kill_switch.PanicPlaybook` and the Phase 9
  hard-kill Lambda.

A signer role may not cover another role's operations; the Phase 9
IAM policies enforce this at the KMS layer.

## Treasury Custody Model

- **Safe smart-contract vault** is the default treasury for on-chain
  capital. A `trading_signer` cannot move funds out of Safe — only a
  multisig threshold (`m-of-n`) can.
- **Institutional MPC custodian** (Fireblocks / Fordefi) is an
  optional institutional deployment tier; it does not replace the
  Safe vault but sits alongside it for counterparty-diversification.
- **Broker-native settlement accounts** (Alpaca / IBKR cash) are
  treated as execution-only balances; redemptions to bank / fiat
  off-ramps require treasury_signer + multisig.

## Human Multisig Requirements

Every asset-moving flow that could drain the platform requires
human multisig:

- **Exchange transfers** (broker withdrawals, Fireblocks / Fordefi
  transfer requests) — 2-of-3 approvers minimum.
- **Treasury transfers out of Safe** — the Safe contract's own
  `m-of-n` owner set (operated out of hardware wallets, not the
  signers above).
- **Fiat off-ramps** — the broker's own multi-approver flow plus a
  Phase 5 `ApprovalDecision(subject="treasury_transfer")` audited on
  our side.

## Withdrawal Policy

- Bots must not initiate unrestricted withdrawals. The
  `trading_signer` role is limited to DEX router + known protocol
  addresses via the `ProtocolDenylist` / allowlist + the Phase 2
  `ABIRegistry` registered function names.
- Any withdrawal larger than the per-asset cap declared in Terraform
  variables requires a Phase 5 approval.
- The Phase 9 out-of-band hard-kill Lambda can disable the
  `trading_signer` KMS key in seconds; it cannot itself initiate
  a withdrawal.

## Nitro Enclave / Enclave-Adjacent Signing

AWS Nitro Enclaves are an **optional** deployment tier for
trading_signer isolation. Phase 9 provisions the signer role behind
a Nitro Enclave-adjacent configuration when the risk budget
justifies it:

- The KMS key is the same; attestation-bound policies restrict use to
  the Enclave.
- The `SigningClient` implementation inside the enclave is a thin
  shim; the `Web3Gateway` does not need to know the difference.

Enclave deployments are opt-in per environment and recorded in the
Phase 9 Terraform env `main.tf`.

## Enforcement

- The Phase 9 `iac-scan` workflow flags Terraform modules that would
  grant trading_signer transfer rights or treasury_signer trading
  rights.
- The Phase 5 `code_reviewer` agent treats any PR that puts a signer
  role name outside the allowed three tiers as a blocking review
  finding.
- The Phase 6 `ApprovalQueue` refuses a treasury_transfer decision
  without the required approver count; audit events record every
  approver id.
- Changes to this document require an ADR update (ADR-0006 or a
  superseder).
