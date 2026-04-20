# Environment Boundaries and Promotion (Phase 9)

- Status: Accepted (Phase 9)
- Source ADRs: ADR-0002 (orchestration), ADR-0005 (gateways),
  ADR-0006 (signing / custody)
- Applies to: every deployment target Phase 9 Terraform + Kubernetes +
  CI/CD manages.

## Purpose

Phase 9 commits the platform to four explicit environments with
separate credentials, backends, and approval gates. Promotion is
staged (local → research/dev → paper → production/live) and every
step gated by CI + human approval at the last two tiers.

## Environment Ladder

- `local` — developer laptop / container. No live broker credentials.
- `research / dev` — shared research cluster; paper-only, throwaway
  credentials.
- `paper` — full end-to-end paper-trading against sandbox broker
  endpoints + simulated Web3 chains. Same manifests as production.
- `production / live` — live money. Full KMS signing, multisig
  custody, IBKR / Alpaca live credentials.

Each environment has its own Terraform backend
(`infrastructure/terraform/envs/<env>/backend.tf`), its own KMS keys,
and its own S3 / object-storage buckets. Credentials **must not** be
shared across tiers; each env has separate credentials and separate
IAM roles.

## Local

- Broker credentials: none. All brokers replaced by the Phase 7
  `SimulatedGateway`.
- Web3: the Phase 7 `FakeRpcClient` / `FakeSigningClient`.
- Storage: `./tmp/` filesystem, never a shared bucket.
- Web control plane: runs against the in-memory `ApprovalQueue`; no
  real RBAC / session secrets needed.

## Research / Dev

- Broker credentials: vendor-sandbox only (Alpaca paper keys, IBKR
  demo account).
- Web3: public testnets through Alchemy / Infura keys dedicated to
  the dev env.
- Storage: `s3://quant-system-research-dev/…`.
- Auth: short-lived tokens scoped to the research cluster. The
  `operator` role is disabled here.
- Purpose: factor research, model training, agent experiments,
  backtest reviews.

## Paper

- Broker credentials: sandbox + paper accounts (Alpaca paper, IBKR
  DU-prefixed accounts).
- Web3: public testnets only. Production protocol addresses are
  **not** allow-listed at this tier.
- Storage: `s3://quant-system-paper/…`; lifecycle policies in place.
- Auth: same RBAC surface as production so the promotion path is
  identical.
- Purpose: full paper-trading parity, panic playbook drills, IB
  Gateway daily-restart automation rehearsals.

## Production / Live

- Broker credentials: live (Alpaca live, IBKR live). Stored in Vault,
  delivered through KMS-backed short-lived tokens. The browser never
  holds them.
- Web3: mainnet RPC endpoints, trading_signer KMS key, Safe vault.
- Storage: `s3://quant-system-production/…` with Object Lock for
  audit + fill logs.
- Auth: full RBAC (viewer / approver / operator) + approvals required
  for all mutating endpoints. The Phase 9 out-of-band hard-kill
  Lambda has disable rights on the trading_signer KMS key.

## Promotion Gates

Promotion is `workflow_dispatch` from GitHub Actions
(`.github/workflows/staged-deploy.yml`):

- `dev → paper`: CI green + ruff clean + coverage ≥ 80% + Phase 4
  `test_phase_4_exit_criteria` green + Phase 6 `test_phase_6_exit_criteria`
  green.
- `paper → production`: paper-tier IB Gateway restart drill completed
  successfully + panic playbook drill in the last 14 days + on-call
  sign-off + approval_id from a `kill_switch_reset`-class approval.

Every promotion emits an `AuditEvent` with actor + approval_id +
target env.

## Data Boundaries

- No production credential ever reaches a non-production environment.
- No production broker endpoint is addressable from the research /
  dev cluster.
- The Phase 9 recovery workflow for gateway / container failures
  (including the IB Gateway daily restart automation) runs per env
  and never crosses env boundaries.

## Enforcement

- `.github/workflows/iac-scan.yml` scans Terraform + Kubernetes
  overlays for cross-env references.
- `.github/workflows/staged-deploy.yml` uses separate `environment:`
  gates so approvals are required at each step.
- The Phase 5 `code_reviewer` agent treats any PR that hard-codes a
  production URL / ARN / key id outside the production env overlay
  as a blocking review finding.
- Changes to this document require an ADR update (ADR-0002, 0005, or
  0006) plus a matching update to
  `tests/phase_9/test_environment_boundaries_doc.py`.
