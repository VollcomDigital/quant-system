# Quant System Unified Roadmap

This roadmap consolidates all prior findings from:
- Repository audit (architecture, performance, security, DX/CI-CD, market positioning)
- Maintenance audit (pre-commit, coverage gate, LGTM/Kubernetes, Dependabot/Sonar posture)
- Feature-gap assessment (tier-1 competitor capabilities and white-space opportunities)
- Trading execution expansion plan (Freqtrade + NautilusTrader with strict paper/live separation)
- Awesome-quant alignment plan (missing ecosystem capabilities to adopt)

---

## 1) Consolidated Findings (What is missing)

### A. Critical engineering and platform gaps
- No production OMS/RMS stack (risk-first execution is not implemented).
- No strict paper/live segregation for configs, credentials, storage, and runtime.
- Execution path is absent (backtesting is strong; trading runtime is missing).
- Backtest runner is monolithic and serial in key paths.
- Cache efficiency bottlenecks:
  - repeated data fetch per strategy/symbol/timeframe
  - signal generation before cache hit check
  - frequent SQLite connect/commit in hot loops
- Coverage gate exists, but scope is narrow (not full `src` risk surface).
- Security hardening gaps:
  - possible sensitive URL/error leakage into logs/artifacts
  - dynamic external strategy imports with weak trust boundary controls
  - dashboard has no auth

### B. DevOps and observability gaps
- Dependabot is configured; Sonar/SonarCloud is not configured.
- No explicit in-repo security scan workflows (CodeQL/Gitleaks/Trivy not present as workflow files).
- No Kubernetes/Helm deployment manifests in repository.
- No LGTM deployment evidence (Grafana Alloy DaemonSet, Tempo/Loki/Prometheus-Mimir wiring).
- Release pipeline lacks SBOM, image scan, and signing/provenance steps.

### C. Product and quant capability gaps
- Portfolio construction layer is missing (allocation and optimization beyond per-strategy backtests).
- No continuous alpha-decay monitoring and auto de-weighting lifecycle.
- No regime-aware allocator.
- No execution-quality feedback loop (slippage/fill analytics feeding strategy selection).
- No strong experiment lineage/model governance comparable to tier-1 platforms.

---

## 2) Prioritization Model

### Priority scale
- **Very High**: Required to safely enable live trading.
- **High**: Required to achieve robust institutional baseline.
- **Medium**: Competitive advantages that materially improve risk-adjusted returns.
- **Low/Lowest**: Optional differentiators after core maturity.

### Category labels
- **Must Have**: Required for production-readiness and controlled capital deployment.
- **Nice to Have**: Strong improvements after baseline safety and reliability.
- **Optional**: Long-horizon strategic improvements.

---

## 3) Unified Phased Roadmap

## Phase 1 - Critical Fixes and Safety Foundation (Immediate, 0-30 days)

### Must Have / Very High
- [ ] Implement strict paper/live separation:
  - separate configs, credentials, databases, logs, and deployment services
  - runtime mode guards that block cross-environment credential misuse
- [ ] Build RMS core:
  - position sizing, exposure limits, pending-order limits, global drawdown kill-switch
- [ ] Build OMS core:
  - canonical order state machine and idempotent order handling
  - broker/exchange reconciliation as source of truth
- [ ] Fix hot-path performance regressions in backtest runner:
  - fetch once per `(collection, symbol, timeframe)`
  - move cache lookup before `generate_signals`
  - reduce SQLite connect/commit churn and add needed indexes
- [ ] Security hardening:
  - redact tokens/query params from HTTP logs
  - avoid persisting raw exception strings in run artifacts
  - protect dashboard endpoints (authn/authz) and remove sensitive path leakage

### Must Have / High
- [ ] Add explicit security workflows:
  - CodeQL
  - secret scanning (Gitleaks or GitHub native integration workflow)
  - dependency vulnerability checks for Python dependencies
- [ ] Container/release hardening:
  - pin base image digest
  - adopt multi-stage runtime (remove build tools from runtime image)
  - add SBOM + image scanning + signing/provenance

### Phase 1 success criteria
- Live mode cannot start without safety controls.
- No known credential leakage vectors in logs/artifacts.
- Backtest runtime shows measurable speedup from cache/data-path fixes.

---

## Phase 2 - Industry Standard Trading Platform (30-60 days)

### Must Have / Very High
- [ ] Implement Freqtrade execution adapter (paper first, then controlled live).
- [ ] Add transaction cost analytics (slippage/fill latency/adverse selection) in post-trade pipeline.

### Must Have / High
- [ ] Implement NautilusTrader adapter behind same OMS/RMS contracts.
- [ ] Expand test and quality gates:
  - broaden coverage target to full critical modules
  - add typing gate (mypy/pyright)
  - increase integration tests for order lifecycle and reconciliation
- [ ] Establish production observability baseline:
  - structured logs with trace IDs
  - RED metrics for execution services
  - trace spans for signal -> risk -> order -> fill pipeline

### Nice to Have / Medium
- [ ] Add deployment manifests (Kubernetes/Helm) for execution services and dashboard APIs.
- [ ] Add initial LGTM stack manifests and wiring (Tempo/Loki/Prometheus endpoint integration).

### Phase 2 success criteria
- End-to-end paper trading is stable across at least one adapter.
- OMS/RMS risk controls and reconciliation pass fault-injection tests.
- CI/CD blocks unsafe or low-quality changes by default.

---

## Phase 3 - Outperforming the 99% (60-120 days)

### Must Have / High
- [ ] Add portfolio optimization service:
  - PyPortfolioOpt baseline methods (HRP, min-vol, max-sharpe)
  - Riskfolio-Lib advanced risk objectives (CVaR/drawdown-aware allocation)
- [ ] Add alpha-decay monitor with policy-based auto de-weight/quarantine.
- [ ] Add regime detection and regime-aware allocator.

### Nice to Have / Medium
- [ ] Add walk-forward and promotion gates:
  - mandatory paper burn-in before live promotion
  - drift/stability checks as release criteria
- [ ] Add richer performance analytics (tear sheets, rolling risk attribution).
- [ ] Add vectorized research lane (vectorbt-style parameter sweeps) for faster experimentation.

### Phase 3 success criteria
- Portfolio-level allocation drives deployment decisions.
- Decaying strategies are automatically constrained without manual intervention.
- Strategy promotion process is measurable, repeatable, and auditable.

---

## Phase 4 - 1% Visionary Capabilities (120+ days)

### Nice to Have / Medium
- [ ] Self-healing data pipeline:
  - provider failover and gap backfill with data confidence scoring
- [ ] Autonomous anomaly remediation:
  - policy-constrained auto responses for execution/data incidents
- [ ] Cross-venue smart order routing.

### Optional / Low-Lowest
- [ ] Meta-learning ensemble allocator.
- [ ] LLM research copilot with strict governance guardrails.
- [ ] Enterprise ontology/lineage-style decision graph for full strategy lifecycle explainability.

### Phase 4 success criteria
- Autonomous controls reduce incident recovery time and live alpha leakage.
- Decision automation remains explainable and policy-compliant.

---

## 4) Feature Classification Snapshot

### Must Have
- Paper/live isolation
- RMS + OMS + reconciliation
- Freqtrade + Nautilus adapters
- Core security hardening and scanning workflows
- Hot-path performance and cache redesign
- Baseline observability

### Nice to Have
- Portfolio optimizer (PyPortfolioOpt/Riskfolio-Lib)
- Regime-aware allocation
- Alpha-decay automation
- Walk-forward promotion gates
- Kubernetes/Helm + LGTM operationalization

### Optional
- Smart order routing
- Meta-learning allocators
- LLM research copilot

---

## 5) Execution Order (Strict)

1. Safety and trust boundary first (paper/live + RMS/OMS + leakage fixes).
2. First execution engine (Freqtrade), then second (NautilusTrader).
3. Portfolio intelligence (optimizer + decay + regime).
4. Scale and autonomy enhancements (self-healing, remediation, routing).

---

## 6) Governance Notes

- No live deployment is allowed until Phase 1 is complete and validated.
- All live-trading changes must include:
  - integration tests for RMS/OMS checks
  - reconciliation tests
  - rollback/fail-safe procedures
- Security and observability controls are part of the definition of done, not post-release tasks.

