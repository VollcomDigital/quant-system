## Lessons

### 2026-04-19

- When the user wants roadmap or planning guidance incorporated, prefer updating `tasks/todo.md` directly instead of creating a parallel standalone planning/spec document unless they explicitly ask for a separate doc.

### Phase 0 — Architecture Baseline and Repo Restructure

- **Test-first on documentation is useful, not theatrical.** Enforcing
  section headings, required concepts, numeric latency budgets, and
  forbidden phrases in ADRs and architecture docs caught multiple cases
  where prose drifted away from the intended contract (e.g. "kernel-bypass"
  vs "kernel bypass"). The tests now act as a lint layer for architectural
  prose.
- **Regex for Markdown headings needs `\s*$`, not `\b`.** A pattern like
  `^##\s+Decision\b` silently matches `## Decision Drivers`. Require an
  explicit end-of-line anchor when a heading must be exact.
- **Phase 0 scaffolding tests must not depend on legacy runtime.** The
  existing repo tests require `pandas`, `pyarrow`, `requests`, and more.
  Isolating Phase 0 invariants under `tests/phase_0/` with its own
  `conftest.py` that does not import `src.*` kept the new gate runnable on
  a minimal toolchain and on CI runners that have not yet installed
  Phase 2+ deps.
- **Empty directories must carry `.gitkeep`.** Git does not track empty
  dirs, so `infrastructure/`, `web_control_plane/`, and
  `trading_system/native/...` would have vanished on push without marker
  files. The Phase 0 invariant tests caught the skeleton locally, but
  remote CI would have failed against the same tree until `.gitkeep` was
  added.
- **Legacy code was not touched.** The compatibility-facade rule ("no
  domain package imports `src.*`") is now enforced statically, which lets
  future phases refactor `src/` confidently without breaking new
  packages. The legacy CLI (`src/main.py`), root `pyproject.toml`, and
  existing tests were left untouched by Phase 0 work.
- **ADR status as a first-class test target.** Moving ADR-0001 and
  ADR-0003 from "Proposed" to "Accepted" with binding `## Decision`
  sections made subsequent tasks (naming, service transport, HFT
  boundary) cleanly derivable from a fixed reference. Keeping
  ADR-0002/4/5/6 as "Proposed" with explicit Implementation Owners
  satisfies the delivery rule for later phases.
- **Coverage gate deferred to Phase 1+.** Phase 0 introduced no runtime
  code, so the >80% coverage rule applies to executable code from
  Phase 1 onward. This is recorded here so the expectation for
  Phase 1 tests is unambiguous.

### Phase 1 — Shared Contracts, Telemetry, and Core Utilities

- **Phase 0 invariants that were snapshots, not perpetual rules, need to
  be retired explicitly when the next phase starts.** Phase 0's
  "shared_lib must be scaffold-only" check blocked the very Phase 1 work
  the roadmap describes. Relaxed to "package root still exists" once
  Phase 1 legitimately added runtime code.
- **shared_lib.logging idempotency is subtle under pytest.** pytest
  fixtures that clear `logger.handlers` between tests defeat a naive
  module-level cache keyed by service name. The working pattern is to
  cache by *state* (`handlers empty -> rebuild`) rather than by name
  alone.
- **Fallback tracer must stay quiet.** An early implementation of
  `start_span` emitted its own span-start/end log lines. That broke
  tests that counted emitted lines and — more importantly — added
  noise to every caller's log stream. The fallback must bind trace
  context without writing log records.
- **Pydantic v2 + `from __future__ import annotations`.** Returning a
  self-referencing model type from a `model_validator` (`-> "Bar"`)
  works with stringified annotations; ruff rewrites these to bare class
  names, which is fine because pydantic resolves the class at model
  build time.
- **Coverage ≥80% is achievable without heroics** by covering every
  branch in `if/else` paths (size checks, zero-division guards, enum
  closure). 96–100% on every Phase 1 module came from targeted edge
  tests rather than duplicating happy-path coverage.
- **Contract-test lint-layer catches prose drift**: Phase 0's approach
  of lint-like tests applied to schemas too — e.g., explicitly testing
  `weights must sum to 1.0`, `confidence must be in [0,1]`,
  `HealthStatus.ok == all(checks.values())`. These are cheap to write
  and turn documentation invariants into executable gates.

### Phase 2 — Data Platform Extraction

- **Shared fixtures must be scoped per phase.** Phase 0's `repo_root`
  fixture lives in `tests/phase_0/conftest.py` and is not visible to
  other phase dirs. Each phase conftest must re-declare the fixtures
  it needs. Cleaner than a top-level shared conftest because it keeps
  phase invariant tests self-contained and runnable in isolation.
- **Contracts over implementations for Phase 2.** The roadmap names
  eight concrete vendors (Alpaca, Polygon, Tiingo, …), but Phase 2's
  actual deliverable is the *contract* surface the vendors will plug
  into. `PROVIDER_REGISTRY` records the eight names and their role
  (historical/live/mixed); the live implementations stay behind the
  `src/*` facade until Phase 10. This keeps the new package free of
  vendor-specific deps while giving Phase 6+ a typed surface.
- **Orchestrator-agnostic DAG objects keep Airflow out of the contract.**
  `data_platform.pipelines.DAG` doesn't `import airflow`. An Airflow
  translator is a Phase 9 concern. This avoids pulling ~40 transitive
  deps into the data platform core.
- **`zip(strict=True)` is risky when the slice lengths legitimately
  differ by one.** `zip(seq, seq[1:], strict=True)` raises on pairs of
  (N, N-1) lists. Use `strict=False` for pairwise windows; keep
  `strict=True` only when both iterables should end together.
- **Feature-store reads must fail closed on version ambiguity.** When
  multiple versions exist for a factor_id, `read(factor_id=x)` without
  a version is ambiguous and could silently mix cohorts. The contract
  raises `ValueError` rather than guessing. Research notebooks can
  still ask for the latest version via `registry.list(factor_id)[-1]`.
- **On-chain event schemas are content-addressed by protocol+version.**
  The ABI registry is intentionally append-only per `(protocol,
  version)` key so a deployed contract's decoder cannot be retroactively
  rewritten. New deployments require a new version string.

### Phase 3 — Alpha Research Workspace

- **Factor metadata should require a `leakage_review` string by contract,
  not just a boolean.** Forcing reviewers to write *why* the factor is
  leakage-safe prevents rubber-stamping and gives the code_reviewer
  agent a reliable surface to inspect during promotion.
- **Registries returning `ValidationResult` rather than raising are
  friendlier to UI/approval workflows.** `promote_factor` /
  `promote_model` return a single structured result that the Phase 5
  web control plane can display. Raising would force every caller into
  try/except around a business outcome.
- **Stage-skip is a separate invariant from stage-validity.**
  `candidate → promoted` is not a type error but it is a governance
  error. Modelling both as distinct guards (adjacent-stage rule +
  per-transition checks) keeps the promotion logic readable.
- **One-production-version-per-model is the moral equivalent of a
  kill-switch invariant.** `ModelRegistry.transition` refuses a second
  version moving to `production` while another is live so there is no
  implicit shadow-deployment race.
- **Walk-forward and CV window generators belong in the research
  package, not the backtest engine.** They are pure-Python and don't
  depend on simulator state, so research and backtest both consume
  them. Keeps the Phase 4 engine dependency-light and lets CI run
  CV-related tests without booting the simulator.
- **Notebooks are the highest-risk code path for look-ahead leakage
  and secret leakage.** Writing the governance doc as a normative
  contract (with CI-testable rules) turned the usual "don't commit
  notebook outputs" custom into an enforceable policy. The Phase 3
  static test that domain code cannot import from
  `alpha_research.notebooks` closes the corresponding type-level leak.
