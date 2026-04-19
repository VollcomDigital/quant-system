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
