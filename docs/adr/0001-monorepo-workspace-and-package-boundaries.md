# ADR 0001: Monorepo Workspace and Package Boundaries

- Status: Accepted
- Owners: Platform, Research, Execution
- Target phase: Phase 0
- Last updated: Phase 0 execution

## Context

The current repository is a single Python package under `src/`. The target
state is a multi-domain monorepo with shared foundations and divergent runtime
paths. The first architectural decision is how package boundaries and build
tooling should work without breaking the current CLI.

## Decision Drivers

- low-friction migration from `src/`
- clean internal package ownership
- support for Python and native components
- CI/CD simplicity
- backward-compatible CLI transition

## Options Considered

1. Single root `pyproject.toml` with workspace-style internal packages
2. Multi-package layout with package-local manifests
3. Hybrid approach: root Python workspace plus package-local manifests, with a
   separate native build surface for Rust/C++

## Decision

Adopt **Option 3 — the hybrid root-workspace + package-local manifest model**
as the binding Phase 0 packaging strategy.

Concretely:

- A **root `pyproject.toml`** defines shared tooling, workspace metadata, and
  the default developer entrypoint. During Phase 0 this stays as the current
  Poetry manifest that owns the legacy `src/` application. The repository is
  treated as a Python workspace whose members are the domain packages listed
  below.
- Each of the six Python domain packages owns its own **package-local
  `pyproject.toml`** manifest under its package root:
  - `shared_lib/pyproject.toml`
  - `data_platform/pyproject.toml`
  - `backtest_engine/pyproject.toml`
  - `alpha_research/pyproject.toml`
  - `ai_agents/pyproject.toml`
  - `trading_system/pyproject.toml`
- Each domain package uses a `src/<domain>/` layout so imports resolve as
  `<domain>.<module>` without legacy `src.` prefixes.
- **Native HFT code** is kept out of the Python workspace. It lives at
  `trading_system/native/hft_engine/{core,network,fast_inference,fpga}/` and
  under `trading_system/native/shared/`, with its own Rust/C++ build manifests
  so toolchain complexity does not leak into every Python package.
- The existing **`src/` application is preserved** unchanged during Phase 0
  and acts as the compatibility CLI (`src.main`) and migration reference. No
  Phase 0 change deletes or moves large portions of `src/`.
- The `web_control_plane/` and `infrastructure/` directories exist at the
  monorepo root but are intentionally not Python packages in Phase 0. They
  will gain manifests in their owning phases (Phase 5 and Phase 9).

Rejected alternatives:

- Option 1 (single root manifest only) collapses ownership and forces every
  subsystem into one dependency tree. Rejected because it blocks independent
  CI and dependency isolation required by Phase 2+.
- Option 2 (package-local only, no root workspace) removes the shared tooling
  surface and complicates the legacy `src/` compatibility story on day one.
  Rejected because Phase 0 must be non-breaking.

## Canonical Import Convention

Absolute imports only, rooted at the domain package name:

- `shared_lib.<module>`
- `data_platform.<module>`
- `backtest_engine.<module>`
- `alpha_research.<module>`
- `ai_agents.<module>`
- `trading_system.<module>`

Legacy `src.*` imports remain allowed only inside `src/` and inside
compatibility adapters that explicitly delegate to the new packages.

## Consequences

- Phase 0 can create empty package roots with manifests without touching
  existing behaviour.
- Each domain can adopt independent dependency pins and test matrices from
  Phase 1 onward.
- Native build complexity (cargo, cmake, vendor toolchains) is isolated to
  `trading_system/native/` and does not block Python-only contributors.
- The legacy `src/` CLI continues to ship and remains the user-facing entry
  point until Phase 10 cutover.

## Exit Criteria

- one packaging strategy selected (done: Option 3)
- import conventions documented (done: see "Canonical Import Convention")
- compatibility path for `src.main` approved (done: retained as facade)
