# Package Naming and Import Conventions

- Status: Accepted (Phase 0)
- Source ADRs: ADR-0001
- Applies to: every Python package in the monorepo
- Enforcement: ruff + pre-commit + CI lint job (see "Enforcement" below)

## Scope

These rules apply to every Python package under the monorepo root. They are
normative for all future phases and for every PR, including agent-authored
PRs.

## Package Naming Rules

- Package names use `snake_case`.
- No dashes. No camelCase. No leading digits.
- Names are singular and describe a domain, not a layer
  (`ai_agents`, not `agents_lib`).
- A sub-package directory carries an `__init__.py`; namespace packages are
  not used in Phase 0.
- The six Python domain packages are fixed at the monorepo root:
  - `shared_lib`
  - `data_platform`
  - `backtest_engine`
  - `alpha_research`
  - `ai_agents`
  - `trading_system`
- Non-Python roots (`web_control_plane`, `infrastructure`, `docs`, `src`,
  `tasks`, `tests`) do not follow the Python package rules.

## Import Convention

Absolute imports only. Resolve against the domain package name, never via
`src.*`, never via relative traversal.

Canonical examples (these prefixes are the contract):

- `shared_lib.contracts.<module>`
- `shared_lib.logging.<module>`
- `shared_lib.math_utils.<module>`
- `data_platform.connectors.<module>`
- `data_platform.feature_store.<module>`
- `backtest_engine.simulator.<module>`
- `backtest_engine.analytics.<module>`
- `alpha_research.factor_library.<module>`
- `alpha_research.ml_models.<module>`
- `ai_agents.runtime.<module>`
- `ai_agents.alpha_researcher.<module>`
- `trading_system.oms.<module>`
- `trading_system.ems.<module>`
- `trading_system.shared_gateways.<module>`

Cross-package imports always go through the canonical domain prefix. Relative
imports (`from ..foo import bar`) are allowed only within a single
sub-package.

## Forbidden Patterns

The following patterns are forbidden in new code:

- `from src.<anything> import ...` in any domain package.
- `import src.<anything>` in any domain package.
- Bare wildcard imports (`from <pkg> import *`).
- Mixing `src.` and domain-prefixed imports in the same module.
- Directory or package names with dashes or camelCase.
- Adding a new top-level directory that is not already approved in
  ADR-0001.

The only places where `src.*` imports are allowed:

- files physically located inside `src/` (legacy application); and
- thin compatibility adapters that re-export new-package symbols to keep
  the legacy CLI working during migration.

Any new code outside `src/` that imports `src.*` must not be merged.

## Test Discovery

- Each domain package ships a `tests/` directory at its root.
- Repository-level tests live under `./tests/`.
- Phase-gated tests live under `./tests/phase_<n>/` and enforce
  architectural invariants for that phase.
- Test module names follow `test_*.py`.
- Test functions follow `test_*`.
- Fixtures live in `conftest.py` at the narrowest scope that covers their
  consumers.

## Enforcement

Enforcement is layered so that the rules cannot drift silently:

1. **ruff**: configured to flag forbidden import patterns and enforce import
   ordering. Configured via `ruff.toml` at the repo root.
2. **pre-commit**: runs ruff and the phase-invariant tests on every commit
   via `.pre-commit-config.yaml`.
3. **CI lint job**: re-runs ruff and `pytest tests/phase_0/` on every pull
   request so drift is caught before merge.
4. **Agent contract**: the code_reviewer agent (Phase 5) treats violations
   of this document as blocking review comments.

## Change Control

This document is normative. Changes require:

- an update to ADR-0001 or a superseding ADR; and
- a matching update to the Phase 0 invariant tests in
  `tests/phase_0/test_import_naming_conventions.py`.
