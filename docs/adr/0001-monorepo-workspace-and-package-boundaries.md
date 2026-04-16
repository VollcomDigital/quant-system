# ADR 0001: Monorepo Workspace and Package Boundaries

- Status: Proposed
- Owners: Platform, Research, Execution
- Target phase: Phase 0

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

## Options to Evaluate

1. Single root `pyproject.toml` with workspace-style internal packages
2. Multi-package layout with package-local manifests
3. Hybrid approach: root Python workspace plus native-package manifests

## Proposed Direction

Prefer a root-controlled workspace model for Phase 0 so the repository can
introduce domain directories and shared tooling without immediately fragmenting
dependency management. Native components can still keep local build manifests
when required by Rust/C++ toolchains.

## Questions to Resolve

- Which internal import prefix should be canonical?
- Which packages must remain Python-only in Phase 0?
- Which native modules require independent build pipelines on day one?

## Exit Criteria

- one packaging strategy selected
- import conventions documented
- compatibility path for `src.main` approved
