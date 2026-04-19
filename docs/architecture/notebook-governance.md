# Notebook Governance

- Status: Accepted (Phase 3)
- Source ADRs: ADR-0001 (package boundaries)
- Applies to: every Jupyter notebook under `alpha_research/notebooks/`.

## Purpose

Notebooks are the primary surface for exploratory research. They are
fast to iterate in but easy to misuse:

- they accumulate silent state;
- they bake in credentials and dataset paths;
- their outputs leak results into version control;
- they drift away from the production factor library.

This document turns those failure modes into explicit rules so
Phase 3 notebooks graduate into the factor library cleanly and never
ship state or secrets to the repo.

## Notebook Execution Rules

- Every notebook runs top-to-bottom with a fresh kernel in CI.
  "Works on my machine" runs do not count.
- Notebooks must not depend on network access or credentials at execution
  time. Data access must go through `data_platform` connectors that can
  be mocked or replayed from `SnapshotIndex` snapshots.
- Long-running cells (> 60 seconds in CI) must be moved into
  `alpha_research/factor_library/` or `alpha_research/ml_models/` as
  regular Python modules so they can be unit-tested.
- Notebook filenames are ordered and numbered so the research narrative
  is linear: `01-universe.ipynb`, `02-features.ipynb`, etc.

## Output and Cleanup Rules

- Notebook outputs must be stripped (**strip output**) before commit. CI
  enforces this by re-running the notebook and comparing output hashes.
- Secrets (API keys, broker credentials, model weights) never appear in
  notebook cells. Secrets are loaded from environment variables via a
  `shared_lib.logging` redaction-safe path, and the redaction rules from
  Phase 1 apply.
- Cell magics that pickle objects to disk must write under
  `alpha_research/notebooks/.scratch/` which is `.gitignore`d.
- Plots render server-side PNG/SVG; no interactive backends are persisted
  to notebooks.

## Notebook to Factor Library Promotion

A notebook cell graduates to the factor library when:

1. It can be expressed as a `Factor` subclass with explicit metadata
   (description, source_dependencies, stationarity_assumption,
   universe_coverage, leakage_review).
2. It passes every check in `alpha_research.promotion.promote_factor`
   with `target_status="validated"`.
3. The notebook cell is replaced with a call into the factor library
   so future notebooks do not duplicate the logic.

Model artifacts follow the same path via
`alpha_research.ml_models.registry.ModelRegistry` and
`alpha_research.promotion.promote_model`.

## Forbidden Patterns

- Notebooks **must not import** from `src/*` (legacy CLI). They also
  **must not** be imported by any domain package. This boundary is
  tested by `tests/phase_3/test_notebook_governance.py`.
- Notebooks must not bypass the feature store. If you need a factor
  for a research step, register it first (even as `candidate`) and
  read through `FeatureStore`.
- Notebooks must not import `trading_system.*`. Research cannot place
  live orders.
- Notebook code must not depend on pandas in new work; use Polars
  (ADR-0002 policy).

## Enforcement

- **CI**: a dedicated notebook job runs every notebook with a clean
  kernel and refuses merges if any notebook contains cell outputs.
- **pre-commit**: a local hook strips outputs and refuses commits that
  contain secret-shaped values in notebook JSON.
- **Agent review**: the Phase 5 `code_reviewer` agent treats any new
  notebook import into a domain package, any notebook with persistent
  outputs, or any long-running notebook cell as a blocking review
  finding.
- **ADR anchor**: changes to this document require an ADR update
  (ADR-0001 or a superseder).
