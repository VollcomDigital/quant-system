# Governance and Branch Protection

This document outlines how we maintain the repository, protect the `main` branch, and merge changes safely.

## Branching Model

- Default branch: `main` (protected)
- Work happens on short-lived feature branches: `feat/...`, `fix/...`, `chore/...`.
- Use pull requests (PRs) for all changes to `main`.
- Tag releases with SemVer: `vMAJOR.MINOR.PATCH`.

## Branch Protection (recommended GitHub settings)

Enable on `Settings → Branches → Branch protection rules` for `main`:

- Require a pull request before merging
  - Require approvals: 1–2 (at least 1 is recommended)
  - Dismiss stale approvals when new commits are pushed
  - Require review from Code Owners (optional)
  - Require conversation resolution
- Require status checks to pass before merging
  - ci / lint (from `.github/workflows/ci.yml`)
  - ci / tests (coverage gate ≥ 80%)
  - pre-commit (runs Ruff, Markdown lint, coverage gate)
  - codeql / analyze (from `.github/workflows/codeql.yml`)
  - gitleaks / secret scan (from `.github/workflows/gitleaks.yml`)
  - daily-backtest is not required (scheduled), optional
  - Require branches to be up to date before merging
- Require signed commits (optional but recommended)
- Linear history (disable merge commits or enforce squash/rebase)
- Restrict who can push to matching branches (maintainers only)
- Do not allow force pushes or deletions on protected branches

## CODEOWNERS

Add `CODEOWNERS` so critical files require review from maintainers:

```text
*       @manuelheck
.github/ @manuelheck
src/    @manuelheck
```

Place this in `.github/CODEOWNERS` (see file added in this repo). Adjust handles as needed.

## CI & Security

- Dependabot updates weekly for Python and GitHub Actions
- CodeQL code scanning on PRs, pushes to `main`, and weekly schedule
- Gitleaks secret scanning on PRs and pushes
- Pre-commit hooks enforced locally and in CI

## Releases

- Create a release branch if needed for stabilization
- Tag `vX.Y.Z` on `main` after CI passes
- Draft GitHub Release notes summarizing changes

## Backports / Hotfixes

- Cherry-pick fixes onto release branches, open PRs, and tag patch releases

## Vulnerability Handling

- Follow `SECURITY.md` for private disclosures
- Do not discuss vulnerabilities in public issues until a fix is released

## Decision Making

- Small changes: PR review by 1 maintainer
- Larger/architectural changes: open an issue or design doc for discussion before implementation
