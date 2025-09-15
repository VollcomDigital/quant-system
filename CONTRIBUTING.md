# Contributing Guide

Thanks for your interest in contributing! This document describes how to propose changes and what we expect for code quality, tests, and security.

## Development Setup

- Use Python 3.10 (vectorbt requires <3.11). The Docker image already provides a working toolchain.
- Recommended path: develop inside Docker with Poetry:

  - Build: `docker-compose build`
  - Shell: `docker-compose run --rm app bash`
  - Install deps: `poetry install`

## Working on Issues

- Check existing issues before opening a new one.
- For significant changes, open a discussion or issue first to align on the approach.

## Branching and Commits

- Create feature branches off `main` using concise names, e.g. `feat/xyz`, `fix/abc`.
- Write clear commit messages (imperative mood). Group small related changes together.

## Code Quality

- Run linters and formatters via pre-commit:

  ```bash
  pip install pre-commit
  pre-commit install
  pre-commit run --all-files
  ```

- Python style is enforced by Ruff (including import sorting).

## Tests & Coverage

- Add unit tests for new logic. Keep tests deterministic (no network calls) unless explicitly marked as integration.
- Coverage gate is 80% (enforced by pre-commit and CI). Prefer small tests over untested features.

## Security

- Never commit secrets. Pre-commit and CI include secret scanning; Dependabot and CodeQL are enabled.
- Report security issues privately (see SECURITY.md). Do not open public issues for vulnerabilities.

## Submitting a PR

1. Ensure pre-commit passes locally.
2. Include a brief description of the change and rationale.
3. Reference related issues.
4. Keep PRs focused; large unrelated changes are likely to be requested to split.

## Release & Changelog

- Maintainers will tag releases and update release notes.

Thanks again for helping improve this project!
