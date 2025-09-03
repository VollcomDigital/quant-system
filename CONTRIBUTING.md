# Contributing

Thanks for your interest in contributing to quant-system!

## License and Use

This repository is released under the Business Source License 1.1 (BUSL‑1.1).
Commercial use is restricted until the Change Date listed in `LICENSE`. On that
date, the project will convert to the MIT License.

## Deprecations and Import Paths

We recently renamed modules:

- `src.core.portfolio_manager` → `src.core.collection_manager` (class: `PortfolioManager`)
- `src.utils.tradingview_alert_exporter` → `src.utils.tv_alert_exporter`

Compatibility shims exist for now and will emit `DeprecationWarning`. Please
update imports to the new modules. The shims are scheduled for removal after
the next minor release.

## Development

- Use `docker compose` and the unified CLI. See `README.md` and `docs/docker.md`.
- Run `pre-commit` locally: `pre-commit install && pre-commit run -a`.
- Tests run inside Docker via the pre-commit hook.

## Pull Requests

- Keep PRs focused and small.
- Include tests for behavior changes.
- Pass pre-commit hooks (format, lint, tests).
