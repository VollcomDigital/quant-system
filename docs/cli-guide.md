# CLI Reference

This guide documents the CLI. It includes a short section for the current unified CLI and a preserved legacy section for older multi-subcommand commands.

Note: The current entrypoint focuses on the `collection` subcommand. Use the README for up-to-date commands. Legacy examples are kept below for context.

## Current (Unified) CLI

```bash
# Show help (inside Docker)
docker compose run --rm quant python -m src.cli.unified_cli --help

# Run bonds collection (1d/max, all strategies)
docker compose run --rm -e STRATEGIES_PATH=/app/external_strategies \
  quant python -m src.cli.unified_cli collection bonds \
  --action direct --interval 1d --period max --strategies all --exports all

# Dry run (plan only) + exports (csv, report, tradingview, ai or all)
docker compose run --rm -e STRATEGIES_PATH=/app/external_strategies \
  quant python -m src.cli.unified_cli collection bonds \
  --interval 1d --period max --strategies all --dry-run --exports all

Exports and naming:
- CSV → `exports/csv/<Year>/<Quarter>/<Collection>_Collection_<Year>_<Quarter>_<Interval>.csv`
- Reports → `exports/reports/<Year>/<Quarter>/<Collection>_Collection_<Year>_<Quarter>_<Interval>.html`
- TV alerts → `exports/tv_alerts/<Year>/<Quarter>/<Collection>_Collection_<Year>_<Quarter>_<Interval>.md`
- AI recos (md/html) → `exports/ai_reco/<Year>/<Quarter>/<Collection>_Collection_<Year>_<Quarter>_<Interval>.*`

When multiple intervals are used, filenames prefer `1d`. Use `--interval 1d` to constrain content and filenames.
```

## Legacy CLI (Preserved)

These examples refer to a previous iteration of the CLI that exposed categories like `portfolio`, `data`, `cache`, and `reports`. Prefer the section above for current usage.

### Quick Start (legacy)

```bash
# Activate environment
poetry shell

# List available portfolios
python -m src.cli.unified_cli portfolio list

# Test a portfolio
python -m src.cli.unified_cli portfolio test crypto --open-browser
```

### Command Structure (legacy)

```
python -m src.cli.unified_cli <category> <command> [options]
```

### Portfolio Commands (legacy)

#### List Portfolios
```bash
python -m src.cli.unified_cli portfolio list
```

#### Test Portfolio
```bash
python -m src.cli.unified_cli portfolio test <name> [options]

Options:
  --metric METRIC        Performance metric (sharpe_ratio, sortino_ratio)
  --period PERIOD        Time period (1d, 1w, 1m, 3m, 6m, 1y, max)
  --test-timeframes      Test multiple timeframes
  --open-browser         Auto-open results in browser
```

#### Test All Strategies and Timeframes
```bash
python -m src.cli.unified_cli portfolio test-all --symbols SYMBOL1,SYMBOL2 [options]

Options:
  --symbols SYMBOLS      Comma-separated symbols to test
  --start-date DATE      Start date (YYYY-MM-DD)
  --end-date DATE        End date (YYYY-MM-DD)
  --strategies LIST      Comma-separated strategies to test
```

### Data Commands (legacy)

#### Download Data
```bash
python -m src.cli.unified_cli data download --symbols AAPL,GOOGL [options]

Options:
  --symbols SYMBOLS      Comma-separated symbols
  --start-date DATE      Start date (YYYY-MM-DD)
  --end-date DATE        End date (YYYY-MM-DD)
  --source SOURCE        Data source (yahoo, alpha_vantage, etc.)
```

### Cache Commands (legacy)

#### Cache Statistics
```bash
python -m src.cli.unified_cli cache stats
```

#### Clear Cache
```bash
python -m src.cli.unified_cli cache clear [--all] [--symbol SYMBOL]
```

### Report Commands (legacy)

#### Generate Reports
```bash
python -m src.cli.unified_cli reports generate <portfolio> [options]

Options:
  --format FORMAT        Output format (html, pdf, json)
  --period PERIOD        Analysis period
  --output-dir DIR       Output directory
```

#### Organize Reports
```bash
python -m src.cli.unified_cli reports organize
```

### Examples (legacy)

#### Test Crypto Portfolio
```bash
# Using Sortino ratio (default - superior to Sharpe)
python -m src.cli.unified_cli portfolio test crypto \
  --metric sortino_ratio \
  --period 1y \
  --test-timeframes \
  --open-browser

# Traditional Sharpe ratio (for comparison)
python -m src.cli.unified_cli portfolio test crypto \
  --metric sharpe_ratio \
  --period 1y
```

#### Download Forex Data
```bash
python -m src.cli.unified_cli data download \
  --symbols EURUSD=X,GBPUSD=X \
  --start-date 2023-01-01 \
  --source twelve_data
```

#### Daily Workflow
```bash
# Check cache status
python -m src.cli.unified_cli cache stats

# Test all portfolios (Sortino ratio default)
python -m src.cli.unified_cli portfolio test-all --metric sortino_ratio --period 1d --open-browser

# Organize reports
python -m src.cli.unified_cli reports organize
```

## Configuration (legacy)

Set environment variables in `.env`:
```bash
LOG_LEVEL=INFO
CACHE_ENABLED=true
DEFAULT_PERIOD=1y
BROWSER_AUTO_OPEN=true
```

## Help (legacy)

Get help for any command:
```bash
python -m src.cli.unified_cli --help
python -m src.cli.unified_cli portfolio --help
python -m src.cli.unified_cli portfolio test --help
```
