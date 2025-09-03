#!/usr/bin/env python3
"""
Prefetch collection market data into the cache.

Modes:
- full   : fetch provider 'max' period for full snapshots (long TTL)
- recent : fetch last N days for recent overlay (short TTL)
- both   : full followed by recent

Examples:
  python scripts/prefetch_collection.py bonds --mode full --interval 1d
  python scripts/prefetch_collection.py config/collections/bonds.json --mode recent --interval 1d --recent-days 90

Cron (daily recent overlay at 01:30):
  30 1 * * * /usr/bin/env bash -lc 'cd /path/to/quant-system && \
    docker compose run --rm quant python scripts/prefetch_collection.py bonds --mode recent --interval 1d --recent-days 90 >/dev/null 2>&1'
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import List

from src.cli.unified_cli import load_collection_symbols, resolve_collection_path
from src.core.data_manager import UnifiedDataManager


def prefetch(collection: str, mode: str, interval: str, recent_days: int) -> None:
    p = (
        resolve_collection_path(collection)
        if not Path(collection).exists()
        else Path(collection)
    )
    symbols: List[str] = load_collection_symbols(p)
    if not symbols:
        print("No symbols found in collection")
        return

    dm = UnifiedDataManager()
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()

    if mode in ("full", "both"):
        print(f"[full] Fetching provider max for {len(symbols)} symbols @ {interval}")
        dm.get_batch_data(
            symbols,
            start_date="1900-01-01",
            end_date=today,
            interval=interval,
            use_cache=False,
            period="max",
            period_mode="max",
        )

    if mode in ("recent", "both"):
        start_recent = (
            dt.datetime.now(dt.timezone.utc).date()
            - dt.timedelta(days=int(recent_days))
        ).isoformat()
        print(
            f"[recent] Fetching {recent_days} days for {len(symbols)} symbols @ {interval}"
        )
        dm.get_batch_data(
            symbols,
            start_date=start_recent,
            end_date=today,
            interval=interval,
            use_cache=False,
        )

    print("Prefetch complete.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prefetch collection data into cache")
    parser.add_argument(
        "collection", help="Collection key (e.g., bonds) or path to JSON"
    )
    parser.add_argument("--mode", choices=["full", "recent", "both"], default="recent")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--recent-days", type=int, default=90)
    args = parser.parse_args(argv)

    prefetch(args.collection, args.mode, args.interval, args.recent_days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
