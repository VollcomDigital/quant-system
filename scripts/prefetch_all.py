#!/usr/bin/env python3
"""
Prefetch multiple collections in one command.

Examples:
  python scripts/prefetch_all.py bonds commodities --mode recent --interval 1d --recent-days 90
  python scripts/prefetch_all.py --all --mode full --interval 1d
"""

from __future__ import annotations

import argparse
from typing import List

from scripts.prefetch_collection import prefetch as prefetch_one

DEFAULT_COLLECTIONS = ["bonds", "commodities", "crypto", "forex", "indices"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prefetch multiple collections")
    parser.add_argument("collections", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--mode", choices=["full", "recent", "both"], default="recent")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--recent-days", type=int, default=90)
    args = parser.parse_args(argv)

    collections: List[str]
    if args.all or not args.collections:
        collections = DEFAULT_COLLECTIONS
    else:
        collections = list(args.collections)

    for c in collections:
        print(f"Prefetching {c} ...")
        prefetch_one(c, args.mode, args.interval, args.recent_days)

    print("All prefetches complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
