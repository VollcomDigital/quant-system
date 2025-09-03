#!/usr/bin/env python3
"""
Data Health Report for a collection.

Outputs CSV with: symbol, rows, first_date, last_date, stale (Y/N)
Optionally prints a summary to stdout.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import pandas as pd
from pandas.tseries.offsets import BDay

from src.cli.unified_cli import load_collection_symbols, resolve_collection_path
from src.core.data_manager import UnifiedDataManager


def is_stale(last_date: pd.Timestamp) -> bool:
    try:
        expected = (pd.Timestamp.today().normalize() - BDay(1)).date()
        return last_date.date() < expected
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Data health report for a collection")
    parser.add_argument("collection", help="Collection key or path to JSON")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--period", default="")
    parser.add_argument("--out", default="artifacts/data_health.csv")
    args = parser.parse_args(argv)

    p = (
        resolve_collection_path(args.collection)
        if not Path(args.collection).exists()
        else Path(args.collection)
    )
    symbols: List[str] = load_collection_symbols(p)
    dm = UnifiedDataManager()

    rows: List[dict] = []
    for s in symbols:
        try:
            df = dm.get_data(
                s,
                start_date="1900-01-01" if args.period == "max" else "2000-01-01",
                end_date=pd.Timestamp.today().strftime("%Y-%m-%d"),
                interval=args.interval,
                use_cache=True,
                period=args.period or None,
                period_mode=args.period or None,
            )
            if df is None or df.empty:
                rows.append(
                    {
                        "symbol": s,
                        "rows": 0,
                        "first_date": "",
                        "last_date": "",
                        "stale": "Y",
                    }
                )
                continue
            first = df.index[0]
            last = df.index[-1]
            rows.append(
                {
                    "symbol": s,
                    "rows": len(df),
                    "first_date": first.date().isoformat(),
                    "last_date": last.date().isoformat(),
                    "stale": "Y" if is_stale(last) else "N",
                }
            )
        except Exception:
            rows.append(
                {
                    "symbol": s,
                    "rows": 0,
                    "first_date": "",
                    "last_date": "",
                    "stale": "Y",
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["symbol", "rows", "first_date", "last_date", "stale"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote data health report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
