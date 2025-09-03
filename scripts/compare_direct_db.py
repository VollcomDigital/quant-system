#!/usr/bin/env python3
"""
Compare direct backtesting library results (exports/direct_portfolio_comparison.json)
with BestStrategy rows in the database.

Produces exports/compare_direct_db_results.json and prints a summary.

Usage:
  python3 scripts/compare_direct_db.py --direct exports/direct_portfolio_comparison.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.database import unified_models as um


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--direct",
        required=True,
        help="Path to direct backtest JSON (exports/direct_portfolio_comparison.json)",
    )
    return p.parse_args()


def normalize_interval_name(interval: str) -> str:
    """Normalize interval strings to the format stored in DB/timeframe keys."""
    # Accept both "1min" and "1m" etc. We will normalize common variants to short form used across project.
    mapping = {
        "1min": "1m",
        "2min": "2m",
        "5min": "5m",
        "15min": "15m",
        "30min": "30m",
        "60min": "60m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
        "5d": "5d",
        "1wk": "1wk",
        "1mo": "1mo",
        "3mo": "3mo",
    }
    return mapping.get(interval, interval)


def main():
    args = parse_args()
    direct_path = Path(args.direct)
    if not direct_path.exists():
        print("Direct results file not found:", direct_path)
        return

    with direct_path.open() as f:
        data = json.load(f)

    sess = um.Session()
    results = {}
    summary = {"total": 0, "matched": 0, "missing_db": 0, "mismatched": 0}

    try:
        for symbol, intervals in data.items():
            results.setdefault(symbol, {})
            for interval, runs in intervals.items():
                summary["total"] += 1
                norm_interval = normalize_interval_name(interval)

                # runs is a list of dicts: {"strategy":..., "stats":..., "error":...}
                # Find best strategy by native Sortino Ratio (highest). Ignore errored runs.
                candidates = [
                    r
                    for r in runs
                    if (r.get("stats") or {}).get("Sortino Ratio") is not None
                ]
                if not candidates:
                    results[symbol][interval] = {
                        "direct_best": None,
                        "direct_sortino": None,
                        "db_best": None,
                        "db_sortino": None,
                        "match": False,
                        "note": "no_valid_direct_metrics",
                    }
                    summary["mismatched"] += 1
                    continue

                best_direct = max(
                    candidates,
                    key=lambda r: float(
                        (r.get("stats") or {}).get("Sortino Ratio") or float("-inf")
                    ),
                )
                direct_best_name = best_direct.get("strategy")
                try:
                    direct_sortino = float(
                        (best_direct.get("stats") or {}).get("Sortino Ratio") or 0
                    )
                except Exception:
                    direct_sortino = 0.0

                # Query DB for BestStrategy for this symbol/timeframe
                db_row = (
                    sess.query(um.BestStrategy)
                    .filter(
                        um.BestStrategy.symbol == symbol,
                        um.BestStrategy.timeframe == norm_interval,
                    )
                    .first()
                )

                if not db_row:
                    results[symbol][interval] = {
                        "direct_best": direct_best_name,
                        "direct_sortino": direct_sortino,
                        "db_best": None,
                        "db_sortino": None,
                        "match": False,
                        "note": "no_db_row",
                    }
                    summary["missing_db"] += 1
                    continue

                db_best = db_row.strategy
                db_sortino = float(getattr(db_row, "sortino_ratio", 0) or 0)

                match = str(db_best).strip() == str(direct_best_name).strip()
                if match:
                    summary["matched"] += 1
                else:
                    summary["mismatched"] += 1

                results[symbol][interval] = {
                    "direct_best": direct_best_name,
                    "direct_sortino": direct_sortino,
                    "db_best": db_best,
                    "db_sortino": db_sortino,
                    "match": match,
                    "note": None,
                }

    finally:
        sess.close()

    out_path = Path("exports") / "compare_direct_db_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, default=str)

    print("Comparison complete.")
    print("Summary:", summary)
    print("Detailed results written to:", out_path)


if __name__ == "__main__":
    main()
