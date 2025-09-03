#!/usr/bin/env python3
"""
Compare exports/comparison_sample_Q3_2025.json best strategy (rank 1)
with BestStrategy rows in the database (timeframe=1d).

Outputs results to exports/compare_json_db_results.json and prints a summary to stdout.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.database import get_db_session
from src.database.models import BestStrategy

INPUT = Path("exports/comparison_sample_Q3_2025.json")
OUTPUT = Path("exports/compare_json_db_results.json")
TIMEFRAME = "1d"


def main():
    if not INPUT.exists():
        print(f"Input file not found: {INPUT}")
        return

    with INPUT.open() as f:
        data = json.load(f)

    session = get_db_session()
    results = {}
    summary = {"total": 0, "matched": 0, "missing_db": 0, "mismatched": 0}

    try:
        for symbol, symbol_data in data.items():
            summary["total"] += 1
            # Prefer explicit best_strategy field if present
            json_best = None
            json_sortino = None
            if symbol_data.get("best_strategy"):
                json_best = symbol_data["best_strategy"].get("strategy")
                json_sortino = (
                    symbol_data["best_strategy"].get("metrics", {}).get("sortino_ratio")
                )
            else:
                # Fallback to results array where rank==1
                for r in symbol_data.get("results", []):
                    if r.get("rank") == 1:
                        json_best = r.get("strategy")
                        json_sortino = r.get("metrics", {}).get("sortino_ratio")
                        break

            db_row = (
                session.query(BestStrategy)
                .filter_by(symbol=symbol, timeframe=TIMEFRAME)
                .first()
            )

            if not db_row:
                results[symbol] = {
                    "json_best": json_best,
                    "json_sortino": json_sortino,
                    "db_best": None,
                    "db_sortino": None,
                    "match": False,
                    "note": "no_db_row",
                }
                summary["missing_db"] += 1
                continue

            db_best = db_row.strategy
            db_sortino = float(getattr(db_row, "sortino_ratio", 0) or 0)

            match = str(db_best).strip() == str(json_best).strip()

            if match:
                summary["matched"] += 1
            else:
                summary["mismatched"] += 1

            results[symbol] = {
                "json_best": json_best,
                "json_sortino": json_sortino,
                "db_best": db_best,
                "db_sortino": db_sortino,
                "match": match,
                "note": None,
            }

    finally:
        session.close()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, default=str)

    print("Comparison complete.")
    print("Summary:", summary)
    print(f"Detailed results written to: {OUTPUT}")


if __name__ == "__main__":
    main()
