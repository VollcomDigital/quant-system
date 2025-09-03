#!/usr/bin/env python3
"""
Compare DB-backed (unified) best-strategies CSV with direct backtests.

Reads: exports/csv/2025/Q3/bonds_collection_best_strategies_Q3_2025.csv
Writes: exports/csv/compare_unified_vs_direct_bonds_Q3_2025.csv

This script calls src.core.direct_backtest.run_direct_backtest for each symbol/strategy/timeframe
and compares key metrics (sortino_ratio, total_return).
"""

from __future__ import annotations

import csv
import datetime
import traceback
from pathlib import Path
from typing import Optional

INPUT_CSV = Path("exports/csv/2025/Q3/bonds_collection_best_strategies_Q3_2025.csv")
OUT_CSV = Path("exports/csv/compare_unified_vs_direct_bonds_Q3_2025.csv")

# Use direct backtest function from the project
try:
    from src.core.direct_backtest import run_direct_backtest
except Exception as e:
    raise RuntimeError(f"Could not import run_direct_backtest: {e}") from e


def as_float(v: Optional[str]) -> Optional[float]:
    try:
        if v is None or str(v).strip() == "":
            return None
        return float(v)
    except Exception:
        return None


def main():
    if not INPUT_CSV.exists():
        raise SystemExit(f"Input CSV not found: {INPUT_CSV}")

    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    # Use a very wide start to emulate 'max'; data manager will clamp to available history.
    start_date = "1900-01-01"
    end_date = today

    with INPUT_CSV.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    out_rows = []
    total = len(rows)
    succeeded = 0
    failed = 0

    for i, r in enumerate(rows, start=1):
        asset = r.get("Asset") or r.get("symbol") or r.get("Asset")
        strategy = r.get("Best_Strategy") or r.get("BestStrategy") or "adx"
        timeframe = r.get("Best_Timeframe") or r.get("Best_Timeframe") or "1d"
        unified_sortino = as_float(r.get("Sortino_Ratio"))
        unified_total = as_float(r.get("Total_Return_Pct"))

        print(
            f"[{i}/{total}] Running direct backtest for {asset} / {strategy} / {timeframe}"
        )
        try:
            res = run_direct_backtest(
                symbol=str(asset),
                strategy_name=str(strategy),
                start_date=start_date,
                end_date=end_date,
                timeframe=str(timeframe),
                initial_capital=10000.0,
                commission=0.001,
                persistence_context=None,
            )
            direct_sortino = None
            direct_total = None
            err = None
            try:
                native = res.get("bt_results") or {}
                v = native.get("Sortino Ratio", None)
                direct_sortino = float(v) if v is not None else None
            except Exception:
                direct_sortino = None
            try:
                native = res.get("bt_results") or {}
                v2 = native.get("Return [%]", None)
                direct_total = float(v2) if v2 is not None else None
            except Exception:
                direct_total = None

            succeeded += 1
        except Exception as e:
            err = f"{e}\n{traceback.format_exc()}"
            direct_sortino = None
            direct_total = None
            failed += 1

        sortino_diff = (
            None
            if (unified_sortino is None or direct_sortino is None)
            else float(direct_sortino) - float(unified_sortino)
        )
        total_diff = (
            None
            if (unified_total is None or direct_total is None)
            else float(direct_total) - float(unified_total)
        )

        out_rows.append(
            {
                "Asset": asset,
                "Unified_Strategy": strategy,
                "Unified_Timeframe": timeframe,
                "Unified_Sortino": "" if unified_sortino is None else unified_sortino,
                "Unified_TotalReturn": "" if unified_total is None else unified_total,
                "Direct_Sortino": "" if direct_sortino is None else direct_sortino,
                "Direct_TotalReturn": "" if direct_total is None else direct_total,
                "Sortino_Diff": "" if sortino_diff is None else sortino_diff,
                "TotalReturn_Diff": "" if total_diff is None else total_diff,
                "Error": "" if err is None else err,
            }
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        fieldnames = [
            "Asset",
            "Unified_Strategy",
            "Unified_Timeframe",
            "Unified_Sortino",
            "Unified_TotalReturn",
            "Direct_Sortino",
            "Direct_TotalReturn",
            "Sortino_Diff",
            "TotalReturn_Diff",
            "Error",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f"Done. Total={total} succeeded={succeeded} failed={failed}")
    print(f"Wrote comparison CSV to {OUT_CSV}")


if __name__ == "__main__":
    main()
