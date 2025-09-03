#!/usr/bin/env python3
"""
Check available OHLC data ranges for given symbols using UnifiedDataManager.

Prints symbol, number of rows, first date, last date (UTC).
"""

from __future__ import annotations

import datetime

try:
    from src.core.data_manager import UnifiedDataManager
except Exception as e:
    raise SystemExit(f"Could not import UnifiedDataManager: {e}")

SYMBOLS = ["AGG", "HYG", "TLT", "JPST", "EMB"]


def fmt(dt):
    if dt is None:
        return "None"
    try:
        return dt.tz_convert("UTC").isoformat()
    except Exception:
        try:
            return dt.isoformat()
        except Exception:
            return str(dt)


def main():
    dm = UnifiedDataManager()
    for s in SYMBOLS:
        try:
            print(f"--- {s} ---")
            # Request wide range to emulate 'max' (use aware date for lint)
            today_iso = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
            data = dm.get_data(s, "1900-01-01", today_iso, "1d")
            if data is None:
                print("No data returned")
                continue
            # Ensure index is datetime
            idx = data.index
            if len(idx) == 0:
                print("Empty index")
                continue
            first = idx[0]
            last = idx[-1]
            print("rows:", len(data))
            print("first:", fmt(first))
            print("last:", fmt(last))
        except Exception as e:
            print("Error fetching", s, "->", e)


if __name__ == "__main__":
    main()
