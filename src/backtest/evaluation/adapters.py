from __future__ import annotations

from typing import Any

def normalized_rows_to_legacy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "collection": row.get("collection"),
                "symbol": row.get("symbol"),
                "timeframe": row.get("timeframe"),
                "strategy": row.get("strategy"),
                "params": row.get("params") or {},
                "metric": row.get("metric"),
                "metric_value": float(row.get("metric_value", float("nan"))),
                "stats": row.get("stats") or {},
            }
        )
    return out
