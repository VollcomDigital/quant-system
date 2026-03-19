from __future__ import annotations

from typing import Any

def normalized_rows_to_legacy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        metric_raw = row.get("metric_value")
        try:
            metric_value = float(metric_raw) if metric_raw is not None else float("nan")
        except (TypeError, ValueError):
            metric_value = float("nan")

        out.append(
            {
                "collection": row.get("collection"),
                "symbol": row.get("symbol"),
                "timeframe": row.get("timeframe"),
                "strategy": row.get("strategy"),
                "params": row.get("params") or {},
                "metric": row.get("metric"),
                "metric_value": metric_value,
                "stats": row.get("stats") or {},
            }
        )
    return out
