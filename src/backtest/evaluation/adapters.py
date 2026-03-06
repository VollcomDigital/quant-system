from __future__ import annotations

from typing import Any

from .contracts import ResultRecord


def result_record_to_legacy_row(record: ResultRecord) -> dict[str, Any]:
    return {
        "collection": record.collection,
        "symbol": record.symbol,
        "timeframe": record.timeframe,
        "strategy": record.strategy,
        "params": dict(record.params),
        "metric": record.metric_name,
        "metric_value": float(record.metric_value),
        "stats": dict(record.stats),
    }


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
