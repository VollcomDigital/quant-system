from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..backtest.results_cache import ResultsCache
from .dashboard import (
    _build_summary,
    _extract_highlights,
)


def refresh_manifest(
    reports_root: Path,
    current_run_dir: Path,
    cache: ResultsCache,
    current_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    reports_root = Path(reports_root)
    current_run_dir = Path(current_run_dir)
    actions: list[dict[str, Any]] = []
    if not reports_root.exists():
        return actions

    for run_dir in reports_root.iterdir():
        if not run_dir.is_dir():
            continue
        dash_path = run_dir / "dashboard.json"
        summary_path = run_dir / "summary.json"
        if dash_path.exists():
            updated = _ensure_highlights(dash_path)
            actions.append(
                {
                    "run_id": run_dir.name,
                    "status": "existing_updated" if updated else "existing",
                    "path": str(dash_path),
                }
            )
            continue
        summary = _load_json(summary_path)
        if run_dir == current_run_dir:
            dash_path.write_text(
                json.dumps(current_payload, indent=2, default=_json_default),
            )
            actions.append(
                {
                    "run_id": run_dir.name,
                    "status": "current_saved",
                    "path": str(dash_path),
                }
            )
            continue
        if summary is None:
            actions.append(
                {
                    "run_id": run_dir.name,
                    "status": "missing_summary",
                    "summary_path": str(summary_path),
                    "message": "Summary file missing; unable to rebuild dashboard",
                }
            )
            continue
        run_id = run_dir.name
        payload = _build_payload_from_summary(summary, run_id, cache)
        if payload is not None:
            dash_path.write_text(json.dumps(payload, indent=2, default=_json_default))
            actions.append(
                {
                    "run_id": run_id,
                    "status": "created",
                    "path": str(dash_path),
                    "source": payload.get("source"),
                    "message": _message_for_created(payload.get("source")),
                }
            )
        else:
            actions.append(
                {
                    "run_id": run_id,
                    "status": "failed",
                    "summary_path": str(summary_path),
                    "message": "Unable to rebuild dashboard from summary",
                }
            )

    return actions


def _build_payload_from_summary(
    summary: dict[str, Any], run_id: str, cache: ResultsCache
) -> dict[str, Any] | None:
    metric = summary.get("metric")
    if not metric:
        return None
    results_count = summary.get("results_count", 0)
    rows: list[dict[str, Any]] = []
    source = "cache" if results_count else "none"
    if results_count:
        rows = cache.list_by_run(run_id)
    if not rows:
        results_csv = (
            Path(summary.get("base_dir", "")) / "all_results.csv"
            if summary.get("base_dir")
            else None
        )
        if results_csv and results_csv.exists():
            rows = _rows_from_csv(results_csv)
            source = "csv"
    if not rows:
        return None
    summary_data = _build_summary(rows)
    highlights = _extract_highlights(summary_data)
    payload = {
        "run_id": run_id,
        "rows": rows,
        "summary": summary_data,
        "available_metrics": list(summary_data.get("metrics", {}).keys()),
        "highlights": highlights,
        "metric": metric,
        "results_count": results_count,
        "started_at": summary.get("started_at"),
        "finished_at": summary.get("finished_at"),
        "duration_sec": summary.get("duration_sec"),
        "source": source,
    }
    return payload


def _rows_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    import csv

    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats = {
                "sharpe": _float(row.get("sharpe")),
                "sortino": _float(row.get("sortino")),
                "omega": _float(row.get("omega")),
                "tail_ratio": _float(row.get("tail_ratio")),
                "profit": _float(row.get("profit")),
                "pain_index": _float(row.get("pain_index")),
                "max_drawdown": _float(row.get("max_drawdown")),
                "cagr": _float(row.get("cagr")),
                "calmar": _float(row.get("calmar")),
            }
            rows.append(
                {
                    "collection": row.get("collection"),
                    "symbol": row.get("symbol"),
                    "timeframe": row.get("timeframe"),
                    "strategy": row.get("strategy"),
                    "metric": row.get("metric"),
                    "metric_value": _float(row.get("metric_value")),
                    "params": row.get("params"),
                    "stats": stats,
                }
            )
    return rows


def _ensure_highlights(dash_path: Path) -> bool:
    try:
        payload = json.loads(dash_path.read_text())
    except Exception:
        return False
    if payload.get("highlights"):
        return False
    summary = payload.get("summary")
    payload["highlights"] = _extract_highlights(summary)
    dash_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return True


def _message_for_created(source: Any) -> str:
    if source == "csv":
        return "Dashboard regenerated using CSV fallback"
    if source == "cache":
        return "Dashboard regenerated from results cache"
    return "Dashboard regenerated"


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        data.setdefault("base_dir", str(path.parent))
        return data
    except Exception:
        return None


def _float(val: Any) -> float | None:
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return None
    return fval


def _json_default(obj: Any) -> Any:
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
