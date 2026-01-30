from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..backtest.results_cache import ResultsCache
from ..utils.json_utils import safe_json_dumps

if TYPE_CHECKING:  # pragma: no cover
    pass

MetricRow = dict[str, Any]

METRIC_KEYS: tuple[str, ...] = (
    "sharpe",
    "sortino",
    "omega",
    "tail_ratio",
    "profit",
    "pain_index",
    "max_drawdown",
    "cagr",
    "calmar",
)

METRIC_DIRECTIONS: dict[str, str] = {
    "pain_index": "min",
}

HIGHLIGHT_METRICS: tuple[str, ...] = (
    "sharpe",
    "omega",
    "tail_ratio",
    "pain_index",
    "profit",
)


DOWNLOAD_FILE_CANDIDATES: tuple[str, ...] = (
    "report.html",
    "summary.json",
    "summary.csv",
    "all_results.csv",
    "top3.csv",
    "manifest_status.json",
    "notifications.json",
)


def _as_dict(obj: Any) -> dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    raise TypeError(f"Cannot convert object of type {type(obj)} to dict")


def _best_result_to_row(best: Any) -> MetricRow:
    raw = _as_dict(best)
    return {
        "collection": raw.get("collection"),
        "symbol": raw.get("symbol"),
        "timeframe": raw.get("timeframe"),
        "strategy": raw.get("strategy"),
        "metric": raw.get("metric_name"),
        "metric_value": float(raw.get("metric_value", float("nan"))),
        "params": raw.get("params", {}),
        "stats": raw.get("stats", {}),
    }


def build_dashboard_payload(
    cache: ResultsCache, run_id: str, best_results: Iterable[Any] | None
) -> dict[str, Any]:
    rows = cache.list_by_run(run_id)
    if not rows and best_results:
        rows = [_best_result_to_row(br) for br in best_results]

    summary = _build_summary(rows)
    return {
        "run_id": run_id,
        "rows": rows,
        "summary": summary,
        "available_metrics": list(METRIC_KEYS),
        "highlights": _extract_highlights(summary),
    }


def collect_runs_manifest(
    reports_root: Path,
    current_run_id: str,
    current_summary: dict[str, Any] | None,
    current_meta: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    if reports_root.exists():
        for child in reports_root.iterdir():
            if not child.is_dir():
                continue
            if child.name == current_run_id:
                continue
            entry = _manifest_entry_from_dir(child)
            if entry is not None:
                manifest.append(entry)

    summary_current = current_summary or {}
    current_entry = {
        "run_id": current_run_id,
        "path": "dashboard.json",
        "metric": (current_meta or {}).get("metric"),
        "results_count": (current_meta or {}).get("results_count"),
        "started_at": (current_meta or {}).get("started_at"),
        "duration_sec": (current_meta or {}).get("duration_sec"),
        "summary": summary_current,
        "highlights": _extract_highlights(summary_current),
    }
    manifest.append(current_entry)
    manifest.sort(key=lambda item: item.get("run_id", ""), reverse=True)
    return manifest


def _manifest_entry_from_dir(run_dir: Path) -> dict[str, Any] | None:
    run_id = run_dir.name
    dash_path = run_dir / "dashboard.json"
    summary_path = run_dir / "summary.json"

    summary_data: dict[str, Any] = {}
    dashboard_summary: dict[str, Any] | None = None

    highlights: dict[str, Any] | None = None

    if dash_path.exists():
        try:
            dash_payload = json.loads(dash_path.read_text())
            dashboard_summary = dash_payload.get("summary")
            highlights = dash_payload.get("highlights")
        except Exception:
            dashboard_summary = None

    if summary_path.exists():
        try:
            summary_data = json.loads(summary_path.read_text())
        except Exception:
            summary_data = {}

    if not dashboard_summary and summary_data:
        dashboard_summary = summary_data.get("dashboard")

    if not dashboard_summary:
        dashboard_summary = {}

    if not summary_data and not dashboard_summary:
        return None

    if not highlights:
        highlights = _extract_highlights(dashboard_summary)

    return {
        "run_id": run_id,
        "path": f"../{run_id}/dashboard.json",
        "metric": summary_data.get("metric"),
        "results_count": summary_data.get("results_count"),
        "started_at": summary_data.get("started_at"),
        "duration_sec": summary_data.get("duration_sec"),
        "summary": dashboard_summary,
        "highlights": highlights,
    }


def _numeric_values(rows: Iterable[MetricRow], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        stats = row.get("stats") or {}
        val = stats.get(key)
        if val is None:
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if math.isnan(fval):
            continue
        values.append(fval)
    return values


def _best_entry(rows: list[MetricRow], key: str, direction: str) -> dict[str, Any] | None:
    comparator = max if direction != "min" else min
    values = [(row, row.get("stats", {}).get(key)) for row in rows]
    filtered: list[tuple[MetricRow, float]] = []
    for row, val in values:
        if val is None:
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if math.isnan(fval):
            continue
        filtered.append((row, fval))
    if not filtered:
        return None
    chosen_row, chosen_val = comparator(filtered, key=lambda item: item[1])
    return {
        "collection": chosen_row.get("collection"),
        "symbol": chosen_row.get("symbol"),
        "timeframe": chosen_row.get("timeframe"),
        "strategy": chosen_row.get("strategy"),
        "value": chosen_val,
    }


def _build_summary(rows: list[MetricRow]) -> dict[str, Any]:
    counts = {
        "results": len(rows),
        "collections": len({row.get("collection") for row in rows if row.get("collection")}),
        "symbols": len({row.get("symbol") for row in rows if row.get("symbol")}),
        "strategies": len({row.get("strategy") for row in rows if row.get("strategy")}),
        "timeframes": len({row.get("timeframe") for row in rows if row.get("timeframe")}),
    }

    metrics: dict[str, Any] = {}
    for key in METRIC_KEYS:
        values = _numeric_values(rows, key)
        if not values:
            continue
        direction = METRIC_DIRECTIONS.get(key, "max")
        best = _best_entry(rows, key, direction)
        metrics[key] = {
            "mean": float(sum(values) / len(values)) if values else float("nan"),
            "min": float(min(values)),
            "max": float(max(values)),
            "best": best,
            "direction": direction,
        }

    return {
        "counts": counts,
        "metrics": metrics,
    }


def _extract_highlights(summary: dict[str, Any] | None) -> dict[str, Any]:
    highlights: dict[str, Any] = {}
    if not summary:
        return highlights
    metrics = summary.get("metrics") or {}
    for key in HIGHLIGHT_METRICS:
        data = metrics.get(key)
        if not isinstance(data, dict):
            continue
        value = None
        best = data.get("best")
        if isinstance(best, dict):
            value = best.get("value")
        if value is None:
            value = data.get("max")
        if value is None:
            value = data.get("mean")
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(value_float):
            continue
        highlights[key] = {
            "value": value_float,
            "direction": data.get("direction", METRIC_DIRECTIONS.get(key, "max")),
        }
        if isinstance(best, dict):
            for extra in ("collection", "symbol", "timeframe", "strategy"):
                if best.get(extra) is not None:
                    highlights[key][extra] = best.get(extra)
    return highlights


def _detect_downloads(out_dir: Path) -> list[str]:
    return [name for name in DOWNLOAD_FILE_CANDIDATES if (Path(out_dir) / name).exists()]


class DashboardReporter:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def export(self, payload: dict[str, Any]) -> None:
        rows = payload.get("rows", [])
        if not rows:
            return
        payload.setdefault("downloads", _detect_downloads(self.out_dir))
        html_content = self._render_html(payload)
        (self.out_dir / "dashboard.html").write_text(html_content)
        (self.out_dir / "dashboard.json").write_text(safe_json_dumps(payload, indent=2))

    def _render_html(self, payload: dict[str, Any]) -> str:
        payload_json = safe_json_dumps(payload)

        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Backtest Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    body {{ background-color: rgb(15 23 42); color: rgb(226 232 240); font-family: 'Inter', sans-serif; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 1.5rem; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 1rem; }}
    .card {{ background: rgb(30 41 59); border-radius: 0.75rem; padding: 1rem; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.35); }}
    .card-title {{ font-size: 0.95rem; color: rgb(148 163 184); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.75rem; }}
    .metric-value {{ font-size: 1.75rem; font-weight: 600; }}
    .metric-sub {{ font-size: 0.85rem; color: rgb(148 163 184); }}
    .metric-label {{ font-size: 0.75rem; color: rgb(100 116 139); margin-top: 0.25rem; }}
    .counts {{ margin-top: 1.5rem; background: rgb(30 41 59); border-radius: 0.75rem; padding: 1rem; }}
    .counts ul {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.5rem; list-style: none; padding: 0; margin: 0; }}
    .counts li {{ display: flex; justify-content: space-between; font-size: 0.95rem; }}
    .highlights {{ margin-top: 1.5rem; background: rgb(30 41 59); border-radius: 0.75rem; padding: 1rem; }}
    .highlights-list {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem; list-style: none; padding: 0; margin: 0; }}
    .highlight-card {{ background: rgba(15, 23, 42, 0.7); border-radius: 0.5rem; padding: 0.75rem; box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.15); }}
    .highlight-card h3 {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: rgb(148, 163, 184); margin-bottom: 0.35rem; }}
    .highlight-card .value {{ font-size: 1.4rem; font-weight: 600; }}
    .highlight-card .meta {{ font-size: 0.75rem; color: rgb(148, 163, 184); margin-top: 0.25rem; }}
    .controls {{ margin: 1.5rem 0 1rem; display: flex; gap: 1rem; align-items: flex-end; flex-wrap: wrap; }}
    .control-group {{ display: flex; flex-direction: column; gap: 0.35rem; min-width: 160px; }}
    .control-group label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: rgb(148 163 184); }}
    .control-group select {{ background: rgb(30 41 59); border: 1px solid rgba(148, 163, 184, 0.4); border-radius: 0.5rem; padding: 0.5rem 0.75rem; color: inherit; }}
    .table-wrapper {{ margin-top: 2rem; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    thead {{ background: rgb(30 41 59); }}
    th, td {{ padding: 0.65rem 0.75rem; text-align: left; border-bottom: 1px solid rgba(148, 163, 184, 0.2); }}
    tr:hover {{ background: rgba(148, 163, 184, 0.1); }}
    .search {{ margin: 1.5rem 0; }}
    .search input {{ width: 100%; padding: 0.65rem 0.75rem; border-radius: 0.5rem; background: rgb(30 41 59); border: 1px solid rgba(148, 163, 184, 0.25); color: inherit; }}
    #scatter {{ height: 420px; margin-top: 2rem; border-radius: 0.75rem; background: rgb(30 41 59); padding: 0.75rem; }}
   #histogram {{ height: 320px; margin-top: 1.5rem; border-radius: 0.75rem; background: rgb(30 41 59); padding: 0.75rem; }}
    .downloads {{ margin-top: 1.5rem; background: rgb(30 41 59); border-radius: 0.75rem; padding: 1rem; }}
    .downloads ul {{ display: flex; flex-wrap: wrap; gap: 0.75rem; list-style: none; padding: 0; margin: 0; }}
    .downloads li {{ margin: 0; }}
    .downloads a {{ display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.5rem 0.85rem; border-radius: 9999px; background: rgba(148, 163, 184, 0.15); color: rgb(226 232 240); text-decoration: none; font-size: 0.8rem; transition: background 0.2s ease; }}
    .downloads a:hover {{ background: rgba(148, 163, 184, 0.3); }}
    .history {{ margin-top: 2rem; background: rgb(30 41 59); border-radius: 0.75rem; padding: 1rem; }}
    .history-table {{ overflow-x: auto; margin-top: 1rem; }}
    .history-table table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    .history-table thead {{ background: rgba(148, 163, 184, 0.1); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: rgb(148, 163, 184); }}
    .history-table th, .history-table td {{ padding: 0.65rem 0.75rem; text-align: left; border-bottom: 1px solid rgba(148, 163, 184, 0.2); }}
    .history-run-label {{ display: flex; align-items: center; gap: 0.5rem; }}
    .history-run-label span.tag {{ font-size: 0.7rem; text-transform: uppercase; background: rgba(56, 189, 248, 0.2); color: rgb(125, 211, 252); border-radius: 9999px; padding: 0.15rem 0.55rem; }}
    .compare-controls {{ margin-top: 1rem; display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; }}
    .compare-controls label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: rgb(148 163 184); }}
    .compare-controls select {{ background: rgb(30 41 59); border: 1px solid rgba(148, 163, 184, 0.4); border-radius: 0.5rem; padding: 0.45rem 0.75rem; color: inherit; }}
    .compare-controls button {{ background: rgba(148, 163, 184, 0.15); border: none; border-radius: 0.5rem; padding: 0.45rem 0.9rem; color: inherit; font-size: 0.8rem; cursor: pointer; }}
    .compare-controls button:hover {{ background: rgba(148, 163, 184, 0.3); }}
    .compare-output {{ margin-top: 1rem; overflow-x: auto; }}
    .compare-output table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    .compare-output thead {{ background: rgba(148, 163, 184, 0.1); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: rgb(148, 163, 184); }}
    .compare-output th, .compare-output td {{ padding: 0.6rem 0.75rem; text-align: left; border-bottom: 1px solid rgba(148, 163, 184, 0.2); }}
    .compare-output .compare-empty {{ color: rgb(100, 116, 139); font-size: 0.85rem; }}
  </style>
</head>
<body>
  <div class="container">
    <header class="mb-6">
      <h1 class="text-3xl font-semibold">Backtest Dashboard</h1>
      <p class="text-slate-400 text-sm" id="run-meta">Interactive overview of the latest run, highlighting Omega, Tail Ratio, Pain Index, and more.</p>
    </header>
    <section class="downloads" id="downloads-section">
      <h2 class="text-lg font-semibold mb-2">Downloads</h2>
      <ul id="downloads-list"></ul>
    </section>
    <section class="controls">
      <div class="control-group">
        <label for="run-selector">Run</label>
        <select id="run-selector"></select>
      </div>
      <div class="control-group">
        <label for="x-metric">X Metric</label>
        <select id="x-metric"></select>
      </div>
      <div class="control-group">
        <label for="y-metric">Y Metric</label>
        <select id="y-metric"></select>
      </div>
      <div class="control-group">
        <label for="hist-metric">Histogram Metric</label>
        <select id="hist-metric"></select>
      </div>
    </section>
    <section class="cards" id="cards-container"></section>
    <section class="counts">
      <h2 class="text-lg font-semibold mb-2">Coverage</h2>
      <ul id="counts-list"></ul>
    </section>
    <section class="highlights">
      <h2 class="text-lg font-semibold mb-2">Highlights</h2>
      <ul class="highlights-list" id="highlights-list"></ul>
    </section>
    <section id="scatter"></section>
    <section id="histogram"></section>
    <section class="search">
      <input type="search" id="search" placeholder="Filter by symbol, strategy, timeframe..." />
    </section>
    <section class="table-wrapper">
      <table id="results-table">
        <thead>
          <tr>
            <th>Collection</th>
            <th>Symbol</th>
            <th>Timeframe</th>
            <th>Strategy</th>
            <th>Metric</th>
            <th>Metric Value</th>
            <th>Sharpe</th>
            <th>Sortino</th>
            <th>Omega</th>
            <th>Tail Ratio</th>
            <th>Profit</th>
            <th>Pain Index</th>
            <th>Max DD</th>
            <th>CAGR</th>
            <th>Calmar</th>
          </tr>
        </thead>
        <tbody id="table-body"></tbody>
      </table>
    </section>
    <section class="history" id="history-section">
      <h2 class="text-lg font-semibold">Run History &amp; Comparison</h2>
      <p class="text-slate-400 text-sm mt-1">Select runs to compare aggregate metrics across backtests.</p>
      <div class="history-table">
        <table id="run-history-table">
          <thead>
            <tr>
              <th>Select</th>
              <th>Run</th>
              <th>Metric</th>
              <th>Best</th>
              <th>Results</th>
              <th>Started</th>
            </tr>
          </thead>
          <tbody id="run-history-body"></tbody>
        </table>
      </div>
      <div class="compare-controls">
        <label for="compare-metric">Compare Metric</label>
        <select id="compare-metric"></select>
        <button id="clear-compare" type="button">Clear Selection</button>
      </div>
      <div class="compare-output" id="compare-output">
        <p class="compare-empty">Select runs above to compare metrics.</p>
      </div>
    </section>
  </div>
  <script id="dashboard-data" type="application/json">{payload_json}</script>
  <script>
    (function() {{
      const initialData = JSON.parse(document.getElementById('dashboard-data').textContent);
      const fallbackMetrics = ['omega', 'tail_ratio', 'pain_index', 'profit', 'sharpe'];
      let manifest = [];

      function toManifestEntry(data) {{
        if (!data || !data.run_id) return null;
        const highlights = deriveHighlights(data);
        return {{
          run_id: data.run_id,
          path: data.path || 'dashboard.json',
          metric: data.metric,
          results_count: data.results_count,
          started_at: data.started_at || (data.meta && data.meta.started_at),
          duration_sec: data.duration_sec || (data.meta && data.meta.duration_sec),
          summary: data.summary,
          highlights,
        }};
      }}

      function updateManifest(entries) {{
        if (!Array.isArray(entries)) return;
        const merged = new Map(manifest.map(item => [item.run_id, item]));
        entries.forEach(entry => {{
          if (!entry || !entry.run_id) return;
          const current = merged.get(entry.run_id) || {{}};
          const next = {{ ...current, ...entry }};
          if (entry.path === 'dashboard.json' && current.path && current.path !== 'dashboard.json') {{
            next.path = current.path;
          }}
          merged.set(entry.run_id, next);
        }});
        manifest = Array.from(merged.values()).sort((a, b) => (b.run_id || '').localeCompare(a.run_id || ''));
      }}

      updateManifest(Array.isArray(initialData.runs) ? initialData.runs : []);
      const currentEntry = toManifestEntry(initialData);
      if (currentEntry) {{
        currentEntry.path = 'dashboard.json';
        updateManifest([currentEntry]);
      }}
      if (!manifest.some(run => run.run_id === initialData.run_id)) {{
        manifest.push({{ run_id: initialData.run_id, path: 'dashboard.json' }});
      }}

      const selectedCompareRuns = new Set();

      const cardsContainer = document.getElementById('cards-container');
      const countsList = document.getElementById('counts-list');
      const highlightsList = document.getElementById('highlights-list');
      const tableBody = document.getElementById('table-body');
      const searchInput = document.getElementById('search');
      const runSelector = document.getElementById('run-selector');
      const runMeta = document.getElementById('run-meta');
      const xMetricSelect = document.getElementById('x-metric');
      const yMetricSelect = document.getElementById('y-metric');
      const histMetricSelect = document.getElementById('hist-metric');
      const compareMetricSelect = document.getElementById('compare-metric');
      const downloadsList = document.getElementById('downloads-list');
      const historyBody = document.getElementById('run-history-body');
      const compareOutput = document.getElementById('compare-output');
      const clearCompareButton = document.getElementById('clear-compare');

      let currentData = initialData;
      let availableMetrics = Array.isArray(initialData.available_metrics) && initialData.available_metrics.length
        ? initialData.available_metrics.slice()
        : Object.keys((initialData.summary && initialData.summary.metrics) || {{}});
      if (!availableMetrics.length) {{
        availableMetrics = fallbackMetrics;
      }}
      availableMetrics = computeAvailableMetrics(availableMetrics);

      function htmlEscape(value) {{
        return String(value)
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;')
          .replace(/'/g, '&#39;');
      }}

      function toNumber(value) {{
        if (value === null || value === undefined) {{
          return NaN;
        }}
        const numeric = Number(value);
        return Number.isFinite(numeric) ? numeric : NaN;
      }}

      function formatNumber(value, decimals = 4) {{
        const numeric = toNumber(value);
        if (!Number.isFinite(numeric)) {{
          return 'NA';
        }}
        return numeric.toFixed(decimals);
      }}

      function toTitleCase(value) {{
        if (!value) return '';
        return String(value).replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase());
      }}

      function deriveHighlights(data) {{
        if (data && data.highlights && Object.keys(data.highlights).length) {{
          return data.highlights;
        }}
        const summary = data && data.summary ? data.summary : {{}};
        const metrics = summary.metrics || {{}};
        const derived = {{}};
        Object.entries(metrics).forEach(([metric, meta]) => {{
          if (!meta || typeof meta !== 'object') return;
          const best = meta.best || {{}};
          let value = typeof best.value === 'number' ? best.value : meta.max ?? meta.mean;
          if (!Number.isFinite(value)) return;
          const entry = {{ value, direction: meta.direction || 'max' }};
          ['collection', 'symbol', 'timeframe', 'strategy'].forEach(key => {{
            if (best && typeof best === 'object' && best[key] !== undefined) {{
              entry[key] = best[key];
            }}
          }});
          derived[metric] = entry;
        }});
        return derived;
      }}

      function computeAvailableMetrics(source) {{
        const base = Array.isArray(source) ? source.slice() : [];
        const seen = new Set(base);
        manifest.forEach(entry => {{
          if (!entry) return;
          const summaryMetrics = entry.summary && entry.summary.metrics;
          if (summaryMetrics && typeof summaryMetrics === 'object') {{
            Object.keys(summaryMetrics).forEach(metric => {{
              if (!seen.has(metric)) {{
                base.push(metric);
                seen.add(metric);
              }}
            }});
          }}
          const highlightMetrics = entry.highlights || {{}};
          if (highlightMetrics && typeof highlightMetrics === 'object') {{
            Object.keys(highlightMetrics).forEach(metric => {{
              if (!seen.has(metric)) {{
                base.push(metric);
                seen.add(metric);
              }}
            }});
          }}
        }});
        fallbackMetrics.forEach(metric => {{
          if (!seen.has(metric)) {{
            base.push(metric);
            seen.add(metric);
          }}
        }});
        return base;
      }}

      function metricsChanged(next) {{
        if (!Array.isArray(next)) return false;
        if (next.length !== availableMetrics.length) return true;
        for (let idx = 0; idx < next.length; idx += 1) {{
          if (next[idx] !== availableMetrics[idx]) return true;
        }}
        return false;
      }}

      function renderDownloads(downloads, runId) {{
        if (!downloadsList) return;
        const files = Array.isArray(downloads) ? downloads.filter(Boolean) : [];
        const runEntry = manifest.find(entry => entry && entry.run_id === runId);
        let basePath = '';
        if (runEntry && typeof runEntry.path === 'string' && runEntry.path.length) {{
          const rawPath = runEntry.path;
          if (rawPath.endsWith('dashboard.json')) {{
            basePath = rawPath.slice(0, Math.max(0, rawPath.lastIndexOf('dashboard.json')));
          }} else {{
            const slashIdx = rawPath.lastIndexOf('/');
            basePath = slashIdx >= 0 ? rawPath.slice(0, slashIdx + 1) : '';
          }}
        }}
        if (!files.length) {{
          downloadsList.innerHTML = '<li class="text-slate-500 text-sm">No downloads available</li>';
          return;
        }}
        const items = files.map(file => {{
          const href = basePath ? `${{basePath}}${{file}}` : file;
          return `<li><a href="${{htmlEscape(href)}}" download>${{htmlEscape(file)}}</a></li>`;
        }});
        downloadsList.innerHTML = items.join('');
      }}

      function pruneSelectedRuns() {{
        const known = new Set(manifest.map(entry => entry && entry.run_id).filter(Boolean));
        Array.from(selectedCompareRuns).forEach(runId => {{
          if (!known.has(runId)) {{
            selectedCompareRuns.delete(runId);
          }}
        }});
      }}

      function formatTimestamp(value) {{
        if (!value) return '—';
        try {{
          const date = new Date(value);
          if (!Number.isNaN(date.getTime())) {{
            return date.toLocaleString();
          }}
        }} catch (err) {{
          /* ignore */
        }}
        return String(value);
      }}

      function getMetricDetails(entry, metric) {{
        if (!entry || !metric) return {{ value: NaN, direction: 'max' }};
        const highlights = entry.highlights || {{}};
        const highlight = highlights[metric];
        if (highlight && typeof highlight === 'object') {{
          const val = toNumber(highlight.value);
          if (Number.isFinite(val)) {{
            return {{ value: val, direction: highlight.direction || 'max' }};
          }}
        }} else {{
          const val = toNumber(highlight);
          if (Number.isFinite(val)) {{
            return {{ value: val, direction: 'max' }};
          }}
        }}
        const summaryMetrics = entry.summary && entry.summary.metrics;
        const meta = summaryMetrics && summaryMetrics[metric];
        if (meta && typeof meta === 'object') {{
          const best = meta.best || {{}};
          const bestVal = toNumber(best.value);
          if (Number.isFinite(bestVal)) {{
            return {{ value: bestVal, direction: meta.direction || 'max' }};
          }}
          const maxVal = toNumber(meta.max);
          if (Number.isFinite(maxVal)) {{
            return {{ value: maxVal, direction: meta.direction || 'max' }};
          }}
          const meanVal = toNumber(meta.mean);
          if (Number.isFinite(meanVal)) {{
            return {{ value: meanVal, direction: meta.direction || 'max' }};
          }}
        }}
        return {{ value: NaN, direction: 'max' }};
      }}

      function renderRunHistory() {{
        if (!historyBody) return;
        pruneSelectedRuns();
        if (!manifest.length) {{
          historyBody.innerHTML = '<tr><td class="text-slate-500 text-sm" colspan="6">No historical runs available</td></tr>';
          return;
        }}
        const rows = manifest.map(entry => {{
          const runId = entry.run_id || '';
          const metricName = entry.metric || currentData.metric;
          const metricDetails = getMetricDetails(entry, metricName);
          const checkboxId = `compare-${{runId.replace(/[^a-zA-Z0-9_-]/g, '-') || 'current'}}`;
          const checked = selectedCompareRuns.has(runId) ? ' checked' : '';
          const metricLabel = metricName ? toTitleCase(metricName) : '—';
          const metricValue = Number.isFinite(metricDetails.value) ? formatNumber(metricDetails.value) : 'NA';
          const tag = runId === currentData.run_id ? '<span class="tag">Current</span>' : '';
          return `
            <tr>
              <td><input type="checkbox" id="${{htmlEscape(checkboxId)}}" data-run-id="${{htmlEscape(runId)}}"${{checked}}></td>
              <td>
                <label class="history-run-label" for="${{htmlEscape(checkboxId)}}">
                  <span>${{htmlEscape(runId)}}</span>
                  ${{tag}}
                </label>
              </td>
              <td>${{htmlEscape(metricLabel)}}</td>
              <td>${{metricValue}}</td>
              <td>${{entry.results_count ?? '—'}}</td>
              <td>${{htmlEscape(formatTimestamp(entry.started_at))}}</td>
            </tr>
          `;
        }});
        historyBody.innerHTML = rows.join('');
        historyBody.querySelectorAll('input[type="checkbox"]').forEach(input => {{
          input.addEventListener('change', event => {{
            const target = event.target;
            const runId = target.getAttribute('data-run-id');
            if (!runId) return;
            if (target.checked) {{
              selectedCompareRuns.add(runId);
            }} else {{
              selectedCompareRuns.delete(runId);
            }}
            renderComparison();
          }});
        }});
      }}

      function renderComparison() {{
        if (!compareOutput) return;
        if (!selectedCompareRuns.size) {{
          compareOutput.innerHTML = '<p class="compare-empty">Select runs above to compare metrics.</p>';
          return;
        }}
        const runEntries = manifest.filter(entry => entry && selectedCompareRuns.has(entry.run_id));
        if (!runEntries.length) {{
          compareOutput.innerHTML = '<p class="compare-empty">Selected runs are no longer available.</p>';
          return;
        }}
        const selectedMetric = resolveMetric(compareMetricSelect, 'sharpe');
        const extraMetrics = ['omega', 'tail_ratio', 'pain_index', 'profit'];
        const metricColumns = [selectedMetric, ...extraMetrics.filter(metric => metric !== selectedMetric && availableMetrics.includes(metric))];
        if (!metricColumns.length) {{
          metricColumns.push(selectedMetric);
        }}
        const header = `
          <thead>
            <tr>
              <th>Run</th>
              <th>Configured</th>
              <th>Value</th>
              <th>Results</th>
              <th>Started</th>
              ${{metricColumns.map(metric => `<th>${{htmlEscape(toTitleCase(metric))}}</th>`).join('')}}
            </tr>
          </thead>
        `;
        const rows = runEntries.map(entry => {{
          const runId = entry.run_id || '';
          const configuredMetric = entry.metric || currentData.metric;
          const configuredDetails = getMetricDetails(entry, configuredMetric);
          const configuredLabel = configuredMetric ? toTitleCase(configuredMetric) : '—';
          const configuredValue = Number.isFinite(configuredDetails.value) ? formatNumber(configuredDetails.value) : 'NA';
          const compareCells = metricColumns.map(metric => {{
            const details = getMetricDetails(entry, metric);
            return `<td>${{Number.isFinite(details.value) ? formatNumber(details.value) : 'NA'}}</td>`;
          }}).join('');
          const tag = runId === currentData.run_id ? '<span class="tag">Current</span>' : '';
          return `
            <tr>
              <td><div class="history-run-label"><span>${{htmlEscape(runId)}}</span>${{tag}}</div></td>
              <td>${{htmlEscape(configuredLabel)}}</td>
              <td>${{configuredValue}}</td>
              <td>${{entry.results_count ?? '—'}}</td>
              <td>${{htmlEscape(formatTimestamp(entry.started_at))}}</td>
              ${{compareCells}}
            </tr>
          `;
        }}).join('');
        compareOutput.innerHTML = `<table>${{header}}<tbody>${{rows}}</tbody></table>`;
      }}

      function populateMetricSelectors(preserve = false) {{
        const previous = {{
          x: xMetricSelect && xMetricSelect.value,
          y: yMetricSelect && yMetricSelect.value,
          hist: histMetricSelect && histMetricSelect.value,
          compare: compareMetricSelect && compareMetricSelect.value,
        }};
        const optionsHtml = availableMetrics
          .map(metric => `<option value="${{htmlEscape(metric)}}">${{htmlEscape(toTitleCase(metric))}}</option>`)
          .join('');
        [xMetricSelect, yMetricSelect, histMetricSelect].forEach(select => {{
          if (!select) return;
          select.innerHTML = optionsHtml;
        }});
        if (compareMetricSelect) {{
          compareMetricSelect.innerHTML = optionsHtml;
        }}
        if (xMetricSelect) {{
          const fallback = availableMetrics.includes('omega') ? 'omega' : availableMetrics[0];
          const chosen = preserve && previous.x && availableMetrics.includes(previous.x) ? previous.x : fallback;
          xMetricSelect.value = chosen;
        }}
        if (yMetricSelect) {{
          const fallback = availableMetrics.includes('tail_ratio') ? 'tail_ratio' : availableMetrics[1] || availableMetrics[0];
          const chosen = preserve && previous.y && availableMetrics.includes(previous.y) ? previous.y : fallback;
          yMetricSelect.value = chosen;
        }}
        if (histMetricSelect) {{
          const fallback = availableMetrics.includes('pain_index') ? 'pain_index' : availableMetrics[2] || availableMetrics[0];
          const chosen = preserve && previous.hist && availableMetrics.includes(previous.hist) ? previous.hist : fallback;
          histMetricSelect.value = chosen;
        }}
        if (compareMetricSelect) {{
          const fallback = availableMetrics.includes('sharpe') ? 'sharpe' : availableMetrics[0];
          const chosen = preserve && previous.compare && availableMetrics.includes(previous.compare) ? previous.compare : fallback;
          compareMetricSelect.value = chosen;
        }}
      }}

      function resolveMetric(select, fallback) {{
        const value = select && select.value;
        if (value && availableMetrics.includes(value)) return value;
        if (availableMetrics.includes(fallback)) return fallback;
        return availableMetrics[0];
      }}

      function renderCards(summary) {{
        const metrics = (summary && summary.metrics) || {{}};
        const fragments = [];
        Object.entries(metrics).forEach(([name, data]) => {{
          const display = toTitleCase(name);
          const best = data.best || {{}};
          const direction = data.direction === 'min' ? 'Lowest' : 'Highest';
          const bestValue = typeof best.value === 'number' ? formatNumber(best.value) : 'NA';
          const meanValue = formatNumber(data.mean);
          const targetParts = [best.collection, best.symbol].filter(Boolean);
          let target = targetParts.join(' / ');
          if (best.timeframe) {{
            target = `${{target}} @ ${{best.timeframe}}`;
          }}
          fragments.push(`
                <div class="card">
                  <div class="card-title">${{htmlEscape(display)}}</div>
                  <div class="card-body">
                    <div class="metric-value">${{meanValue}}</div>
                    <div class="metric-sub">${{direction}}: ${{bestValue}}</div>
                    <div class="metric-label">${{htmlEscape(target || 'Not available')}}</div>
                  </div>
                </div>
              `);
        }});
        cardsContainer.innerHTML = fragments.join('');
      }}

      function renderCounts(summary) {{
        const counts = (summary && summary.counts) || {{}};
        const entries = Object.entries(counts).map(([label, value]) => {{
          const display = toTitleCase(label);
          return `<li><span>${{htmlEscape(display)}}</span><strong>${{value}}</strong></li>`;
        }});
        countsList.innerHTML = entries.join('');
      }}

      function renderHighlights(highlights) {{
        if (!highlightsList) return;
        const entries = Object.entries(highlights || {{}});
        if (!entries.length) {{
          highlightsList.innerHTML = '<li class="text-slate-500 text-sm">No highlights available</li>';
          return;
        }}
        const fragments = entries.map(([metric, data]) => {{
          const value = typeof data === 'object' && data !== null
            ? toNumber(data.value)
            : toNumber(data);
          const formatted = formatNumber(value);
          const metaParts = [];
          if (data && typeof data === 'object') {{
            const location = [data.collection, data.symbol].filter(Boolean).join(' / ');
            if (location) metaParts.push(location);
            if (data.timeframe) metaParts.push(`@ ${{data.timeframe}}`);
            if (data.strategy) metaParts.push(data.strategy);
          }}
          const meta = metaParts.join(' • ');
          return `
            <li class="highlight-card">
              <h3>${{htmlEscape(toTitleCase(metric))}}</h3>
              <div class="value">${{formatted}}</div>
              <div class="meta">${{meta ? htmlEscape(meta) : '—'}}</div>
            </li>
          `;
        }});
        highlightsList.innerHTML = fragments.join('');
      }}

      function renderTable(rows) {{
        const fragments = rows
          .slice()
          .sort((a, b) => (b.metric_value ?? 0) - (a.metric_value ?? 0))
          .map(row => {{
            const stats = row.stats || {{}};
            const searchBlob = [row.collection, row.symbol, row.timeframe, row.strategy, row.metric]
              .filter(Boolean)
              .map(part => String(part).toLowerCase())
              .join(' ');
            return `
                  <tr data-search="${{htmlEscape(searchBlob)}}">
                    <td>${{htmlEscape(row.collection ?? '')}}</td>
                    <td>${{htmlEscape(row.symbol ?? '')}}</td>
                    <td>${{htmlEscape(row.timeframe ?? '')}}</td>
                    <td>${{htmlEscape(row.strategy ?? '')}}</td>
                    <td>${{htmlEscape(row.metric ?? '')}}</td>
                    <td>${{formatNumber(row.metric_value, 6)}}</td>
                    <td>${{formatNumber(stats.sharpe)}}</td>
                    <td>${{formatNumber(stats.sortino)}}</td>
                    <td>${{formatNumber(stats.omega)}}</td>
                    <td>${{formatNumber(stats.tail_ratio)}}</td>
                    <td>${{formatNumber(stats.profit)}}</td>
                    <td>${{formatNumber(stats.pain_index)}}</td>
                    <td>${{formatNumber(stats.max_drawdown)}}</td>
                    <td>${{formatNumber(stats.cagr)}}</td>
                    <td>${{formatNumber(stats.calmar)}}</td>
                  </tr>
                `;
          }});
        tableBody.innerHTML = fragments.join('');
      }}

      function renderScatter(rows) {{
        const xMetric = resolveMetric(xMetricSelect, 'omega');
        const yMetric = resolveMetric(yMetricSelect, 'tail_ratio');
        const colorMetric = resolveMetric(histMetricSelect, 'pain_index');
        const scatterRows = rows.filter(r => {{
          const stats = r.stats || {{}};
          const xVal = toNumber(stats[xMetric]);
          const yVal = toNumber(stats[yMetric]);
          return Number.isFinite(xVal) && Number.isFinite(yVal);
        }});
        const xValues = scatterRows.map(r => toNumber(r.stats[xMetric]));
        const yValues = scatterRows.map(r => toNumber(r.stats[yMetric]));
        const colorValues = scatterRows.map(r => {{
          const val = toNumber(r.stats[colorMetric]);
          return Number.isFinite(val) ? val : 0;
        }});
        const textValues = scatterRows.map(r => {{
          const stats = r.stats || {{}};
          return `${{r.collection}}/${{r.symbol}} • ${{r.strategy}} @ ${{r.timeframe}}<br>${{toTitleCase(xMetric)}}: ${{formatNumber(stats[xMetric])}}<br>${{toTitleCase(yMetric)}}: ${{formatNumber(stats[yMetric])}}`;
        }});
        const trace = {{
          x: xValues,
          y: yValues,
          text: textValues,
          mode: 'markers',
          hovertemplate: '%{{text}}<extra></extra>',
          marker: {{
            size: 10,
            color: colorValues,
            colorscale: 'Turbo',
            showscale: true,
            colorbar: {{ title: toTitleCase(colorMetric) }}
          }}
        }};
        const layout = {{
          margin: {{ l: 60, r: 30, t: 40, b: 60 }},
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          xaxis: {{ title: toTitleCase(xMetric) }},
          yaxis: {{ title: toTitleCase(yMetric) }}
        }};
        Plotly.react('scatter', [trace], layout, {{ displayModeBar: false, responsive: true }});
      }}

      function renderHistogram(rows) {{
        const metric = resolveMetric(histMetricSelect, 'pain_index');
        const values = rows
          .map(r => toNumber(r.stats && r.stats[metric]))
          .filter(val => Number.isFinite(val));
        const trace = {{
          x: values,
          type: 'histogram',
          marker: {{ color: '#38bdf8' }},
          opacity: 0.75
        }};
        const layout = {{
          margin: {{ l: 60, r: 30, t: 40, b: 40 }},
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          xaxis: {{ title: toTitleCase(metric) }},
          yaxis: {{ title: 'Frequency' }}
        }};
        Plotly.react('histogram', values.length ? [trace] : [], layout, {{ displayModeBar: false, responsive: true }});
      }}

      function renderRunMeta(meta, runId) {{
        if (!runMeta) return;
        const parts = [];
        if (meta && meta.metric) parts.push(`metric: ${{meta.metric}}`);
        if (meta && Number.isFinite(meta.results_count)) parts.push(`${{meta.results_count}} results`);
        if (meta && meta.started_at) {{
          try {{
            parts.push(new Date(meta.started_at).toLocaleString());
          }} catch (err) {{
            parts.push(meta.started_at);
          }}
        }}
        runMeta.textContent = parts.length ? `${{runId}} – ${{parts.join(' • ')}}` : `Run ${{runId}}`;
      }}

      function renderDashboard(data) {{
        currentData = data;
        if (Array.isArray(data.runs) && data.runs.length) {{
          updateManifest(data.runs);
        }}
        const manifestEntry = toManifestEntry(data);
        if (manifestEntry) {{
          updateManifest([manifestEntry]);
        }}
        const candidateMetrics = Array.isArray(data.available_metrics) && data.available_metrics.length
          ? data.available_metrics.slice()
          : availableMetrics.slice();
        const mergedMetrics = computeAvailableMetrics(candidateMetrics);
        if (metricsChanged(mergedMetrics)) {{
          availableMetrics = mergedMetrics;
          populateMetricSelectors(true);
        }}
        const summary = data.summary || {{}};
        renderCards(summary);
        renderCounts(summary);
        renderTable(data.rows || []);
        renderScatter(data.rows || []);
        renderHistogram(data.rows || []);
        const highlights = deriveHighlights(data);
        renderHighlights(highlights);
        populateRuns(data.run_id);
        renderRunMeta({{
          metric: data.metric,
          results_count: data.results_count,
          started_at: data.started_at || (data.meta && data.meta.started_at),
        }}, data.run_id);
        renderDownloads(data.downloads || [], data.run_id);
        renderRunHistory();
        renderComparison();
        if (searchInput) {{
          searchInput.value = '';
        }}
        document.querySelectorAll('#results-table tbody tr').forEach(row => {{
          row.style.display = '';
        }});
      }}

      function populateRuns(selectedRunId) {{
        if (!runSelector) return;
        const options = manifest
          .slice()
          .sort((a, b) => (b.run_id || '').localeCompare(a.run_id || ''))
          .map(run => {{
            const labelParts = [run.run_id];
            if (run.metric) labelParts.push(run.metric);
            if (run.results_count !== undefined) labelParts.push(`${{run.results_count}} results`);
            const runHighlights = run.highlights || {{}};
            const metricHighlight = run.metric && runHighlights[run.metric]
              ? (typeof runHighlights[run.metric] === 'object' ? runHighlights[run.metric].value : runHighlights[run.metric])
              : undefined;
            if (Number.isFinite(metricHighlight)) {{
              labelParts.push(`${{run.metric}}: ${{Number(metricHighlight).toFixed(3)}}`);
            }}
            const isSelected = run.run_id === selectedRunId ? ' selected' : '';
            return `<option value="${{htmlEscape(run.run_id)}}" data-path="${{htmlEscape(run.path || '')}}"${{isSelected}}>${{htmlEscape(labelParts.join(' – '))}}</option>`;
          }});
        runSelector.innerHTML = options.join('');
      }}

      populateRuns(initialData.run_id);
      populateMetricSelectors(false);
      renderDashboard(initialData);

      if (searchInput) {{
        searchInput.addEventListener('input', (event) => {{
          const query = event.target.value.trim().toLowerCase();
          const rows = document.querySelectorAll('#results-table tbody tr');
          rows.forEach(row => {{
            const haystack = row.getAttribute('data-search') || '';
            row.style.display = haystack.includes(query) ? '' : 'none';
          }});
        }});
      }}

      if (runSelector) {{
        runSelector.addEventListener('change', async (event) => {{
          const runId = event.target.value;
          const selected = manifest.find(item => item.run_id === runId);
          if (!selected) return;
          if (runId === initialData.run_id) {{
            renderDashboard(initialData);
            return;
          }}
          const path = selected.path || `../${{runId}}/dashboard.json`;
          try {{
            const response = await fetch(path);
            if (!response.ok) throw new Error('Failed to load run');
            const payload = await response.json();
            payload.runs = manifest;
            selected.highlights = deriveHighlights(payload);
            renderDashboard(payload);
          }} catch (error) {{
            console.error('Unable to load selected run', error);
          }}
        }});
      }}

      if (compareMetricSelect) {{
        compareMetricSelect.addEventListener('change', () => {{
          renderComparison();
        }});
      }}

      if (clearCompareButton) {{
        clearCompareButton.addEventListener('click', () => {{
          selectedCompareRuns.clear();
          renderRunHistory();
          renderComparison();
        }});
      }}

      [xMetricSelect, yMetricSelect, histMetricSelect].forEach(select => {{
        if (!select) return;
        select.addEventListener('change', () => {{
          if (!currentData) return;
          renderScatter(currentData.rows || []);
          renderHistogram(currentData.rows || []);
        }});
      }});
    }})();

  </script>
</body>
</html>
        """
