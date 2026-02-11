from __future__ import annotations

import html as html_stdlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..backtest.results_cache import ResultsCache

if TYPE_CHECKING:
    pass


class HTMLReporter:
    def __init__(
        self,
        out_dir: Path,
        results_cache: ResultsCache,
        run_id: str,
        top_n: int = 3,
        inline_css: bool = False,
    ):
        self.out_dir = Path(out_dir)
        self.cache = results_cache
        self.run_id = run_id
        self.top_n = top_n
        self.inline_css = inline_css

    def export(self, best: list[object]):
        def _esc(value: Any) -> str:
            if value is None:
                return ""
            return html_stdlib.escape(str(value), quote=True)

        def _json_for_inline_script(value: Any) -> str:
            """JSON-safe for embedding directly in a <script> tag.

            This prevents `</script>` breakouts by escaping `<` and also guards common HTML parser
            edge cases (`>`, `&`). Values come from configs/strategies and may be user-controlled.
            """

            text = json.dumps(value, ensure_ascii=True)
            return (
                text.replace("<", "\\u003c")
                .replace(">", "\\u003e")
                .replace("&", "\\u0026")
            )

        def _write_plotly_asset() -> str:
            """Write Plotly JS to disk and return relative script src.

            Plotly is loaded from a CDN by default, but the offline HTML mode uses a local copy
            written alongside the report so opening `report.html` works without internet access.
            """

            dest = self.out_dir / "plotly.min.js"
            if dest.exists() and dest.stat().st_size > 0:
                return "plotly.min.js"

            try:
                import importlib.resources as resources

                src = resources.files("plotly").joinpath("package_data/plotly.min.js")
                dest.write_bytes(src.read_bytes())
            except Exception:
                # Fallback: keep report functional without charts if we cannot materialize the asset.
                # (The rest of the HTML is still useful.)
                return "https://cdn.plot.ly/plotly-2.32.0.min.js"

            return "plotly.min.js"

        # Load all rows for top-N sections from results cache
        all_rows = self.cache.list_by_run(self.run_id)
        # Fallback: if the current run reused cached results and didn't write
        # rows with this run_id, synthesize rows from the provided BestResult list.
        if not all_rows and best:
            try:
                from ..backtest.runner import BestResult as _BR  # type: ignore

                tmp: list[dict[str, Any]] = []
                for b in best:
                    if isinstance(b, _BR):
                        tmp.append(
                            {
                                "collection": b.collection,
                                "symbol": b.symbol,
                                "timeframe": b.timeframe,
                                "strategy": b.strategy,
                                "params": b.params,
                                "metric": b.metric_name,
                                "metric_value": float(b.metric_value),
                                "stats": b.stats if isinstance(b.stats, dict) else {},
                            }
                        )
                all_rows = tmp
            except Exception:
                pass
        grouped: dict[tuple, list[dict[str, Any]]] = {}
        for r in all_rows:
            key = (r["collection"], r["symbol"])
            grouped.setdefault(key, []).append(r)

        flat_rows = list(all_rows)
        top_rows = sorted(
            flat_rows, key=lambda x: x.get("metric_value", float("nan")), reverse=True
        )[:5]

        strategy_best: dict[tuple[str, str], dict[str, Any]] = {}
        for row in flat_rows:
            collection = row.get("collection") or "unknown"
            strategy = row.get("strategy")
            if not strategy:
                continue
            key = (collection, strategy)
            if key not in strategy_best or row.get("metric_value", float("nan")) > strategy_best[
                key
            ].get("metric_value", float("nan")):
                strategy_best[key] = row

        metrics_highlight: dict[str, dict[str, Any]] = {}
        metrics_to_track = ["sharpe", "sortino", "omega", "tail_ratio", "profit", "pain_index"]
        for row in flat_rows:
            stats = row.get("stats", {}) or {}
            for metric_name in metrics_to_track:
                val = stats.get(metric_name)
                if val is None:
                    continue
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    continue
                if math.isnan(fval):
                    continue
                current = metrics_highlight.get(metric_name)
                if current is None or fval > current.get("value", float("-inf")):
                    metrics_highlight[metric_name] = {"value": fval, "row": row}

        def strategy_section() -> str:
            if not strategy_best:
                return ""
            rows_html = []
            for key in sorted(strategy_best.keys()):
                row = strategy_best[key]
                rows_html.append(
                    "<tr>"
                    f"<td class='px-3 py-2'>{_esc(row.get('symbol'))}</td>"
                    f"<td class='px-3 py-2'>{_esc(row.get('strategy'))}</td>"
                    f"<td class='px-3 py-2'>{_esc(row.get('timeframe'))}</td>"
                    f"<td class='px-3 py-2'>{_esc(row.get('metric'))}</td>"
                    f"<td class='px-3 py-2'>{row.get('metric_value', float('nan')):.4f}</td>"
                    f"<td class='px-3 py-2'>{_esc(row.get('collection'))}</td>"
                    "</tr>"
                )
            body = "\n".join(rows_html)
            return f"""
            <section class='p-4 rounded-lg bg-slate-800 text-slate-100 shadow'>
                <h2 class='text-lg font-semibold mb-3'>Best Per Strategy</h2>
                <div class='overflow-x-auto'>
                  <table class='min-w-full text-sm text-left text-slate-200'>
                    <thead class='text-xs uppercase bg-slate-700 text-slate-300'>
                      <tr>
                        <th class='px-3 py-2'>Symbol</th>
                        <th class='px-3 py-2'>Strategy</th>
                        <th class='px-3 py-2'>Timeframe</th>
                        <th class='px-3 py-2'>Metric</th>
                        <th class='px-3 py-2'>Value</th>
                        <th class='px-3 py-2'>Collection</th>
                      </tr>
                    </thead>
                    <tbody>{body}</tbody>
                  </table>
                </div>
            </section>
            """

        def metric_section() -> str:
            if not metrics_highlight:
                return ""
            rows_html = []
            for metric_name in metrics_to_track:
                entry = metrics_highlight.get(metric_name)
                if not entry:
                    continue
                row = entry["row"]
                rows_html.append(
                    "<tr>"
                    f"<td class='px-3 py-2'>{_esc(metric_name.title())}</td>"
                    f"<td class='px-3 py-2'>{entry['value']:.4f}</td>"
                    f"<td class='px-3 py-2'>{_esc(row.get('strategy'))}</td>"
                    f"<td class='px-3 py-2'>{_esc(row.get('collection'))}</td>"
                    f"<td class='px-3 py-2'>{_esc(row.get('symbol'))}</td>"
                    f"<td class='px-3 py-2'>{_esc(row.get('timeframe'))}</td>"
                    "</tr>"
                )
            if not rows_html:
                return ""
            body = "\n".join(rows_html)
            return f"""
            <section class='p-4 rounded-lg bg-slate-800 text-slate-100 shadow'>
                <h2 class='text-lg font-semibold mb-3'>Metric Highlights</h2>
                <div class='overflow-x-auto'>
                  <table class='min-w-full text-sm text-left text-slate-200'>
                    <thead class='text-xs uppercase bg-slate-700 text-slate-300'>
                      <tr>
                        <th class='px-3 py-2'>Metric</th>
                        <th class='px-3 py-2'>Value</th>
                        <th class='px-3 py-2'>Strategy</th>
                        <th class='px-3 py-2'>Collection</th>
                        <th class='px-3 py-2'>Symbol</th>
                        <th class='px-3 py-2'>Timeframe</th>
                      </tr>
                    </thead>
                    <tbody>{body}</tbody>
                  </table>
                </div>
            </section>
            """

        def top_section() -> str:
            if not top_rows:
                return ""
            rows_html = []
            for idx, row in enumerate(top_rows, start=1):
                rows_html.append(
                    "<tr>"
                    f"<td class='px-3 py-2'>{idx}</td>"
                    f"<td class='px-3 py-2'>{_esc(row.get('collection'))}</td>"
                    f"<td class='px-3 py-2'>{_esc(row.get('symbol'))}</td>"
                    f"<td class='px-3 py-2'>{_esc(row.get('strategy'))}</td>"
                    f"<td class='px-3 py-2'>{_esc(row.get('timeframe'))}</td>"
                    f"<td class='px-3 py-2'>{_esc(row.get('metric'))}</td>"
                    f"<td class='px-3 py-2'>{row.get('metric_value', float('nan')):.4f}</td>"
                    "</tr>"
                )
            body = "\n".join(rows_html)
            return f"""
            <section class='p-4 rounded-lg bg-slate-800 text-slate-100 shadow'>
                <h2 class='text-lg font-semibold mb-3'>Top Results (Overall)</h2>
                <div class='overflow-x-auto'>
                  <table class='min-w-full text-sm text-left text-slate-200'>
                    <thead class='text-xs uppercase bg-slate-700 text-slate-300'>
                      <tr>
                        <th class='px-3 py-2'>Rank</th>
                        <th class='px-3 py-2'>Collection</th>
                        <th class='px-3 py-2'>Symbol</th>
                        <th class='px-3 py-2'>Strategy</th>
                        <th class='px-3 py-2'>Timeframe</th>
                        <th class='px-3 py-2'>Metric</th>
                        <th class='px-3 py-2'>Value</th>
                      </tr>
                    </thead>
                    <tbody>{body}</tbody>
                  </table>
                </div>
            </section>
            """

        summary_sections = [top_section(), strategy_section(), metric_section()]
        summary_html = "".join([section for section in summary_sections if section])

        def _safe_float(val: Any) -> float | None:
            try:
                f = float(val)
            except (TypeError, ValueError):
                return None
            if math.isnan(f):
                return None
            return f

        chart_data = []
        for row in flat_rows:
            stats = row.get("stats", {}) or {}
            metric_val = _safe_float(row.get("metric_value"))
            sharpe_val = _safe_float(stats.get("sharpe"))
            if metric_val is None or sharpe_val is None:
                continue
            chart_data.append(
                {
                    "collection": row.get("collection"),
                    "symbol": row.get("symbol"),
                    "strategy": row.get("strategy"),
                    "timeframe": row.get("timeframe"),
                    "metric": row.get("metric"),
                    "metric_value": metric_val,
                    "sharpe": sharpe_val,
                }
            )

        scatter_section = ""
        chart_json = _json_for_inline_script(chart_data)
        if chart_data:
            scatter_section = """
            <section class='p-4 rounded-lg bg-slate-800 text-slate-100 shadow'>
                <h2 class='text-lg font-semibold mb-3'>Metric vs. Sharpe</h2>
                <p class='text-sm text-slate-300 mb-3'>Scatter of top evaluations. Hover points to inspect collection, strategy, and timeframe.</p>
                <div id="metric-scatter" style="height:360px;"></div>
            </section>
            """

        detail_records: list[dict[str, Any]] = []
        for idx, row in enumerate(top_rows):
            stats = row.get("stats", {}) or {}
            detail_records.append(
                {
                    "id": f"{idx}-{row.get('collection')}-{row.get('symbol')}-{row.get('strategy')}-{row.get('timeframe')}",
                    "collection": row.get("collection"),
                    "symbol": row.get("symbol"),
                    "strategy": row.get("strategy"),
                    "timeframe": row.get("timeframe"),
                    "metric": row.get("metric"),
                    "metric_value": _safe_float(row.get("metric_value")),
                    "equity_curve": stats.get("equity_curve") or [],
                    "drawdown_curve": stats.get("drawdown_curve") or [],
                    "trades_log": stats.get("trades_log") or [],
                }
            )

        detail_json = _json_for_inline_script(detail_records)

        detail_section = ""
        if detail_records:
            detail_section = """
            <section class='p-4 rounded-lg bg-slate-800 text-slate-100 shadow'>
                <div class='detail-header'>
                  <div>
                    <h2 class='text-lg font-semibold'>Equity & Drawdown Explorer</h2>
                    <p class='text-sm text-slate-300'>Inspect equity, drawdown, and trades for the top-ranked results.</p>
                  </div>
                  <div class='detail-control'>
                    <label for="detail-selector">Result</label>
                    <select id="detail-selector"></select>
                  </div>
                </div>
                <p id="detail-empty" class='text-slate-400 text-sm hidden'>No equity or drawdown data available for the selected result.</p>
                <div id="detail-charts" class='grid gap-4 md:grid-cols-2 detail-charts hidden'>
                  <div>
                    <h3 class='text-sm text-slate-300 mb-2'>Equity Curve</h3>
                    <div id="equity-chart" style="height:320px;"></div>
                  </div>
                  <div>
                    <h3 class='text-sm text-slate-300 mb-2'>Drawdown</h3>
                    <div id="drawdown-chart" style="height:320px;"></div>
                  </div>
                </div>
                <div id="trade-table-container" class='mt-4'></div>
            </section>
            """

        def card_for_row(row: dict[str, Any]) -> str:
            stats = row.get("stats", {}) or {}
            return f"""
            <div class='p-4 rounded-lg bg-slate-800 text-slate-100 shadow'>
                <div class='flex justify-between items-baseline'>
                    <h3 class='text-lg font-semibold'>{_esc(row.get("collection", ""))} / {_esc(row.get("symbol", ""))}</h3>
                    <span class='text-xs px-2 py-1 rounded bg-slate-700'>{_esc(row.get("timeframe", ""))}</span>
                </div>
                <div class='mt-2 text-sm'>
                    <div><span class='font-semibold'>Strategy:</span> {_esc(row.get("strategy", ""))}</div>
                    <div><span class='font-semibold'>Metric:</span> {_esc(row.get("metric", ""))} = {float(row.get("metric_value", float("nan"))):.6f}</div>
                    <div><span class='font-semibold'>Params:</span> <code class='text-xs'>{_esc(row.get("params", {}))}</code></div>
                </div>
                <div class='mt-3 grid grid-cols-2 gap-2 text-sm'>
                    <div>Sharpe: {float(stats.get("sharpe", float("nan"))):.4f}</div>
                    <div>Sortino: {float(stats.get("sortino", float("nan"))):.4f}</div>
                    <div>Omega: {float(stats.get("omega", float("nan"))):.4f}</div>
                    <div>Tail Ratio: {float(stats.get("tail_ratio", float("nan"))):.4f}</div>
                    <div>Profit: {float(stats.get("profit", float("nan"))):.4f}</div>
                    <div>Pain Index: {float(stats.get("pain_index", float("nan"))):.4f}</div>
                    <div>Trades: {int(stats.get("trades", 0))}</div>
                    <div class='col-span-2'>Max DD: {float(stats.get("max_drawdown", float("nan"))):.4f}</div>
                </div>
            </div>
            """

        def table_for_topn(rows: list[dict[str, Any]]) -> str:
            rows = sorted(rows, key=lambda x: x["metric_value"], reverse=True)[: self.top_n]
            body = "\n".join(
                f"<tr><td>{_esc(r['timeframe'])}</td><td>{_esc(r['strategy'])}</td><td>{_esc(r['metric'])}</td><td>{r['metric_value']:.6f}</td>"
                f"<td><code class='text-xs'>{_esc(r['params'])}</code></td>"
                f"<td>{r['stats'].get('sharpe', float('nan')):.4f}</td>"
                f"<td>{r['stats'].get('sortino', float('nan')):.4f}</td>"
                f"<td>{r['stats'].get('omega', float('nan')):.4f}</td>"
                f"<td>{r['stats'].get('tail_ratio', float('nan')):.4f}</td>"
                f"<td>{r['stats'].get('profit', float('nan')):.4f}</td>"
                f"<td>{r['stats'].get('pain_index', float('nan')):.4f}</td>"
                f"<td>{r['stats'].get('trades', 0)}</td>"
                f"<td>{r['stats'].get('max_drawdown', float('nan')):.4f}</td></tr>"
                for r in rows
            )
            return f"""
            <div class='overflow-x-auto mt-3'>
              <table class='min-w-full text-sm text-left text-slate-200'>
                <thead class='text-xs uppercase bg-slate-700 text-slate-300'>
                  <tr>
                    <th class='px-3 py-2'>Timeframe</th>
                    <th class='px-3 py-2'>Strategy</th>
                    <th class='px-3 py-2'>Metric</th>
                    <th class='px-3 py-2'>Value</th>
                    <th class='px-3 py-2'>Params</th>
                    <th class='px-3 py-2'>Sharpe</th>
                    <th class='px-3 py-2'>Sortino</th>
                    <th class='px-3 py-2'>Omega</th>
                    <th class='px-3 py-2'>Tail Ratio</th>
                    <th class='px-3 py-2'>Profit</th>
                    <th class='px-3 py-2'>Pain Index</th>
                    <th class='px-3 py-2'>Trades</th>
                    <th class='px-3 py-2'>Max DD</th>
                  </tr>
                </thead>
                <tbody>{body}</tbody>
              </table>
            </div>
            """

        # Build a card per (collection, symbol) showing the true best across
        # all timeframes/strategies by metric_value
        cards = []
        for key in sorted(grouped.keys()):
            rows = grouped[key]
            if not rows:
                continue
            top = max(rows, key=lambda x: x["metric_value"])  # best overall
            cards.append(
                "<section class='mb-6'>" + card_for_row(top) + table_for_topn(rows) + "</section>"
            )

        plotly_cdn_sri = (
            "sha384-7TVmlZWH60iKX5Uk7lSvQhjtcgw2tkFjuwLcXoRSR4zXTyWFJRm9aPAguMh7CIra"
        )
        plotly_src = _write_plotly_asset() if self.inline_css else "https://cdn.plot.ly/plotly-2.32.0.min.js"
        if plotly_src.startswith("http"):
            plotly_tag = (
                f'<script src="{plotly_src}" integrity="{plotly_cdn_sri}" '
                'crossorigin="anonymous" referrerpolicy="no-referrer"></script>'
            )
        else:
            plotly_tag = f'<script src="{plotly_src}"></script>'

        head_assets = ""
        if self.inline_css:
            css_inline = (
                ":root{--bg:#020617;--panel:#0f172a;--text:#e2e8f0;--muted:#94a3b8} "
                "body{background:var(--bg);color:var(--text);} "
                ".container{max-width:72rem;margin:0 auto;padding:1.5rem} "
                ".btn{padding:.25rem .75rem;border-radius:.375rem;background:#1e293b;color:#e2e8f0} "
                ".grid{display:grid;gap:1rem} "
                ".card{padding:1rem;border-radius:.75rem;background:var(--panel);box-shadow:0 1px 2px rgba(0,0,0,.3)} "
                ".badge{font-size:.75rem;padding:.1rem .5rem;border-radius:.25rem;background:#1e293b} "
                ".space-y-6>*+*{margin-top:1.5rem} "
                ".summary-section{margin-bottom:1.5rem} "
                ".hidden{display:none} "
                ".detail-header{display:flex;justify-content:space-between;align-items:flex-end;gap:1.5rem;flex-wrap:wrap} "
                ".detail-control{display:flex;flex-direction:column;gap:.4rem;min-width:220px} "
                ".detail-control label{font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;color:#94a3b8} "
                ".detail-control select{background:#1e293b;border:1px solid rgba(148,163,184,.35);border-radius:.5rem;padding:.5rem .75rem;color:#e2e8f0} "
                ".detail-charts{margin-top:1.5rem} "
                ".trade-table{overflow-x:auto} "
                ".trade-table table{width:100%;border-collapse:collapse;font-size:.8rem} "
                ".trade-table thead{background:rgba(148,163,184,.1);text-transform:uppercase;letter-spacing:.06em;color:#94a3b8} "
                ".trade-table th,.trade-table td{padding:.4rem .55rem;text-align:left;border-bottom:1px solid rgba(148,163,184,.2)} "
                "table{width:100%;font-size:.875rem} thead{background:#334155;color:#cbd5e1} "
                "th,td{padding:.5rem .75rem;text-align:left} code{font-family:ui-monospace,Menlo,Monaco,Consolas,monospace}"
            )
            head_assets = f"<style>{css_inline}</style>"
        else:
            # Tailwind is convenient for rich reports, but requires internet. Users who want a fully
            # offline report can pass `--inline-css`.
            head_assets = (
                '<script src="https://cdn.tailwindcss.com/3.4.17" '
                'integrity="sha384-igm5BeiBt36UU4gqwWS7imYmelpTsZlQ45FZf+XBn9MuJbn4nQr7yx1yFydocC/K" '
                'referrerpolicy="no-referrer"></script>\n'
                "  <script>\n"
                "    tailwind.config = { darkMode: 'class' };\n"
                "  </script>"
            )

        html = f"""
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Backtest Report</title>
  {head_assets}
  <style>
    body {{ background-color: rgb(2 6 23); }}
    .hidden {{ display: none; }}
    .detail-header {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 1.5rem; flex-wrap: wrap; }}
    .detail-control {{ display: flex; flex-direction: column; gap: 0.4rem; min-width: 220px; }}
    .detail-control label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: rgb(148 163 184); }}
    .detail-control select {{ background: rgb(30 41 59); border: 1px solid rgba(148, 163, 184, 0.35); border-radius: 0.5rem; padding: 0.5rem 0.75rem; color: rgb(226 232 240); }}
    .detail-charts {{ margin-top: 1.5rem; }}
    .trade-table {{ overflow-x: auto; }}
    .trade-table table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
    .trade-table thead {{ background: rgba(148, 163, 184, 0.1); text-transform: uppercase; letter-spacing: 0.06em; color: rgb(148, 163, 184); }}
    .trade-table th, .trade-table td {{ padding: 0.4rem 0.55rem; text-align: left; border-bottom: 1px solid rgba(148, 163, 184, 0.2); }}
  </style>
  </head>
<body class="bg-slate-950">
  <div class="max-w-6xl mx-auto p-6">
    <div class="space-y-6">
      <div class="flex items-center justify-between">
        <h1 class="text-2xl font-bold text-slate-100">Backtest Report</h1>
        <button onclick="document.documentElement.classList.toggle('dark');" class="px-3 py-1 rounded bg-slate-800 text-slate-200 hover:bg-slate-700">Toggle Theme</button>
      </div>
      {summary_html}
      {scatter_section}
      {detail_section}
      <div class="grid gap-4">
        {"".join(cards)}
      </div>
    </div>
  </div>
</body>
{plotly_tag}
<script>
  (function() {{
    const scatterData = {chart_json};
    if (scatterData.length) {{
      const trace = {{
        x: scatterData.map(r => r.metric_value),
        y: scatterData.map(r => r.sharpe),
        text: scatterData.map(r => `${{escapeHtml(r.collection)}}/${{escapeHtml(r.symbol)}} • ${{escapeHtml(r.strategy)}} (${{escapeHtml(r.timeframe)}})`),
        mode: 'markers',
        hovertemplate: '%{{text}}<br>Metric: %{{x:.4f}}<br>Sharpe: %{{y:.4f}}<extra></extra>',
        marker: {{
          size: 10,
          color: scatterData.map(r => r.metric_value),
          colorscale: 'Turbo',
          showscale: true,
          colorbar: {{ title: 'Metric Value' }}
        }}
      }};
      const layout = {{
        margin: {{ l: 60, r: 20, t: 40, b: 60 }},
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {{ title: 'Metric Value' }},
        yaxis: {{ title: 'Sharpe' }}
      }};
      Plotly.newPlot('metric-scatter', [trace], layout, {{ displayModeBar: false, responsive: true }});
    }}

    const detailData = {detail_json};
    const detailSelector = document.getElementById('detail-selector');
    const detailEmpty = document.getElementById('detail-empty');
    const detailCharts = document.getElementById('detail-charts');
    const tradeTableContainer = document.getElementById('trade-table-container');

    function escapeHtml(value) {{
      return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }}

    function detailLabel(entry) {{
      if (!entry) return '—';
      const parts = [
        `${{entry.collection || 'unknown'}}/${{entry.symbol || ''}}`,
        entry.strategy ? `• ${{entry.strategy}}` : null,
        entry.timeframe ? `@ ${{entry.timeframe}}` : null,
      ].filter(Boolean);
      const suffix = Number.isFinite(entry.metric_value)
        ? '(' + (entry.metric || 'metric') + ' ' + entry.metric_value.toFixed(4) + ')'
        : '';
      return [parts.join(' '), suffix].filter(Boolean).join(' ').trim();
    }}

    function renderTradeTable(entry) {{
      if (!tradeTableContainer) return;
      tradeTableContainer.innerHTML = '';
      const trades = Array.isArray(entry?.trades_log) ? entry.trades_log : [];
      if (!trades.length) {{
        tradeTableContainer.innerHTML = '<p class="text-slate-400 text-sm">No trades captured for this result.</p>';
        return;
      }}
      const headers = Object.keys(trades[0] || {{}});
      if (!headers.length) {{
        tradeTableContainer.innerHTML = '<p class="text-slate-400 text-sm">No trades captured for this result.</p>';
        return;
      }}
      const headCells = headers.map(h => `<th>${{escapeHtml(h)}}</th>`).join('');
      const limit = Math.min(trades.length, 50);
      const bodyRows = trades.slice(0, 50).map(row => {{
        const cells = headers.map(h => `<td>${{escapeHtml(row[h] ?? '')}}</td>`).join('');
        return `<tr>${{cells}}</tr>`;
      }}).join('');
      tradeTableContainer.innerHTML = `
        <div class="trade-table">
          <h3 class="text-sm text-slate-300 mb-2">Trade Log (First ${{limit}})</h3>
          <table>
            <thead><tr>${{headCells}}</tr></thead>
            <tbody>${{bodyRows}}</tbody>
          </table>
        </div>
      `;
    }}

    function renderDetail(entry) {{
      if (!detailEmpty || !detailCharts) return;
      const hasEquity = Array.isArray(entry?.equity_curve) && entry.equity_curve.length;
      const hasDrawdown = Array.isArray(entry?.drawdown_curve) && entry.drawdown_curve.length;
      if (!hasEquity && !hasDrawdown) {{
        detailEmpty.classList.remove('hidden');
        detailCharts.classList.add('hidden');
        Plotly.purge('equity-chart');
        Plotly.purge('drawdown-chart');
      }} else {{
        detailEmpty.classList.add('hidden');
        detailCharts.classList.remove('hidden');
        if (hasEquity) {{
          Plotly.react('equity-chart', [{{
            x: entry.equity_curve.map(p => p.ts),
            y: entry.equity_curve.map(p => p.value),
            mode: 'lines',
            line: {{ color: '#38bdf8' }},
            hovertemplate: '%{{x}}<br>Equity: %{{y:.4f}}<extra></extra>'
          }}], {{
            margin: {{ l: 40, r: 10, t: 20, b: 40 }},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            xaxis: {{ title: 'Time' }},
            yaxis: {{ title: 'Equity' }}
          }}, {{ displayModeBar: false, responsive: true }});
        }} else {{
          Plotly.purge('equity-chart');
          document.getElementById('equity-chart').innerHTML = '<div class="text-slate-500 text-sm">No equity data.</div>';
        }}
        if (hasDrawdown) {{
          Plotly.react('drawdown-chart', [{{
            x: entry.drawdown_curve.map(p => p.ts),
            y: entry.drawdown_curve.map(p => p.value),
            mode: 'lines',
            line: {{ color: '#f87171' }},
            hovertemplate: '%{{x}}<br>Drawdown: %{{y:.4f}}<extra></extra>'
          }}], {{
            margin: {{ l: 40, r: 10, t: 20, b: 40 }},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            xaxis: {{ title: 'Time' }},
            yaxis: {{ title: 'Drawdown' }}
          }}, {{ displayModeBar: false, responsive: true }});
        }} else {{
          Plotly.purge('drawdown-chart');
          document.getElementById('drawdown-chart').innerHTML = '<div class="text-slate-500 text-sm">No drawdown data.</div>';
        }}
      }}
      renderTradeTable(entry);
    }}

    function populateDetailSelector() {{
      if (!detailSelector) return;
      if (!Array.isArray(detailData) || !detailData.length) {{
        detailSelector.innerHTML = '<option>No results available</option>';
        detailSelector.disabled = true;
        renderDetail(null);
        return;
      }}
      const options = detailData.map((entry, idx) => {{
        const label = detailLabel(entry) || `Result ${idx + 1}`;
        const selected = idx === 0 ? ' selected' : '';
        return `<option value="${{escapeHtml(entry.id)}}"${{selected}}>${{escapeHtml(label)}}</option>`;
      }}).join('');
      detailSelector.innerHTML = options;
      detailSelector.disabled = false;
      const initial = detailData[0];
      renderDetail(initial);
      detailSelector.addEventListener('change', () => {{
        const chosen = detailData.find(entry => entry.id === detailSelector.value);
        renderDetail(chosen || null);
      }});
    }}

    populateDetailSelector();

    if (typeof window !== 'undefined') {{
      window.__dashboard_latest_run = scatterData[0] ? scatterData[0].collection : null;
      window.__dashboard_trades = detailData?.[0]?.trades_log || [];
    }}
  }})();
</script>
</html>
        """

        if self.inline_css:
            html = html.replace("max-w-6xl mx-auto p-6", "container")
            html = html.replace(
                "px-3 py-1 rounded bg-slate-800 text-slate-200 hover:bg-slate-700", "btn"
            )
            html = html.replace("p-4 rounded-lg bg-slate-800 text-slate-100 shadow", "card")
            html = html.replace("text-xs px-2 py-1 rounded bg-slate-700", "badge")

        path = self.out_dir / "report.html"
        path.write_text(html)
