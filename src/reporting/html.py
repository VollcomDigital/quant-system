from __future__ import annotations

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

        def card_for_row(row: dict[str, Any]) -> str:
            stats = row.get("stats", {}) or {}
            return f"""
            <div class='p-4 rounded-lg bg-slate-800 text-slate-100 shadow'>
                <div class='flex justify-between items-baseline'>
                    <h3 class='text-lg font-semibold'>{row.get("collection", "")} / {row.get("symbol", "")}</h3>
                    <span class='text-xs px-2 py-1 rounded bg-slate-700'>{row.get("timeframe", "")}</span>
                </div>
                <div class='mt-2 text-sm'>
                    <div><span class='font-semibold'>Strategy:</span> {row.get("strategy", "")}</div>
                    <div><span class='font-semibold'>Metric:</span> {row.get("metric", "")} = {float(row.get("metric_value", float("nan"))):.6f}</div>
                    <div><span class='font-semibold'>Params:</span> <code class='text-xs'>{row.get("params", {})}</code></div>
                </div>
                <div class='mt-3 grid grid-cols-2 gap-2 text-sm'>
                    <div>Sharpe: {float(stats.get("sharpe", float("nan"))):.4f}</div>
                    <div>Sortino: {float(stats.get("sortino", float("nan"))):.4f}</div>
                    <div>Profit: {float(stats.get("profit", float("nan"))):.4f}</div>
                    <div>Trades: {int(stats.get("trades", 0))}</div>
                    <div class='col-span-2'>Max DD: {float(stats.get("max_drawdown", float("nan"))):.4f}</div>
                </div>
            </div>
            """

        def table_for_topn(rows: list[dict[str, Any]]) -> str:
            rows = sorted(rows, key=lambda x: x["metric_value"], reverse=True)[: self.top_n]
            body = "\n".join(
                f"<tr><td>{r['timeframe']}</td><td>{r['strategy']}</td><td>{r['metric']}</td><td>{r['metric_value']:.6f}</td>"
                f"<td><code class='text-xs'>{r['params']}</code></td>"
                f"<td>{r['stats'].get('sharpe', float('nan')):.4f}</td>"
                f"<td>{r['stats'].get('sortino', float('nan')):.4f}</td>"
                f"<td>{r['stats'].get('profit', float('nan')):.4f}</td>"
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
                    <th class='px-3 py-2'>Profit</th>
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

        html = f"""
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Backtest Report</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {{ darkMode: 'class' }};
  </script>
  <style>
    body {{ background-color: rgb(2 6 23); }}
  </style>
  </head>
<body class="bg-slate-950">
  <div class="max-w-6xl mx-auto p-6">
    <div class="flex items-center justify-between mb-6">
      <h1 class="text-2xl font-bold text-slate-100">Backtest Report</h1>
      <button onclick="document.documentElement.classList.toggle('dark');" class="px-3 py-1 rounded bg-slate-800 text-slate-200 hover:bg-slate-700">Toggle Theme</button>
    </div>
    <div class="grid gap-4">
      {"".join(cards)}
    </div>
  </div>
</body>
</html>
        """

        if self.inline_css:
            css_inline = (
                ":root{--bg:#020617;--panel:#0f172a;--text:#e2e8f0;--muted:#94a3b8} "
                "body{background:var(--bg);color:var(--text);} "
                ".container{max-width:72rem;margin:0 auto;padding:1.5rem} "
                ".btn{padding:.25rem .75rem;border-radius:.375rem;background:#1e293b;color:#e2e8f0} "
                ".grid{display:grid;gap:1rem} "
                ".card{padding:1rem;border-radius:.75rem;background:var(--panel);box-shadow:0 1px 2px rgba(0,0,0,.3)} "
                ".badge{font-size:.75rem;padding:.1rem .5rem;border-radius:.25rem;background:#1e293b} "
                "table{width:100%;font-size:.875rem} thead{background:#334155;color:#cbd5e1} "
                "th,td{padding:.5rem .75rem;text-align:left} code{font-family:ui-monospace,Menlo,Monaco,Consolas,monospace}"
            )
            html = html.replace(
                '<script src="https://cdn.tailwindcss.com"></script>',
                f"<style>{css_inline}</style>",
            )
            html = html.replace("tailwind.config = { darkMode: 'class' };", "")
            html = html.replace("max-w-6xl mx-auto p-6", "container")
            html = html.replace(
                "px-3 py-1 rounded bg-slate-800 text-slate-200 hover:bg-slate-700", "btn"
            )
            html = html.replace("p-4 rounded-lg bg-slate-800 text-slate-100 shadow", "card")
            html = html.replace("text-xs px-2 py-1 rounded bg-slate-700", "badge")

        path = self.out_dir / "report.html"
        path.write_text(html)
