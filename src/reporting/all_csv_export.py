from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..backtest.results_cache import ResultsCache
from ..backtest.runner import BestResult


class AllCSVExporter:
    def __init__(self, out_dir: Path, results_cache: ResultsCache, run_id: str, top_n: int = 3):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache = results_cache
        self.run_id = run_id
        self.top_n = top_n

    def export(self, best_results: list[BestResult]):
        all_rows = self.cache.list_by_run(self.run_id)
        # All results CSV
        path_all = self.out_dir / "all_results.csv"
        with open(path_all, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "collection",
                    "symbol",
                    "timeframe",
                    "strategy",
                    "metric",
                    "metric_value",
                    "params",
                    "sharpe",
                    "sortino",
                    "profit",
                    "trades",
                    "max_drawdown",
                ]
            )
            for r in all_rows:
                stats = r.get("stats", {})
                w.writerow(
                    [
                        r["collection"],
                        r["symbol"],
                        r["timeframe"],
                        r["strategy"],
                        r["metric"],
                        f"{r['metric_value']:.6f}",
                        r["params"],
                        f"{stats.get('sharpe', float('nan')):.6f}",
                        f"{stats.get('sortino', float('nan')):.6f}",
                        f"{stats.get('profit', float('nan')):.6f}",
                        stats.get("trades", 0),
                        f"{stats.get('max_drawdown', float('nan')):.6f}",
                    ]
                )

        # Top-N per (collection, symbol)
        path_topn = self.out_dir / f"top{self.top_n}.csv"
        # group
        grouped: dict[tuple, list[dict[str, Any]]] = {}
        for r in all_rows:
            key = (r["collection"], r["symbol"])
            grouped.setdefault(key, []).append(r)
        with open(path_topn, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "collection",
                    "symbol",
                    "timeframe",
                    "strategy",
                    "metric",
                    "metric_value",
                    "params",
                    "sharpe",
                    "sortino",
                    "profit",
                    "trades",
                    "max_drawdown",
                ]
            )
            for _key, rows in grouped.items():
                rows = sorted(rows, key=lambda x: x["metric_value"], reverse=True)[: self.top_n]
                for r in rows:
                    stats = r.get("stats", {})
                    w.writerow(
                        [
                            r["collection"],
                            r["symbol"],
                            r["timeframe"],
                            r["strategy"],
                            r["metric"],
                            f"{r['metric_value']:.6f}",
                            r["params"],
                            f"{stats.get('sharpe', float('nan')):.6f}",
                            f"{stats.get('sortino', float('nan')):.6f}",
                            f"{stats.get('profit', float('nan')):.6f}",
                            stats.get("trades", 0),
                            f"{stats.get('max_drawdown', float('nan')):.6f}",
                        ]
                    )
