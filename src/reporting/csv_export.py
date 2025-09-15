from __future__ import annotations

import csv
from pathlib import Path

from ..backtest.runner import BestResult


class CSVExporter:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def export(self, results: list[BestResult]):
        path = self.out_dir / "summary.csv"
        with open(path, "w", newline="") as f:
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
            for r in results:
                w.writerow(
                    [
                        r.collection,
                        r.symbol,
                        r.timeframe,
                        r.strategy,
                        r.metric_name,
                        f"{r.metric_value:.6f}",
                        r.params,
                        f"{r.stats.get('sharpe', float('nan')):.6f}",
                        f"{r.stats.get('sortino', float('nan')):.6f}",
                        f"{r.stats.get('profit', float('nan')):.6f}",
                        r.stats.get("trades", 0),
                        f"{r.stats.get('max_drawdown', float('nan')):.6f}",
                    ]
                )
