from __future__ import annotations

import csv
from pathlib import Path

from ..backtest.runner import BestResult
from .utils import is_positive


class CSVExporter:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def export(self, results: list[BestResult]):
        best_by_symbol: dict[tuple[str, str], BestResult] = {}
        for r in results:
            key = (r.collection, r.symbol)
            prev = best_by_symbol.get(key)
            if prev is None or float(r.metric_value) > float(prev.metric_value):
                best_by_symbol[key] = r

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
                    "omega",
                    "tail_ratio",
                    "profit",
                    "pain_index",
                    "trades",
                    "max_drawdown",
                ]
            )
            for r in sorted(best_by_symbol.values(), key=lambda x: (x.collection, x.symbol)):
                if not is_positive(r.metric_value):
                    continue
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
                        f"{r.stats.get('omega', float('nan')):.6f}",
                        f"{r.stats.get('tail_ratio', float('nan')):.6f}",
                        f"{r.stats.get('profit', float('nan')):.6f}",
                        f"{r.stats.get('pain_index', float('nan')):.6f}",
                        r.stats.get("trades", 0),
                        f"{r.stats.get('max_drawdown', float('nan')):.6f}",
                    ]
                )
