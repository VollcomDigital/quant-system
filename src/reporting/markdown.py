from __future__ import annotations

from pathlib import Path

from ..backtest.runner import BestResult


class MarkdownReporter:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def export(self, results: list[BestResult]):
        # Group by collection -> symbol
        by_key = {}
        for r in results:
            by_key.setdefault((r.collection, r.symbol), []).append(r)

        lines: list[str] = []
        lines.append("# Backtest Report\n")

        for (collection, symbol), rows in sorted(by_key.items()):
            lines.append(f"## {collection} / {symbol}\n")
            # Sort rows by metric_value desc
            rows = sorted(rows, key=lambda x: x.metric_value, reverse=True)
            best = rows[0]
            lines.append("- Best Combination:")
            lines.append(f"  - Timeframe: {best.timeframe}")
            lines.append(f"  - Strategy: {best.strategy}")
            lines.append(f"  - Metric: {best.metric_name} = {best.metric_value:.6f}")
            lines.append(f"  - Params: {best.params}")
            lines.append("- Key Metrics:")
            lines.append(f"  - Sharpe: {best.stats.get('sharpe', float('nan')):.6f}")
            lines.append(f"  - Sortino: {best.stats.get('sortino', float('nan')):.6f}")
            lines.append(f"  - Profit: {best.stats.get('profit', float('nan')):.6f}")
            lines.append(f"  - Trades: {best.stats.get('trades', 0)}")
            lines.append(f"  - Max Drawdown: {best.stats.get('max_drawdown', float('nan')):.6f}")
            lines.append("")

        path = self.out_dir / "report.md"
        path.write_text("\n".join(lines))
