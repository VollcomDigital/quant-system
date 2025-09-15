from __future__ import annotations

from pathlib import Path

from ..backtest.runner import BestResult


class TradingViewExporter:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def export(self, results: list[BestResult]):
        # Keep only the best strategy per (collection, symbol, timeframe)
        best_per_key: dict[tuple[str, str, str], BestResult] = {}
        for r in results:
            key = (r.collection, r.symbol, r.timeframe)
            prev = best_per_key.get(key)
            if prev is None or float(r.metric_value) > float(prev.metric_value):
                best_per_key[key] = r

        lines: list[str] = []
        lines.append("# TradingView Export\n")
        lines.append("Copy/paste alert messages for the best strategy per symbol/timeframe.\n")

        # Sort for stable output
        for key in sorted(best_per_key.keys()):
            r = best_per_key[key]
            sharpe = r.stats.get("sharpe") if isinstance(r.stats, dict) else None
            sortino = r.stats.get("sortino") if isinstance(r.stats, dict) else None
            calmar = r.stats.get("calmar") if isinstance(r.stats, dict) else None

            def fmt(x):
                try:
                    return f"{float(x):.3f}"
                except Exception:
                    return "N/A"

            msg = (
                f"🚨 QUANT SIGNAL: {r.symbol} 📊\n"
                f"Strategy: {r.strategy}\n"
                f"Timeframe: {r.timeframe}\n"
                f"📈 Sharpe: {fmt(sharpe)}\n"
                f"📊 Sortino: {fmt(sortino)}\n"
                f"⚖️ Calmar: {fmt(calmar)}\n\n"
                f"Price: {{close}}\n"
                f"Time: {{timenow}}\n"
                f"Action: {{strategy.order.action}}\n"
                f"Qty: {{strategy.order.contracts}}\n\n"
                f"#QuantTrading #{r.symbol} #{r.strategy}"
            )
            lines.append(msg)
            lines.append("")

        path = self.out_dir / "tradingview.md"
        path.write_text("\n".join(lines))
