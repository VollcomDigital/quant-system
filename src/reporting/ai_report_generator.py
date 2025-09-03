"""AI Report Generator for Investment Recommendations."""

from __future__ import annotations

from pathlib import Path

from src.ai.models import PortfolioRecommendation


class AIReportGenerator:
    """Generates HTML reports for AI investment recommendations."""

    def __init__(self):
        # Base dir is unified with other exports
        self.base_dir = Path("exports/ai_reco")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def generate_html_report(
        self,
        recommendation: PortfolioRecommendation,
        portfolio_name: str,
        year: str,
        quarter: str,
        interval: str,
    ) -> str:
        """Generate HTML report under exports/ai_reco/<Year>/<Quarter>/ with unified name."""

        quarterly_dir = self.base_dir / str(year) / str(quarter)
        quarterly_dir.mkdir(parents=True, exist_ok=True)

        sanitized = (
            portfolio_name.replace(" ", "_").replace("/", "_").strip("_")
            or "All_Collections"
        )
        safe_interval = (interval or "multi").replace("/", "-")
        filename = f"{sanitized}_Collection_{year}_{quarter}_{safe_interval}.html"
        output_path = quarterly_dir / filename

        html_content = self._create_html_content(recommendation, portfolio_name)

        output_path.write_text(html_content, encoding="utf-8")
        return str(output_path)

    def _create_html_content(
        self, recommendation: PortfolioRecommendation, portfolio_name: str
    ) -> str:
        """Create HTML content for AI recommendations."""

        asset_rows = ""
        for asset in recommendation.asset_recommendations:
            badge_class = (
                "bg-emerald-500/10 text-emerald-300"
                if asset.recommendation_type == "BUY"
                else "bg-amber-500/10 text-amber-300"
                if asset.recommendation_type == "HOLD"
                else "bg-rose-500/10 text-rose-300"
            )

            asset_rows += f"""
            <tr>
              <td class=\"px-4 py-3\">{asset.symbol}</td>
              <td class=\"px-4 py-3\">{asset.strategy}</td>
              <td class=\"px-4 py-3\">{asset.timeframe}</td>
              <td class=\"px-4 py-3\">{asset.allocation_percentage:.1f}%</td>
              <td class=\"px-4 py-3\">{asset.risk_level}</td>
              <td class=\"px-4 py-3\">{asset.confidence_score:.3f}</td>
              <td class=\"px-4 py-3\">{asset.sortino_ratio:.3f}</td>
              <td class=\"px-4 py-3\">{asset.sharpe_ratio:.3f}</td>
              <td class=\"px-4 py-3\">{asset.total_return:.2f}%</td>
              <td class=\"px-4 py-3\">{asset.risk_per_trade:.1f}%</td>
              <td class=\"px-4 py-3\">{asset.position_size:.1f}%</td>
              <td class=\"px-4 py-3\">{asset.stop_loss:.0f}</td>
              <td class=\"px-4 py-3\">{asset.take_profit:.0f}</td>
              <td class=\"px-4 py-3\"><span class=\"inline-flex rounded-md px-2 py-1 text-xs font-medium {badge_class}\">{asset.recommendation_type}</span></td>
            </tr>"""

        html_template = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>AI Investment Recommendations: {portfolio_name}</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body {{font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 0; padding: 20px; background: #0b0f19;}}
  </style>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body class="bg-slate-950 bg-gradient-to-br from-indigo-900/20 via-slate-900 to-cyan-900/20 text-slate-100">
  <div class="max-w-7xl mx-auto p-6">
    <header class="rounded-xl border border-white/10 bg-white/5 backdrop-blur p-5 shadow mb-4">
      <h1 class="text-2xl font-semibold tracking-tight">AI Investment Recommendations</h1>
      <p class="text-slate-300 mt-1">Portfolio: {portfolio_name} • Risk Profile: {recommendation.risk_profile.title()}</p>
      <div class="mt-3">
        <a href="#" id="downloadCsvLink" class="inline-flex items-center gap-2 rounded-lg border border-cyan-400/40 bg-cyan-500/10 px-3 py-1 text-sm text-cyan-300 hover:bg-cyan-500/20">Download CSV</a>
      </div>
    </header>

    <section class="rounded-xl border border-white/10 bg-white/5 p-5 mb-4">
      <h2 class="text-sm font-semibold text-slate-300 mb-3">Summary</h2>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <div class="rounded-xl border border-white/10 bg-white/5 p-4">
          <div class="text-xs uppercase text-slate-300">Total Assets</div>
          <div class="text-xl font-semibold">{recommendation.total_assets}</div>
        </div>
        <div class="rounded-xl border border-white/10 bg-white/5 p-4">
          <div class="text-xs uppercase text-slate-300">Expected Return</div>
          <div class="text-xl font-semibold">{recommendation.expected_return:.2f}%</div>
        </div>
        <div class="rounded-xl border border-white/10 bg-white/5 p-4">
          <div class="text-xs uppercase text-slate-300">Confidence</div>
          <div class="text-xl font-semibold">{recommendation.confidence_score:.3f}</div>
        </div>
      </div>
      <p class="text-slate-300 mt-4">{recommendation.reasoning}</p>
    </section>

    <section class="rounded-xl border border-white/10 bg-white/5 p-5">
      <h2 class="text-sm font-semibold text-slate-300 mb-3">Asset Recommendations</h2>
      <div class="overflow-hidden rounded-xl border border-white/10 bg-white/5">
        <table class="min-w-full divide-y divide-white/10 text-sm">
          <thead class="bg-white/5">
            <tr class="text-slate-300 uppercase tracking-wider text-xs">
              <th class="px-4 py-3 text-left">Symbol</th>
              <th class="px-4 py-3 text-left">Strategy</th>
              <th class="px-4 py-3 text-left">Timeframe</th>
              <th class="px-4 py-3 text-left">Allocation</th>
              <th class="px-4 py-3 text-left">Risk Level</th>
              <th class="px-4 py-3 text-left">Confidence</th>
              <th class="px-4 py-3 text-left">Sortino</th>
              <th class="px-4 py-3 text-left">Sharpe</th>
              <th class="px-4 py-3 text-left">Return</th>
              <th class="px-4 py-3 text-left">Risk/Trade</th>
              <th class="px-4 py-3 text-left">Pos. Size</th>
              <th class="px-4 py-3 text-left">SL</th>
              <th class="px-4 py-3 text-left">TP</th>
              <th class="px-4 py-3 text-left">Action</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-white/5">
            {asset_rows}
          </tbody>
        </table>
      </div>
    </section>
  </div>
</body>
</html>"""

        # Set CSV link dynamically (same directory, same base name with .csv)
        try:
            # Simple replacement: add a small inline script to patch href at runtime
            html_template += "\n<script>\n(function(){\n  try{\n    var link=document.getElementById('downloadCsvLink');\n    if(link){\n      var here=window.location.href;\n      var csv=here.replace(/\\.html(?=#|$)/,'\\.csv');\n      link.href=csv;\n    }\n  }catch(e){}\n})();\n</script>\n"
        except Exception:
            pass

        return html_template
