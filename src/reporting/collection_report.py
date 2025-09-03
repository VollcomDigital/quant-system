"""Clean Portfolio Report Generator (DB-sourced, Tailwind-ready)

This reporter reads only from the database (unified_models lightweight schema) to
render a per-asset HTML report. It prefers detailed stats saved in
unified_models.BacktestResult.engine_ctx and overlays values from
unified_models.BacktestResult.metrics when needed. No JSON files are used.

Styling: Uses Tailwind. In production, set TAILWIND_CSS_HREF to a built CSS
file (e.g., /assets/tailwind.min.css). If unset and no local CSS is found under
exports/reports/assets/tailwind.min.css, falls back to the CDN for dev only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.reporting.report_organizer import ReportOrganizer


class DetailedPortfolioReporter:
    """Generates detailed visual reports using only DB data (unified_models)."""

    def __init__(self):
        self.report_organizer = ReportOrganizer()

    def generate_comprehensive_report(
        self,
        portfolio_config: Dict[str, Any],
        start_date: str,
        end_date: str,
        strategies: list[str],
        timeframes: list[str] | None = None,
        filename_interval: str | None = None,
    ) -> str:
        if timeframes is None:
            timeframes = ["1d"]

        symbols = portfolio_config.get("symbols") or []
        assets_data: Dict[str, Dict[str, Any]] = {}
        for symbol in symbols:
            assets_data[symbol] = self._get_asset_data(
                symbol, preferred_timeframes=timeframes or ["1d"]
            )

        html = self._create_html_report(
            portfolio_config,
            assets_data,
            start_date,
            end_date,
            strategies=strategies,
            timeframes=timeframes,
        )
        # Choose interval token for filename:
        # - If explicit filename_interval is provided, use it (e.g., "multi" for --interval all)
        # - Else prefer '1d' if included, otherwise first of timeframes
        interval = "1d"
        try:
            if filename_interval:
                interval = filename_interval
            elif timeframes:
                interval = "1d" if "1d" in timeframes else timeframes[0]
        except Exception:
            interval = "1d"
        return self._save_report(
            html, portfolio_config.get("name") or "portfolio", interval
        )

    def _get_asset_data(
        self, symbol: str, preferred_timeframes: list[str] | None = None
    ) -> Dict[str, Any]:
        try:
            from src.database import unified_models as um
        except Exception:
            um = None
        # Primary DB models (fallback for metrics when unified tables are empty)
        try:
            from src.database import models as dbm
            from src.database.db_connection import (
                get_db_session as get_primary_session,  # type: ignore[import-not-found]
            )
        except Exception:
            dbm = None
            get_primary_session = None  # type: ignore[assignment]

        sess = um.Session() if um else None
        try:
            # Prefer best strategy for requested timeframes (e.g., ['1d'])
            u_bs = None
            if um and sess:
                try:
                    q = sess.query(um.BestStrategy).filter(
                        um.BestStrategy.symbol == symbol
                    )
                    if preferred_timeframes:
                        q_pref = (
                            q.filter(
                                um.BestStrategy.timeframe.in_(preferred_timeframes)
                            )
                            .order_by(um.BestStrategy.updated_at.desc())
                            .limit(1)
                        )
                        u_bs = q_pref.one_or_none()
                    # Fallback to any timeframe if none found for preference
                    if not u_bs:
                        u_bs = (
                            q.order_by(um.BestStrategy.updated_at.desc())
                            .limit(1)
                            .one_or_none()
                        )
                except Exception:
                    u_bs = None

            # Secondary fallback to primary models BestStrategy (backtests schema)
            b_bs = None
            if not u_bs and dbm is not None and get_primary_session is not None:
                try:
                    s2 = get_primary_session()
                except Exception:
                    s2 = None
                if s2 is not None:
                    try:
                        q2 = s2.query(dbm.BestStrategy).filter(
                            dbm.BestStrategy.symbol == symbol
                        )
                        if preferred_timeframes:
                            q2 = q2.filter(
                                dbm.BestStrategy.timeframe.in_(preferred_timeframes)
                            )
                        b_bs = (
                            q2.order_by(dbm.BestStrategy.updated_at.desc())
                            .limit(1)
                            .one_or_none()
                        )
                    except Exception:
                        b_bs = None
                    finally:
                        try:
                            s2.close()
                        except Exception:
                            pass
            if not u_bs and not b_bs:
                return {
                    "best_strategy": "N/A",
                    "best_timeframe": "1d",
                    "data": {"overview": self._empty_overview(), "orders": []},
                }

            timeframe = getattr(u_bs, "timeframe", None) or getattr(
                b_bs, "timeframe", "1d"
            )
            overview = self._empty_overview()

            def _f(v):
                try:
                    return float(v) if v is not None else 0.0
                except Exception:
                    return 0.0

            # Pull from unified BestStrategy or fallback BestStrategy
            src_bs = u_bs if u_bs is not None else b_bs
            overview["PSR"] = _f(getattr(src_bs, "sortino_ratio", 0))
            overview["sharpe_ratio"] = _f(getattr(src_bs, "sharpe_ratio", 0))
            overview["net_profit"] = _f(getattr(src_bs, "total_return", 0))
            overview["max_drawdown"] = abs(_f(getattr(src_bs, "max_drawdown", 0)))
            # optional calmar
            try:
                overview["calmar_ratio"] = _f(getattr(u_bs, "calmar_ratio", 0))
            except Exception:
                pass

            stats_full: Dict[str, Any] = {}
            period_start_str: str | None = None
            period_end_str: str | None = None

            # Find corresponding BacktestResult for richer stats
            br = None
            try:
                if getattr(u_bs, "backtest_result_id", None):
                    br = (
                        sess.query(um.BacktestResult)
                        .filter(um.BacktestResult.result_id == u_bs.backtest_result_id)
                        .one_or_none()
                    )
                # If BestStrategy doesn't carry result_id, align by the declared best strategy
                if not br:
                    br = (
                        sess.query(um.BacktestResult)
                        .filter(um.BacktestResult.symbol == symbol)
                        .filter(um.BacktestResult.interval == timeframe)
                        .filter(
                            um.BacktestResult.strategy == getattr(u_bs, "strategy", "")
                        )
                        .order_by(um.BacktestResult.end_at_utc.desc().nullslast())
                        .first()
                    )
                # Last fallback: latest any strategy (kept for resilience)
                if not br:
                    br = (
                        sess.query(um.BacktestResult)
                        .filter(um.BacktestResult.symbol == symbol)
                        .filter(um.BacktestResult.interval == timeframe)
                        .order_by(um.BacktestResult.end_at_utc.desc().nullslast())
                        .first()
                    )
            except Exception:
                br = None

            # Prefer engine_ctx for canonical backtesting library stats
            if br and isinstance(br.engine_ctx, dict):
                stats_full.update(br.engine_ctx)
                # Try to derive period from engine context when DB timestamps are missing
                try:
                    if not period_start_str and isinstance(
                        stats_full.get("Start"), str
                    ):
                        period_start_str = stats_full.get("Start")[:10]
                    if not period_end_str and isinstance(stats_full.get("End"), str):
                        period_end_str = stats_full.get("End")[:10]
                except Exception:
                    pass

            # Overlay metrics if engine_ctx lacks fields
            if br and isinstance(br.metrics, dict):
                m = br.metrics or {}
                stats_full.setdefault(
                    "Sortino Ratio", m.get("sortino_ratio") or m.get("Sortino_Ratio")
                )
                stats_full.setdefault(
                    "Sharpe Ratio", m.get("sharpe_ratio") or m.get("Sharpe_Ratio")
                )
                stats_full.setdefault(
                    "Return [%]", m.get("total_return") or m.get("Total_Return")
                )
                stats_full.setdefault(
                    "Max. Drawdown [%]", m.get("max_drawdown") or m.get("Max_Drawdown")
                )
                stats_full.setdefault(
                    "Win Rate [%]", m.get("win_rate") or m.get("Win_Rate")
                )

            # If unified BacktestResult missing, try fallback primary results to populate overview keys
            if not br and b_bs is not None:
                try:
                    overview["PSR"] = (
                        _f(getattr(b_bs, "sortino_ratio", 0)) or overview["PSR"]
                    )
                    overview["sharpe_ratio"] = (
                        _f(getattr(b_bs, "sharpe_ratio", 0)) or overview["sharpe_ratio"]
                    )
                    overview["net_profit"] = (
                        _f(getattr(b_bs, "total_return", 0)) or overview["net_profit"]
                    )
                    md = _f(getattr(b_bs, "max_drawdown", 0))
                    if md:
                        overview["max_drawdown"] = abs(md)
                except Exception:
                    pass

            # Capture period from DB result for display and derived annualized stats
            try:
                if br:
                    sd = getattr(br, "start_at_utc", None)
                    ed = getattr(br, "end_at_utc", None)
                    if sd and not period_start_str:
                        try:
                            period_start_str = sd.date().isoformat()
                        except Exception:
                            pass
                    if ed and not period_end_str:
                        try:
                            period_end_str = ed.date().isoformat()
                        except Exception:
                            pass
            except Exception:
                pass

            # Compute Return (Ann.) [%] if possible
            try:
                if (
                    br
                    and ("Return (Ann.) [%]" not in stats_full)
                    and stats_full.get("Return [%]") is not None
                ):
                    sd = getattr(br, "start_at_utc", None)
                    ed = getattr(br, "end_at_utc", None)
                    if sd and ed:
                        days = max((ed - sd).days, 1)
                        total = 1.0 + float(stats_full["Return [%]"]) / 100.0
                        ann = (total ** (365.0 / float(days))) - 1.0
                        stats_full["Return (Ann.) [%]"] = ann * 100.0
            except Exception:
                pass

            # Compute Equity Final if missing from initial_capital
            try:
                if (
                    br
                    and ("Equity Final [$]" not in stats_full)
                    and stats_full.get("Return [%]") is not None
                ):
                    init_cap = None
                    if getattr(br, "run_id", None):
                        run = (
                            sess.query(um.Run)
                            .filter(um.Run.run_id == br.run_id)
                            .one_or_none()
                        )
                        if run and isinstance(run.args_json, dict):
                            init_cap = run.args_json.get("initial_capital")
                    if init_cap is None:
                        init_cap = 10000.0
                    stats_full["Equity Final [$]"] = float(init_cap) * (
                        1.0 + float(stats_full["Return [%]"]) / 100.0
                    )
            except Exception:
                pass

            # Push enriched values into overview tiles
            def _pull(name_engine: str, key_overview: str):
                try:
                    v = stats_full.get(name_engine)
                    if v is None:
                        return
                    overview[key_overview] = float(v)
                except Exception:
                    pass

            _pull("Sortino Ratio", "PSR")
            _pull("Sharpe Ratio", "sharpe_ratio")
            _pull("Return [%]", "net_profit")
            try:
                v = stats_full.get("Max. Drawdown [%]")
                if v is not None:
                    overview["max_drawdown"] = abs(float(v))
            except Exception:
                pass

            # Ensure the summary metrics table has sensible defaults even if engine_ctx is missing
            # Populate from BestStrategy/overview when BacktestResult engine_ctx is unavailable.
            try:
                if stats_full is None:
                    stats_full = {}
                # Backfill core fields if absent
                if stats_full.get("Sortino Ratio") is None:
                    stats_full["Sortino Ratio"] = overview.get("PSR")
                if stats_full.get("Sharpe Ratio") is None:
                    stats_full["Sharpe Ratio"] = overview.get("sharpe_ratio")
                if stats_full.get("Return [%]") is None:
                    stats_full["Return [%]"] = overview.get("net_profit")
                if stats_full.get("Max. Drawdown [%]") is None:
                    md = overview.get("max_drawdown")
                    if md is not None:
                        # Backtesting.py reports DD as negative percent; keep sign convention for the table
                        stats_full["Max. Drawdown [%]"] = -abs(float(md))
            except Exception:
                pass

            # Trades: prefer normalized Trade table, else parse trades_raw
            trades: list[dict] = []
            try:
                if br and getattr(br, "result_id", None):
                    rows = (
                        sess.query(um.Trade)
                        .filter(um.Trade.result_id == br.result_id)
                        .order_by(um.Trade.trade_index.asc())
                        .all()
                    )
                    for t in rows:
                        trades.append(
                            {
                                "idx": getattr(t, "trade_index", None),
                                "entry_time": getattr(t, "entry_time", None),
                                "exit_time": getattr(t, "exit_time", None),
                                "entry_bar": getattr(t, "entry_bar", None),
                                "exit_bar": getattr(t, "exit_bar", None),
                                "entry_price": getattr(t, "entry_price", None),
                                "exit_price": getattr(t, "exit_price", None),
                                "size": getattr(t, "size", None),
                                "pnl": getattr(t, "pnl", None),
                                "duration": getattr(t, "duration", None),
                                "tag": getattr(t, "tag", None),
                            }
                        )
                elif br and getattr(br, "trades_raw", None):
                    try:
                        raw = json.loads(br.trades_raw)
                        if isinstance(raw, list):
                            for i, tr in enumerate(raw):
                                if not isinstance(tr, dict):
                                    continue
                                trades.append(
                                    {
                                        "idx": tr.get("index") or i,
                                        "entry_time": tr.get("EntryTime")
                                        or tr.get("entry_time"),
                                        "exit_time": tr.get("ExitTime")
                                        or tr.get("exit_time"),
                                        "entry_bar": tr.get("entry_bar")
                                        or tr.get("EntryBar")
                                        or tr.get("entry"),
                                        "exit_bar": tr.get("exit_bar")
                                        or tr.get("ExitBar")
                                        or tr.get("exit"),
                                        "entry_price": tr.get("entry_price"),
                                        "exit_price": tr.get("exit_price"),
                                        "size": tr.get("size"),
                                        "pnl": tr.get("pnl"),
                                        "duration": tr.get("duration"),
                                        "tag": tr.get("tag"),
                                    }
                                )
                    except Exception:
                        pass
            except Exception:
                trades = []

            # As a last resort, attempt to derive total_orders from primary DB Trade rows if unified has none
            if not trades and dbm is not None and get_primary_session is not None:
                try:
                    s3 = get_primary_session()
                except Exception:
                    s3 = None
                if s3 is not None:
                    try:
                        cnt = (
                            s3.query(dbm.Trade)
                            .filter(dbm.Trade.symbol == symbol)
                            .count()
                        )
                        if cnt and cnt > 0:
                            overview["total_orders"] = int(cnt)
                    except Exception:
                        pass
                    finally:
                        try:
                            s3.close()
                        except Exception:
                            pass

            # Set total_orders from persisted trades; do not compute trades locally
            try:
                overview["total_orders"] = len(trades)
            except Exception:
                overview["total_orders"] = 0

            return {
                "best_strategy": (
                    getattr(u_bs, "strategy", None)
                    or getattr(b_bs, "strategy", "")
                    or "N/A"
                ),
                "best_timeframe": timeframe,
                "stats_full": stats_full,
                "data": {"overview": overview, "orders": trades},
                "period_start": period_start_str,
                "period_end": period_end_str,
            }
        finally:
            if sess is not None:
                try:
                    sess.close()
                except Exception:
                    pass

    def _empty_overview(self) -> Dict[str, Any]:
        return {
            "PSR": 0.0,
            "sharpe_ratio": 0.0,
            "total_orders": 0,
            "net_profit": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
        }

    def _create_html_report(
        self,
        portfolio_config: dict,
        assets_data: dict,
        start_date: str,
        end_date: str,
        strategies: list[str] | None = None,
        timeframes: list[str] | None = None,
    ) -> str:
        # Tailwind include: prefer local stylesheet or env var; fallback to CDN in dev
        try:
            import os

            tw_href = os.environ.get("TAILWIND_CSS_HREF", "").strip()
            # If env var points to a local path that doesn't exist, ignore it to allow CDN fallback
            if tw_href and not tw_href.startswith(("http://", "https://")):
                try:
                    if not Path(tw_href).exists():
                        tw_href = ""
                except Exception:
                    tw_href = ""
            if not tw_href:
                cand = Path("exports/reports/assets/tailwind.min.css")
                if cand.exists():
                    tw_href = str(cand)
            tailwind_tag = (
                f'<link rel="stylesheet" href="{tw_href}">'
                if tw_href
                else '<script src="https://cdn.tailwindcss.com"></script>'
            )
        except Exception:
            tailwind_tag = '<script src="https://cdn.tailwindcss.com"></script>'
        # Plotly include (for inline equity charts)
        plotly_tag = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'

        # Top overview (computed from assets_data)
        total_assets = len(assets_data)
        avg_sortino = 0.0
        winners = 0
        traders = 0
        vals = []
        for data in assets_data.values():
            ov = (data.get("data") or {}).get("overview") or {}
            try:
                vals.append(float(ov.get("PSR", 0) or 0))
            except Exception:
                pass
            try:
                if float(ov.get("net_profit", 0) or 0) > 0:
                    winners += 1
            except Exception:
                pass
            try:
                if int(ov.get("total_orders", 0) or 0) > 0:
                    traders += 1
            except Exception:
                pass
        if vals:
            avg_sortino = sum(vals) / len(vals)

        # Backtest settings card (strategies, intervals, period)
        strat_list = ", ".join(strategies or [])
        tf_list = ", ".join(timeframes or [])
        # Prefer derived period from assets_data (global earliest start, latest end)
        try:
            derived_starts = []
            derived_ends = []
            for v in assets_data.values():
                ps = v.get("period_start")
                pe = v.get("period_end")
                if isinstance(ps, str) and len(ps) >= 10:
                    derived_starts.append(ps[:10])
                if isinstance(pe, str) and len(pe) >= 10:
                    derived_ends.append(pe[:10])
            derived_start = min(derived_starts) if derived_starts else None
            derived_end = max(derived_ends) if derived_ends else None
        except Exception:
            derived_start = None
            derived_end = None

        period_str = (
            f"{derived_start} → {derived_end}"
            if (derived_start and derived_end)
            else (f"{start_date} → {end_date}" if (start_date and end_date) else "max")
        )
        settings_card = f"""
        <details class=\"rounded-xl border border-white/10 bg-white/5 mb-4\">
          <summary class=\"px-4 py-3 cursor-pointer select-none text-sm font-semibold text-slate-300\">Backtest Settings</summary>
          <div class=\"p-4 pt-2\">
            <div class=\"grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3\">
              <div><div class=\"text-xs uppercase text-slate-400\">Intervals</div><div class=\"font-medium\">{tf_list or "-"} </div></div>
              <div><div class=\"text-xs uppercase text-slate-400\">Strategies</div><div class=\"font-medium\">{strat_list or "-"} </div></div>
              <div><div class=\"text-xs uppercase text-slate-400\">Period</div><div class=\"font-medium\">{period_str}</div></div>
            </div>
          </div>
        </details>
        """

        top_overview = f"""
        <div class=\"grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-4\">
          <div class=\"rounded-xl border border-white/10 bg-white/5 p-4\">
            <div class=\"text-xs uppercase tracking-wide text-slate-300\">Assets</div>
            <div class=\"text-xl font-semibold\">{total_assets}</div>
          </div>
          <div class=\"rounded-xl border border-white/10 bg-white/5 p-4\">
            <div class=\"text-xs uppercase tracking-wide text-slate-300\">Avg Sortino</div>
            <div class=\"text-xl font-semibold text-emerald-400\">{avg_sortino:.3f}</div>
          </div>
          <div class=\"rounded-xl border border-white/10 bg-white/5 p-4\">
            <div class=\"text-xs uppercase tracking-wide text-slate-300\">Positive Returns</div>
            <div class=\"text-xl font-semibold\">{winners}</div>
          </div>
          <div class=\"rounded-xl border border-white/10 bg-white/5 p-4\">
            <div class=\"text-xs uppercase tracking-wide text-slate-300\">With Trades</div>
            <div class=\"text-xl font-semibold\">{traders}</div>
          </div>
        </div>
        """

        # Sidebar TOC (TailAdmin-style): sticky on large screens, compact chips on mobile
        toc_items = [
            f'<li><a href="#asset-{sym}" class="block px-3 py-2 rounded-lg hover:bg-white/10 transition text-slate-200">{sym}</a></li>'
            for sym in assets_data.keys()
        ]
        sidebar_html = (
            '<aside class="hidden lg:block w-64 mr-6">'
            '  <div class="sticky top-6 rounded-xl border border-white/10 bg-white/5 backdrop-blur p-4">'
            '    <h2 class="text-sm font-semibold text-slate-300 mb-3">Assets</h2>'
            '    <input id="assetFilter" type="text" placeholder="Filter…" class="mb-3 w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-400" />'
            '    <ul id="assetList" class="space-y-1">' + "".join(toc_items) + "</ul>"
            "  </div>"
            "</aside>"
        )
        # Mobile chips
        chips_html = (
            '<div class="lg:hidden sticky top-4 z-10 mb-4 overflow-x-auto whitespace-nowrap py-2">'
            '  <div class="inline-flex gap-2">'
            + "".join(
                [
                    f'<a href="#asset-{sym}" class="px-3 py-1 rounded-full border border-white/10 bg-white/5 hover:bg-white/10 text-sm">{sym}</a>'
                    for sym in assets_data.keys()
                ]
            )
            + "  </div>"
            "</div>"
        )

        # Asset sections
        asset_sections = []
        for symbol, data in assets_data.items():
            overview = (data.get("data") or {}).get("overview") or {}
            stats = data.get("stats_full") or {}

            def fmt(v: Any, prec=2, pct=False, money=False) -> str:
                try:
                    if v is None:
                        return "-"
                    f = float(v)
                    if money:
                        return f"${f:,.{prec}f}"
                    if pct:
                        return f"{f:.{prec}f}%"
                    return f"{f:.{prec}f}"
                except Exception:
                    return str(v) if v is not None else "-"

            metrics_row = f"""
            <div class=\"overflow-hidden rounded-xl border border-white/10 bg-white/5\">
              <table class=\"min-w-full divide-y divide-white/10\">
                <thead class=\"bg-white/5\">
                  <tr class=\"text-slate-300 text-xs uppercase tracking-wider\">
                    <th class=\"px-4 py-3 text-left\">Equity Final</th>
                    <th class=\"px-4 py-3 text-left\">Commissions</th>
                    <th class=\"px-4 py-3 text-left\">Return</th>
                    <th class=\"px-4 py-3 text-left\">Buy &amp; Hold Return</th>
                    <th class=\"px-4 py-3 text-left\">Sortino</th>
                    <th class=\"px-4 py-3 text-left\">Sharpe</th>
                    <th class=\"px-4 py-3 text-left\">Return (Ann.)</th>
                    <th class=\"px-4 py-3 text-left\">Max DD</th>
                    <th class=\"px-4 py-3 text-left\">Win Rate</th>
                  </tr>
                </thead>
                <tbody class=\"divide-y divide-white/5\">
                  <tr>
                    <td class=\"px-4 py-3\">{fmt(stats.get("Equity Final [$]"), 2, money=True)}</td>
                    <td class=\"px-4 py-3\">{fmt(stats.get("Commissions [$]"), 2, money=True)}</td>
                    <td class=\"px-4 py-3\">{fmt(stats.get("Return [%]"), 2, pct=True)}</td>
                    <td class=\"px-4 py-3\">{fmt(stats.get("Buy & Hold Return [%]"), 2, pct=True)}</td>
                    <td class=\"px-4 py-3\">{fmt(stats.get("Sortino Ratio"), 3)}</td>
                    <td class=\"px-4 py-3\">{fmt(stats.get("Sharpe Ratio"), 3)}</td>
                    <td class=\"px-4 py-3\">{fmt(stats.get("Return (Ann.) [%]"), 2, pct=True)}</td>
                    <td class=\"px-4 py-3\">{fmt(stats.get("Max. Drawdown [%]"), 2, pct=True)}</td>
                    <td class=\"px-4 py-3\">{fmt(stats.get("Win Rate [%]"), 2, pct=True)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            """

            # Build simple sparkline from any available equity series
            equity_series = []
            try:
                for k in (
                    "equity_curve",
                    "equity",
                    "equity_values",
                    "Equity Curve",
                    "equity_series",
                ):
                    v = stats.get(k)
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        equity_series = [float(x) for x in v if x is not None]
                        break
                # Backtesting.py direct stats often store '_equity_curve' as list of dicts
                if not equity_series:
                    v2 = stats.get("_equity_curve")
                    if isinstance(v2, list) and len(v2) >= 2:
                        try:
                            pts = []
                            for row in v2:
                                if isinstance(row, dict) and "Equity" in row:
                                    pts.append(float(row.get("Equity")))
                            if len(pts) >= 2:
                                equity_series = pts
                        except Exception:
                            pass
                if not equity_series and isinstance(stats.get("series"), list):
                    s0 = stats["series"][0]
                    if isinstance(s0, dict) and isinstance(s0.get("y"), list):
                        equity_series = [float(x) for x in s0["y"] if x is not None]
            except Exception:
                equity_series = []

            def _spark(points: list[float], width=600, height=80) -> str:
                try:
                    if not points or len(points) < 2:
                        return ""
                    mn = min(points)
                    mx = max(points)
                    rng = (mx - mn) or 1.0
                    step = width / (len(points) - 1)
                    cmds = []
                    for i, v in enumerate(points):
                        x = i * step
                        y = height - ((float(v) - mn) / rng) * height
                        cmds.append(("M" if i == 0 else "L") + f" {x:.2f} {y:.2f}")
                    d = " ".join(cmds)
                    return (
                        f'<svg viewBox="0 0 {int(width)} {int(height)}" class="w-full h-20" '
                        'xmlns="http://www.w3.org/2000/svg">'
                        f'<path d="{d}" fill="none" stroke="#22d3ee" stroke-width="2" />'
                        "</svg>"
                    )
                except Exception:
                    return ""

            spark = _spark(equity_series)
            # Fallback: embed plot HTML if provided by engine_ctx
            plot_embed = None
            try:
                for k in ("plot_html", "plot_div", "plot", "chart_html"):
                    v = stats.get(k)
                    if isinstance(v, str) and ("<svg" in v or "<div" in v):
                        plot_embed = v
                        break
            except Exception:
                plot_embed = None

            # Plotly chart from equity series (primary when available)
            def _safe_id(s: str) -> str:
                try:
                    import re as _re

                    return _re.sub(r"[^A-Za-z0-9_\-]", "_", s)
                except Exception:
                    return s

            def _plotly_equity(
                sym: str, eq: list[float], stats_obj: dict, orders: list[dict]
            ) -> str:
                try:
                    import json as _json

                    if not eq or len(eq) < 2:
                        return ""
                    x = list(range(len(eq)))
                    dd = []
                    try:
                        if isinstance(stats_obj.get("_equity_curve"), list):
                            vals = []
                            has_dd = False
                            for r in stats_obj["_equity_curve"]:
                                if (
                                    isinstance(r, dict)
                                    and r.get("DrawdownPct") is not None
                                ):
                                    has_dd = True
                                    vals.append(float(r.get("DrawdownPct")))
                            if has_dd and len(vals) == len(eq):
                                dd = vals
                    except Exception:
                        dd = []
                    div_id = f"plot_{_safe_id(sym)}"
                    data = [
                        {
                            "x": x,
                            "y": eq,
                            "type": "scatter",
                            "mode": "lines",
                            "name": "Equity",
                            "line": {"color": "#22d3ee"},
                        }
                    ]
                    layout = {
                        "margin": {"l": 30, "r": 10, "t": 10, "b": 30},
                        "paper_bgcolor": "rgba(0,0,0,0)",
                        "plot_bgcolor": "rgba(0,0,0,0)",
                        "xaxis": {"showgrid": False, "zeroline": False},
                        "yaxis": {"showgrid": False, "zeroline": False},
                        "showlegend": True,
                    }
                    if dd:
                        data.append(
                            {
                                "x": x,
                                "y": dd,
                                "type": "scatter",
                                "mode": "lines",
                                "name": "Drawdown [%]",
                                "line": {"color": "#f43f5e"},
                                "yaxis": "y2",
                            }
                        )
                        layout["yaxis2"] = {"overlaying": "y", "side": "right"}

                    # Buy & Hold overlay if metric present
                    try:
                        bnh = stats_obj.get("Buy & Hold Return [%]")
                        if bnh is not None and len(eq) >= 2:
                            eq0 = float(eq[0])
                            eq_bnh_end = eq0 * (1.0 + float(bnh) / 100.0)
                            # Linear interpolation for lack of series
                            y_bnh = [
                                eq0 + (eq_bnh_end - eq0) * (i / (len(x) - 1))
                                for i in range(len(x))
                            ]
                            data.append(
                                {
                                    "x": x,
                                    "y": y_bnh,
                                    "type": "scatter",
                                    "mode": "lines",
                                    "name": "Buy & Hold",
                                    "line": {"color": "#a3e635", "dash": "dash"},
                                }
                            )
                    except Exception:
                        pass

                    # Entry/Exit markers from orders using entry_bar/exit_bar indices
                    try:
                        entries_x = []
                        entries_y = []
                        exits_x = []
                        exits_y = []
                        for od in orders or []:
                            eb = od.get("entry_bar")
                            xb = od.get("exit_bar")
                            if isinstance(eb, (int, float)) and 0 <= int(eb) < len(eq):
                                idx = int(eb)
                                entries_x.append(x[idx])
                                entries_y.append(eq[idx])
                            if isinstance(xb, (int, float)) and 0 <= int(xb) < len(eq):
                                idx = int(xb)
                                exits_x.append(x[idx])
                                exits_y.append(eq[idx])
                        if entries_x:
                            data.append(
                                {
                                    "x": entries_x,
                                    "y": entries_y,
                                    "type": "scatter",
                                    "mode": "markers",
                                    "name": "Entry",
                                    "marker": {
                                        "color": "#22c55e",
                                        "size": 6,
                                        "symbol": "triangle-up",
                                    },
                                }
                            )
                        if exits_x:
                            data.append(
                                {
                                    "x": exits_x,
                                    "y": exits_y,
                                    "type": "scatter",
                                    "mode": "markers",
                                    "name": "Exit",
                                    "marker": {
                                        "color": "#ef4444",
                                        "size": 6,
                                        "symbol": "triangle-down",
                                    },
                                }
                            )
                    except Exception:
                        pass
                    payload = _json.dumps(
                        {
                            "data": data,
                            "layout": layout,
                            "config": {"displayModeBar": False, "responsive": True},
                        }
                    )
                    return (
                        f'<div id="{div_id}" class="w-full h-64"></div>'
                        f"<script>(function(){{var p={payload};Plotly.newPlot('{div_id}', p.data, p.layout, p.config);}})();</script>"
                    )
                except Exception:
                    return ""

            plotly_plot = _plotly_equity(
                symbol,
                equity_series,
                stats,
                (data.get("data") or {}).get("orders") or [],
            )
            placeholder_plot = plot_embed or plotly_plot or spark
            if not placeholder_plot:
                placeholder_plot = '<div class="text-slate-300">Plotting disabled in this environment.</div>'
            plot_section = f"""
          <h3 class=\"text-sm font-medium text-slate-300 mt-4 mb-2\">Equity Curve</h3>
          <div class=\"rounded-xl border border-white/10 bg-white/5 p-4\">{placeholder_plot}</div>
                """

            # Trades table if any
            trades = (data.get("data") or {}).get("orders") or []
            trades_html = ""
            if isinstance(trades, list) and trades:
                trade_rows = []
                for tr in trades[:200]:

                    def _fmt_dt(v):
                        try:
                            import datetime as _dt

                            if v is None:
                                return ""
                            if isinstance(v, str):
                                return v
                            if isinstance(v, (_dt.datetime, _dt.date)):
                                return v.isoformat()
                        except Exception:
                            return str(v) if v is not None else ""
                        return str(v)

                    trade_rows.append(
                        f"<tr>"
                        f'<td class="px-3 py-2">{tr.get("idx", "")}</td>'
                        f'<td class="px-3 py-2">{_fmt_dt(tr.get("entry_time"))}</td>'
                        f'<td class="px-3 py-2">{_fmt_dt(tr.get("exit_time"))}</td>'
                        f'<td class="px-3 py-2">{tr.get("size", "")}</td>'
                        f'<td class="px-3 py-2">{tr.get("entry_price", "")}</td>'
                        f'<td class="px-3 py-2">{tr.get("exit_price", "")}</td>'
                        f'<td class="px-3 py-2">{tr.get("pnl", "")}</td>'
                        f'<td class="px-3 py-2">{tr.get("duration", "")}</td>'
                        f'<td class="px-3 py-2">{tr.get("tag", "")}</td>'
                        f"</tr>"
                    )
                trades_html = f"""
<h3 class="text-sm font-medium text-slate-300 mt-4 mb-2">Trades</h3>
<div class="overflow-hidden rounded-xl border border-white/10 bg-white/5">
  <table class="min-w-full divide-y divide-white/10 text-sm">
    <thead class="bg-white/5">
      <tr class="text-slate-300 uppercase tracking-wider text-xs">
        <th class="px-3 py-2 text-left">#</th>
        <th class="px-3 py-2 text-left">Entry Time</th>
        <th class="px-3 py-2 text-left">Exit Time</th>
        <th class="px-3 py-2 text-left">Size</th>
        <th class="px-3 py-2 text-left">Entry</th>
        <th class="px-3 py-2 text-left">Exit</th>
        <th class="px-3 py-2 text-left">PnL</th>
        <th class="px-3 py-2 text-left">Duration</th>
        <th class="px-3 py-2 text-left">Tag</th>
      </tr>
    </thead>
    <tbody class="divide-y divide-white/5">{"".join(trade_rows)}</tbody>
  </table>
</div>
                """

            asset_sections.append(
                f"""
        <section id=\"asset-{symbol}\" class=\"py-6 border-t border-white/10\">
          <div class=\"flex items-center justify-between mb-4\">
            <h2 class=\"text-xl font-semibold text-white\">{symbol}</h2>
            <div class=\"flex items-center gap-2\">
              <span class=\"inline-flex items-center gap-2 rounded-full border border-indigo-400/40 bg-indigo-500/10 px-3 py-1 text-sm\">Best: {(data.get("best_strategy") or "N/A")}</span>
              <span class=\"inline-flex items-center gap-2 rounded-full border border-cyan-400/40 bg-cyan-500/10 px-3 py-1 text-sm\">⏰ {data.get("best_timeframe", "1d")}</span>
            </div>
          </div>
          <h3 class=\"text-sm font-medium text-slate-300 mb-2\">Summary Metrics</h3>
          {metrics_row}
          <div class=\"grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mt-4\">
            <div class=\"rounded-xl border border-white/10 bg-white/5 p-4\">
              <div class=\"text-xs uppercase tracking-wide text-slate-300\">Sortino</div>
              <div class=\"text-lg font-semibold text-emerald-400\">{overview.get("PSR", 0):.3f}</div>
            </div>
            <div class=\"rounded-xl border border-white/10 bg-white/5 p-4\">
              <div class=\"text-xs uppercase tracking-wide text-slate-300\">Sharpe</div>
              <div class=\"text-lg font-semibold\">{overview.get("sharpe_ratio", 0):.3f}</div>
            </div>
            <div class=\"rounded-xl border border-white/10 bg-white/5 p-4\">
              <div class=\"text-xs uppercase tracking-wide text-slate-300\">Orders</div>
              <div class=\"text-lg font-semibold\">{int(overview.get("total_orders", 0) or 0)}</div>
            </div>
            <div class=\"rounded-xl border border-white/10 bg-white/5 p-4\">
              <div class=\"text-xs uppercase tracking-wide text-slate-300\">Net Profit</div>
              <div class=\"text-lg font-semibold {("text-emerald-400" if float(overview.get("net_profit", 0) or 0) > 0 else "text-rose-400")}\">{overview.get("net_profit", 0):.2f}%</div>
            </div>
            <div class=\"rounded-xl border border-white/10 bg-white/5 p-4\">
              <div class=\"text-xs uppercase tracking-wide text-slate-300\">Max Drawdown</div>
              <div class=\"text-lg font-semibold text-rose-400\">-{overview.get("max_drawdown", 0):.2f}%</div>
            </div>
            <div class=\"rounded-xl border border-white/10 bg-white/5 p-4\">
              <div class=\"text-xs uppercase tracking-wide text-slate-300\">Calmar</div>
              <div class=\"text-lg font-semibold\">{overview.get("calmar_ratio", 0):.3f}</div>
            </div>
          </div>
          {plot_section}
          {trades_html}
        </section>
                """
            )

        # Use double braces for literal braces in .format()
        # Footer: educational disclaimer + project link
        footer_html = (
            '<footer class="mt-10 text-center text-slate-400 text-xs">'
            '<div class="rounded-xl border border-white/10 bg-white/5 p-4">'
            "This report is for educational purposes only and does not constitute financial advice. "
            'Project: <a class="text-cyan-300 underline" href="https://github.com/LouisLetcher/quant-system" target="_blank" rel="noopener noreferrer">quant-system</a>.'
            "</div>"
            "</footer>"
        )
        html_template = """<!DOCTYPE html>
<html>
<head>
  <meta charset=\"UTF-8\">
  <title>Collection Analysis: {{portfolio_name}}</title>
  {tailwind_tag}
  {plotly_tag}
  <style>
    body {font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 0; padding: 20px; background: #0b0f19;}
  </style>
</head>
<body class=\"bg-slate-950 bg-gradient-to-br from-indigo-900/20 via-slate-900 to-cyan-900/20 text-slate-100\">
  <div class=\"max-w-7xl mx-auto p-6\">
    <header class=\"rounded-xl border border-white/10 bg-white/5 backdrop-blur p-5 shadow mb-4\">
      <h1 class=\"text-2xl font-semibold tracking-tight\">{{portfolio_name}}</h1>
      <p class=\"text-slate-300 mt-1\">Real Backtesting Data • {{start_date}} → {{end_date}}</p>
    </header>
    <div class=\"flex\">
      {sidebar_html}
      <main class=\"flex-1\">
        {settings_card}
        {top_overview}
        {chips_html}
        {{asset_sections}}
      </main>
    </div>
  </div>
  {footer_html}
  <script>
    (function(){
      // Sidebar filter is already wired in the template; add Expand/Collapse All
      function setAll(open){
        const sections = document.querySelectorAll('section[id^="asset-"]');
        sections.forEach(sec => {
          const children = Array.from(sec.children);
          // keep first child (header) visible; toggle the rest
          for (let i = 1; i < children.length; i++) {
            children[i].style.display = open ? '' : 'none';
          }
        });
      }
      const expandBtn = document.getElementById('expandAll');
      const collapseBtn = document.getElementById('collapseAll');
      if (expandBtn) expandBtn.addEventListener('click', () => setAll(true));
      if (collapseBtn) collapseBtn.addEventListener('click', () => setAll(false));
    })();
  </script>
</body>
</html>"""

        # Brace-safe rendering: protect placeholders, escape all braces, then restore placeholders
        tokens = {
            "[[PORTFOLIO_NAME]]": "{portfolio_name}",
            "[[START_DATE]]": "{start_date}",
            "[[END_DATE]]": "{end_date}",
            "[[ASSET_SECTIONS]]": "{asset_sections}",
            "[[TAILWIND_TAG]]": "{tailwind_tag}",
            "[[PLOTLY_TAG]]": "{plotly_tag}",
            "[[SIDEBAR_HTML]]": "{sidebar_html}",
            "[[TOP_OVERVIEW]]": "{top_overview}",
            "[[CHIPS_HTML]]": "{chips_html}",
            "[[SETTINGS_CARD]]": "{settings_card}",
            "[[FOOTER_HTML]]": "{footer_html}",
        }
        # Mark placeholders
        html_template_marked = (
            html_template.replace("{{portfolio_name}}", "[[PORTFOLIO_NAME]]")
            .replace("{{start_date}}", "[[START_DATE]]")
            .replace("{{end_date}}", "[[END_DATE]]")
            .replace("{{asset_sections}}", "[[ASSET_SECTIONS]]")
            .replace("{tailwind_tag}", "[[TAILWIND_TAG]]")
            .replace("{plotly_tag}", "[[PLOTLY_TAG]]")
            .replace("{sidebar_html}", "[[SIDEBAR_HTML]]")
            .replace("{top_overview}", "[[TOP_OVERVIEW]]")
            .replace("{chips_html}", "[[CHIPS_HTML]]")
            .replace("{settings_card}", "[[SETTINGS_CARD]]")
            .replace("{footer_html}", "[[FOOTER_HTML]]")
        )
        # Escape all remaining braces so they render literally
        html_template_escaped = html_template_marked.replace("{", "{{").replace(
            "}", "}}"
        )
        # Restore placeholders
        for t, ph in tokens.items():
            html_template_escaped = html_template_escaped.replace(t, ph)

        # Choose header dates: prefer derived period
        header_start = derived_start or start_date
        header_end = derived_end or end_date

        return html_template_escaped.format(
            portfolio_name=portfolio_config.get("name") or "Portfolio",
            start_date=header_start,
            end_date=header_end,
            asset_sections="\n".join(asset_sections),
            tailwind_tag=tailwind_tag,
            plotly_tag=plotly_tag,
            sidebar_html=sidebar_html,
            top_overview=top_overview,
            chips_html=chips_html,
            settings_card=settings_card,
            footer_html=footer_html,
        )

    def _save_report(
        self, html_content: str, portfolio_name: str, interval: str
    ) -> str:
        # Save via organizer using unified naming (exports/reports/<year>/Q<q>/<name>_Collection_<year>_Q<q>_<interval>.html)
        tmp = Path("temp_report.html")
        tmp.write_text(html_content, encoding="utf-8")
        try:
            return str(
                self.report_organizer.organize_report(
                    str(tmp), portfolio_name, None, interval=interval
                )
            )
        finally:
            if tmp.exists():
                tmp.unlink()
