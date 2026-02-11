from __future__ import annotations

import html
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from ..reporting.dashboard import DOWNLOAD_FILE_CANDIDATES


def create_app(reports_dir: Path) -> FastAPI:
    root = Path(reports_dir)
    base_root: Path | None = None

    base_css = """
    :root {
      --bg: #020617;
      --panel: #0f172a;
      --panel-2: #111c33;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --border: rgba(148, 163, 184, 0.22);
      --link: #38bdf8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell,
        Noto Sans, Arial, "Apple Color Emoji", "Segoe UI Emoji";
    }
    a { color: var(--link); text-decoration: underline; }
    a:hover { opacity: 0.9; }
    code { font-family: ui-monospace, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    .container { max-width: 72rem; margin: 0 auto; padding: 2rem 1rem; }
    .stack > * + * { margin-top: 1rem; }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 1rem;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.35);
    }
    .row { display: flex; gap: 1rem; align-items: baseline; justify-content: space-between; flex-wrap: wrap; }
    .muted { color: var(--muted); font-size: 0.9rem; }
    .title { font-size: 1.25rem; font-weight: 700; margin: 0; }
    .subtitle { margin: 0.25rem 0 0; }
    .btn {
      display: inline-block;
      padding: 0.35rem 0.6rem;
      border-radius: 0.5rem;
      background: var(--panel-2);
      border: 1px solid var(--border);
      text-decoration: none;
      color: var(--text);
      font-size: 0.9rem;
    }
    table { width: 100%; border-collapse: collapse; }
    thead th {
      text-align: left;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      background: rgba(148, 163, 184, 0.08);
      border-bottom: 1px solid var(--border);
      padding: 0.6rem 0.75rem;
      white-space: nowrap;
    }
    tbody td {
      border-bottom: 1px solid var(--border);
      padding: 0.55rem 0.75rem;
      vertical-align: top;
    }
    tbody tr:hover td { background: rgba(148, 163, 184, 0.05); }
    ul { margin: 0.25rem 0 0.25rem 1.2rem; padding: 0; }
    li { margin: 0.2rem 0; }
    .table-wrap { overflow-x: auto; }
    """.strip()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal base_root
        if not root.exists():
            raise RuntimeError(f"Reports directory not found: {root}")
        base_root = root.resolve()
        yield

    app = FastAPI(title="Quant System Dashboard", lifespan=lifespan)

    def _escape(value: Any) -> str:
        if value is None:
            return ""
        return html.escape(str(value), quote=True)

    def _url_segment(value: Any) -> str:
        return quote(str(value), safe="")

    def _file_url(path: Path) -> str:
        return f"file://{quote(str(path), safe='/:')}"

    def _base_root() -> Path:
        return base_root if base_root is not None else root.resolve()

    def _is_relative_to(path: Path, base: Path) -> bool:
        try:
            path.relative_to(base)
        except ValueError:
            return False
        return True

    def _safe_filename(filename: str) -> str:
        if not filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        if "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        if filename.startswith(".") or ".." in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        if Path(filename).is_absolute():
            raise HTTPException(status_code=400, detail="Invalid filename")
        return filename

    def _runs() -> list[Path]:
        if not root.exists():
            return []
        return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)

    def _runs_by_name() -> dict[str, Path]:
        return {run_dir.name: run_dir for run_dir in _runs()}

    def _load_summary(run_dir: Path) -> dict[str, Any]:
        summary_path = (run_dir / "summary.json").resolve()
        if not _is_relative_to(summary_path, _base_root()):
            return {}
        if not summary_path.exists():
            return {}
        try:
            return json.loads(summary_path.read_text())
        except Exception:
            return {}

    def _load_summary_csv(run_dir: Path, limit: int = 5) -> list[dict[str, Any]]:
        csv_path = (run_dir / "summary.csv").resolve()
        if not _is_relative_to(csv_path, _base_root()):
            return []
        if not csv_path.exists():
            return []
        try:
            import csv

            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = []
                for idx, row in enumerate(reader):
                    if idx >= limit:
                        break
                    rows.append(row)
                return rows
        except Exception:
            return []

    @app.get("/api/runs")
    async def list_runs() -> list[dict[str, Any]]:
        runs = []
        for run_dir in _runs():
            summary = _load_summary(run_dir)
            runs.append(
                {
                    "run_id": run_dir.name,
                    "metric": summary.get("metric"),
                    "results_count": summary.get("results_count"),
                    "started_at": summary.get("started_at"),
                    "finished_at": summary.get("finished_at"),
                    "path": str(run_dir),
                    "summary": summary,
                }
            )
        return runs

    @app.get("/api/runs/{run_id}")
    async def run_detail(run_id: str) -> dict[str, Any]:
        run_dir = _runs_by_name().get(run_id)
        if run_dir is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if not run_dir.exists() or not run_dir.is_dir():
            raise HTTPException(status_code=404, detail="Run not found")
        summary = _load_summary(run_dir)
        top_rows = _load_summary_csv(run_dir)
        return {
            "run_id": run_dir.name,
            "summary": summary,
            "top": top_rows,
            "path": str(run_dir),
        }

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        rows = []
        for run_dir in _runs():
            summary = _load_summary(run_dir)
            rows.append(
                {
                    "run_id": _escape(run_dir.name),
                    "metric": _escape(summary.get("metric")),
                    "results_count": _escape(summary.get("results_count")),
                    "started_at": _escape(summary.get("started_at")),
                    "finished_at": _escape(summary.get("finished_at")),
                    "detail_url": _escape(f"/run/{_url_segment(run_dir.name)}"),
                    "report_url": _escape(
                        f"/api/runs/{_url_segment(run_dir.name)}/files/report.html"
                    ),
                }
            )
        html_rows = "".join(
            "<tr>"
            f"<td>{r['run_id']}</td>"
            f"<td>{r.get('metric', '')}</td>"
            f"<td>{r.get('results_count', '')}</td>"
            f"<td>{r.get('started_at', '')}</td>"
            f"<td><a href='{r['detail_url']}'>View</a></td>"
            f"<td><a href='{r['report_url']}'>Open</a></td>"
            "</tr>"
            for r in rows
        )
        html = f"""
<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Quant System Dashboard</title>
  <style>{base_css}</style>
</head>
<body>
  <div class='container stack'>
    <header>
      <h1 class='title'>Quant System Runs</h1>
      <p class='muted subtitle'>Browse recent runs, review summaries, and open detailed reports.</p>
    </header>
    <section class='card'>
      <div class='table-wrap'>
        <table>
          <thead>
            <tr>
              <th>Run ID</th>
              <th>Metric</th>
              <th>Results</th>
              <th>Started</th>
              <th>Detail</th>
              <th>Report</th>
            </tr>
          </thead>
          <tbody>
            {html_rows if html_rows else "<tr><td class='muted' colspan='6'>No runs available</td></tr>"}
          </tbody>
        </table>
      </div>
    </section>
  </div>
</body>
</html>
        """
        return HTMLResponse(html)

    @app.get("/run/{run_id}", response_class=HTMLResponse)
    async def run_page(run_id: str) -> HTMLResponse:
        run_dir = _runs_by_name().get(run_id)
        if run_dir is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail="Run not found")
        summary = _load_summary(run_dir)
        manifest = []
        manifest_path = run_dir / "manifest_status.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except Exception:
                manifest = []
        notifications = []
        notifications_path = run_dir / "notifications.json"
        if notifications_path.exists():
            try:
                notifications = json.loads(notifications_path.read_text())
            except Exception:
                notifications = []

        summary_rows = "".join(
            f"<tr><td><strong>{_escape(k)}</strong></td><td>{_escape(v)}</td></tr>"
            for k, v in summary.items()
            if k in {"metric", "results_count", "started_at", "finished_at", "duration_sec"}
        )

        manifest_rows = (
            "".join(
                "<tr>"
                f"<td>{_escape(m.get('run_id'))}</td>"
                f"<td>{_escape(m.get('status'))}</td>"
                f"<td>{_escape(m.get('message', ''))}</td>"
                "</tr>"
                for m in manifest
            )
            or "<tr><td class='muted' colspan='3'>No manifest actions</td></tr>"
        )

        notification_rows = (
            "".join(
                "<tr>"
                f"<td>{_escape(n.get('channel'))}</td>"
                f"<td>{_escape(n.get('metric'))}</td>"
                f"<td>{_escape('sent' if n.get('sent') else n.get('reason', 'skipped'))}</td>"
                "</tr>"
                for n in notifications
            )
            or "<tr><td class='muted' colspan='3'>No notifications</td></tr>"
        )

        downloads = []
        for name in DOWNLOAD_FILE_CANDIDATES:
            if (run_dir / name).exists():
                run_id_safe = _url_segment(run_dir.name)
                name_safe = _url_segment(name)
                href = _escape(f"/api/runs/{run_id_safe}/files/{name_safe}")
                downloads.append(
                    f"<li><a class='text-sky-400 underline' href='{href}'>{_escape(name)}</a></li>"
                )
        downloads_html = "".join(downloads) or "<li class='muted'>No files</li>"

        run_id_text = _escape(run_dir.name)
        html = f"""
<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Run {run_id_text}</title>
  <style>{base_css}</style>
</head>
<body>
  <div class='container stack'>
    <header>
      <div class='row'>
        <h1 class='title'>Run {run_id_text}</h1>
        <a class='btn' href='/'>Back to runs</a>
      </div>
    </header>
    <section class='card'>
      <h2 class='title'>Summary</h2>
      <div class='table-wrap'>
        <table>
          <tbody>{summary_rows}</tbody>
        </table>
      </div>
      <div>
        <h3 class='title' style='font-size: 1rem;'>Downloads</h3>
        <ul>{downloads_html}</ul>
      </div>
    </section>
    <section class='card'>
      <h2 class='title'>Manifest Actions</h2>
      <div class='table-wrap'>
        <table>
          <thead>
            <tr><th>Run</th><th>Status</th><th>Message</th></tr>
          </thead>
          <tbody>{manifest_rows}</tbody>
        </table>
      </div>
    </section>
    <section class='card'>
      <h2 class='title'>Notifications</h2>
      <div class='table-wrap'>
        <table>
          <thead>
            <tr><th>Channel</th><th>Metric</th><th>Outcome</th></tr>
          </thead>
          <tbody>{notification_rows}</tbody>
        </table>
      </div>
    </section>
  </div>
</body>
</html>
        """
        return HTMLResponse(html)

    @app.get("/api/runs/{run_id}/files/{filename}")
    async def download(run_id: str, filename: str):
        run_dir = _runs_by_name().get(run_id)
        if run_dir is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if not run_dir.exists() or not run_dir.is_dir():
            raise HTTPException(status_code=404, detail="Run not found")
        filename = _safe_filename(filename)
        allowed = set(DOWNLOAD_FILE_CANDIDATES)
        if filename not in allowed:
            raise HTTPException(status_code=404, detail="File not available")
        file_path = (run_dir / filename).resolve()
        if not _is_relative_to(file_path, run_dir):
            raise HTTPException(status_code=404, detail="File not found")
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(file_path)

    @app.get("/api/compare")
    async def compare_runs(runs: str) -> dict[str, Any]:
        run_ids = [r.strip() for r in runs.split(",") if r.strip()]
        if not run_ids:
            raise HTTPException(status_code=400, detail="Provide at least one run id")
        available_runs = _runs_by_name()
        payload = {}
        for run_id in run_ids:
            run_dir = available_runs.get(run_id)
            if run_dir is None:
                continue
            payload[run_dir.name] = {
                "summary": _load_summary(run_dir),
                "top": _load_summary_csv(run_dir, limit=10),
            }
        if not payload:
            raise HTTPException(status_code=404, detail="No runs found")
        return payload

    @app.get("/compare", response_class=HTMLResponse)
    async def compare_page(runs: str) -> HTMLResponse:
        api_data = await compare_runs(runs)
        run_cards = []
        for run_id, data in api_data.items():
            rows = data.get("top") or []
            table_rows = (
                "".join(
                    "<tr>"
                    f"<td>{_escape(row.get('symbol'))}</td>"
                    f"<td>{_escape(row.get('strategy'))}</td>"
                    f"<td>{_escape(row.get('metric'))}</td>"
                    f"<td>{_escape(row.get('metric_value'))}</td>"
                    "</tr>"
                    for row in rows
                )
                or "<tr><td class='muted' colspan='4'>No summary.csv data</td></tr>"
            )
            summary_meta = data.get("summary", {})
            run_id_text = _escape(run_id)
            run_cards.append(
                f"""
                <section class='card'>
                  <header class='row'>
                    <h2 class='title' style='font-size: 1.05rem; margin: 0;'>{run_id_text}</h2>
                    <p class='muted' style='margin: 0;'>Metric: {_escape(summary_meta.get("metric", ""))}</p>
                  </header>
                  <div class='table-wrap'>
                    <table>
                      <thead>
                        <tr><th>Symbol</th><th>Strategy</th><th>Metric</th><th>Value</th></tr>
                      </thead>
                      <tbody>{table_rows}</tbody>
                    </table>
                  </div>
                </section>
                """
            )
        html = """<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>Run Comparison</title>
  <style>{base_css}</style>
</head>
<body>
  <div class='container stack'>
    <header>
      <div class='row'>
        <h1 class='title'>Run Comparison</h1>
        <a class='btn' href='/'>Back to runs</a>
      </div>
    </header>
    {cards}
  </div>
</body>
</html>
        """.format(cards="".join(run_cards), base_css=base_css)
        return HTMLResponse(html)

    return app
