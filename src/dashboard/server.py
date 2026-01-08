from __future__ import annotations

import html
import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from ..reporting.dashboard import DOWNLOAD_FILE_CANDIDATES

RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def create_app(reports_dir: Path) -> FastAPI:
    # Resolve the reports directory to an absolute path to use as a trusted root
    root = Path(reports_dir).resolve()

    def _escape_html(value: Any) -> str:
        return html.escape(str(value), quote=True)

    def _validate_run_id(run_id: str) -> str:
        if not RUN_ID_RE.fullmatch(run_id):
            raise HTTPException(status_code=400, detail="Invalid run id")
        return run_id

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not root.exists():
            raise RuntimeError(f"Reports directory not found: {root}")
        yield

    app = FastAPI(title="Quant System Dashboard", lifespan=lifespan)

    def _runs() -> list[Path]:
        if not root.exists():
            return []
        return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)

    def _load_summary(run_dir: Path) -> dict[str, Any]:
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            return {}
        try:
            return json.loads(summary_path.read_text())
        except Exception:
            return {}

    def _load_summary_csv(run_dir: Path, limit: int = 5) -> list[dict[str, Any]]:
        csv_path = run_dir / "summary.csv"
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
        run_id = _validate_run_id(run_id)
        run_dir = root / run_id
        if not run_dir.exists() or not run_dir.is_dir():
            raise HTTPException(status_code=404, detail="Run not found")
        summary = _load_summary(run_dir)
        top_rows = _load_summary_csv(run_dir)
        return {
            "run_id": run_id,
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
                    "run_id": run_dir.name,
                    "metric": summary.get("metric"),
                    "results_count": summary.get("results_count"),
                    "started_at": summary.get("started_at"),
                    "finished_at": summary.get("finished_at"),
                    "report_path": str((run_dir / "report.html").resolve()),
                }
            )
        html_rows = "".join(
            f"<tr><td class='px-4 py-2 font-semibold'>{_escape_html(r['run_id'])}</td>"
            f"<td class='px-4 py-2'>{_escape_html(r.get('metric', ''))}</td>"
            f"<td class='px-4 py-2'>{_escape_html(r.get('results_count', ''))}</td>"
            f"<td class='px-4 py-2'>{_escape_html(r.get('started_at', ''))}</td>"
            f"<td class='px-4 py-2'><a class='text-sky-400 underline' href='file://{_escape_html(r['report_path'])}'>Open report</a></td></tr>"
            for r in rows
        )
        html = f"""
<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Quant System Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class='bg-slate-900 text-slate-100'>
  <div class='max-w-5xl mx-auto py-8 px-4 space-y-6'>
    <header>
      <h1 class='text-2xl font-bold'>Quant System Runs</h1>
      <p class='text-sm text-slate-400'>Browse recent runs, review summaries, and open detailed reports.</p>
    </header>
    <section class='bg-slate-800 rounded-lg shadow'>
      <table class='w-full text-sm'>
        <thead class='text-xs uppercase bg-slate-700 text-slate-300'>
          <tr>
            <th class='px-4 py-2 text-left'>Run ID</th>
            <th class='px-4 py-2 text-left'>Metric</th>
            <th class='px-4 py-2 text-left'>Results</th>
            <th class='px-4 py-2 text-left'>Started</th>
            <th class='px-4 py-2 text-left'>Detail</th>
            <th class='px-4 py-2 text-left'>Report</th>
          </tr>
        </thead>
        <tbody>
          {html_rows if html_rows else "<tr><td class='px-4 py-3 text-slate-400' colspan='6'>No runs available</td></tr>"}
        </tbody>
      </table>
    </section>
  </div>
</body>
</html>
        """
        return HTMLResponse(html)

    @app.get("/run/{run_id}", response_class=HTMLResponse)
    async def run_page(run_id: str) -> HTMLResponse:
        run_id = _validate_run_id(run_id)
        run_dir = (root / run_id).resolve()
        try:
            # Ensure the resolved run directory is within the trusted root
            run_dir.relative_to(root)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid run directory")
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
            f"<tr><td class='px-3 py-2 font-semibold'>{_escape_html(k)}</td><td class='px-3 py-2'>{_escape_html(v)}</td></tr>"
            for k, v in summary.items()
            if k in {"metric", "results_count", "started_at", "finished_at", "duration_sec"}
        )

        manifest_rows = (
            "".join(
                f"<tr><td class='px-3 py-2'>{_escape_html(m.get('run_id'))}</td>"
                f"<td class='px-3 py-2'>{_escape_html(m.get('status'))}</td>"
                f"<td class='px-3 py-2'>{_escape_html(m.get('message', ''))}</td></tr>"
                for m in manifest
            )
            or "<tr><td class='px-3 py-2 text-slate-400' colspan='3'>No manifest actions</td></tr>"
        )

        notification_rows = (
            "".join(
                f"<tr><td class='px-3 py-2'>{_escape_html(n.get('channel'))}</td>"
                f"<td class='px-3 py-2'>{_escape_html(n.get('metric'))}</td>"
                f"<td class='px-3 py-2'>{_escape_html('sent' if n.get('sent') else n.get('reason', 'skipped'))}</td></tr>"
                for n in notifications
            )
            or "<tr><td class='px-3 py-2 text-slate-400' colspan='3'>No notifications</td></tr>"
        )

        downloads = []
        for name in DOWNLOAD_FILE_CANDIDATES:
            if (run_dir / name).exists():
                downloads.append(
                    f"<li><a class='text-sky-400 underline' href='/api/runs/{_escape_html(run_id)}/files/{_escape_html(name)}'>{_escape_html(name)}</a></li>"
                )
        downloads_html = "".join(downloads) or "<li class='text-slate-400'>No files</li>"

        html = f"""
<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Run {_escape_html(run_id)}</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class='bg-slate-900 text-slate-100'>
  <div class='max-w-5xl mx-auto py-8 px-4 space-y-6'>
    <header>
      <h1 class='text-2xl font-bold'>Run {_escape_html(run_id)}</h1>
      <a class='text-sky-400 underline text-sm' href='/'>Back to runs</a>
    </header>
    <section class='bg-slate-800 rounded-lg shadow'>
      <h2 class='px-4 pt-4 text-lg font-semibold'>Summary</h2>
      <table class='w-full text-sm'>
        <tbody>{summary_rows}</tbody>
      </table>
      <div class='px-4 pb-4'>
        <h3 class='text-sm font-semibold mt-4 mb-2'>Downloads</h3>
        <ul class='list-disc list-inside text-sm'>
          {downloads_html}
        </ul>
      </div>
    </section>
    <section class='bg-slate-800 rounded-lg shadow'>
      <h2 class='px-4 pt-4 text-lg font-semibold'>Manifest Actions</h2>
      <table class='w-full text-sm'>
        <thead class='text-xs uppercase bg-slate-700 text-slate-300'>
          <tr><th class='px-3 py-2 text-left'>Run</th><th class='px-3 py-2 text-left'>Status</th><th class='px-3 py-2 text-left'>Message</th></tr>
        </thead>
        <tbody>{manifest_rows}</tbody>
      </table>
    </section>
    <section class='bg-slate-800 rounded-lg shadow'>
      <h2 class='px-4 pt-4 text-lg font-semibold'>Notifications</h2>
      <table class='w-full text-sm'>
        <thead class='text-xs uppercase bg-slate-700 text-slate-300'>
          <tr><th class='px-3 py-2 text-left'>Channel</th><th class='px-3 py-2 text-left'>Metric</th><th class='px-3 py-2 text-left'>Outcome</th></tr>
        </thead>
        <tbody>{notification_rows}</tbody>
      </table>
    </section>
  </div>
</body>
</html>
        """
        return HTMLResponse(html)

    @app.get("/api/runs/{run_id}/files/{filename}")
    async def download(run_id: str, filename: str):
        run_id = _validate_run_id(run_id)
        run_dir = root / run_id
        if not run_dir.exists() or not run_dir.is_dir():
            raise HTTPException(status_code=404, detail="Run not found")
        allowed = set(DOWNLOAD_FILE_CANDIDATES)
        if filename not in allowed:
            raise HTTPException(status_code=404, detail="File not available")
        file_path = run_dir / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(file_path)

    @app.get("/api/compare")
    async def compare_runs(runs: str) -> dict[str, Any]:
        run_ids = [r.strip() for r in runs.split(",") if r.strip()]
        if not run_ids:
            raise HTTPException(status_code=400, detail="Provide at least one run id")
        payload = {}
        for run_id in run_ids:
            if not RUN_ID_RE.fullmatch(run_id):
                continue
            candidate_dir = root / run_id
            try:
                run_dir = candidate_dir.resolve()
            except FileNotFoundError:
                # If the path cannot be resolved, treat it as nonexistent
                continue
            # Ensure the resolved run directory is contained within the reports root
            try:
                # This will raise ValueError if run_dir is not within root
                _ = run_dir.relative_to(root)
            except ValueError:
                # Attempt to escape the reports root; skip this run_id
                continue
            # At this point, run_dir is a resolved, validated path under root.
            safe_run_dir = run_dir
            if not safe_run_dir.exists() or not safe_run_dir.is_dir():
                continue
            payload[run_id] = {
                "summary": _load_summary(safe_run_dir),
                "top": _load_summary_csv(safe_run_dir, limit=10),
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
                    f"<tr><td class='px-3 py-2'>{_escape_html(row.get('symbol'))}</td>"
                    f"<td class='px-3 py-2'>{_escape_html(row.get('strategy'))}</td>"
                    f"<td class='px-3 py-2'>{_escape_html(row.get('metric'))}</td>"
                    f"<td class='px-3 py-2'>{_escape_html(row.get('metric_value'))}</td></tr>"
                    for row in rows
                )
                or "<tr><td class='px-3 py-2 text-slate-400' colspan='4'>No summary.csv data</td></tr>"
            )
            summary_meta = data.get("summary", {})
            run_cards.append(
                f"""
                <section class='bg-slate-800 rounded-lg shadow p-4 space-y-3'>
                  <header>
                    <h2 class='text-lg font-semibold'>{_escape_html(run_id)}</h2>
                    <p class='text-xs text-slate-400'>Metric: {_escape_html(summary_meta.get("metric", ""))}</p>
                  </header>
                  <div class='overflow-x-auto'>
                    <table class='min-w-full text-sm text-left text-slate-200'>
                      <thead class='text-xs uppercase bg-slate-700 text-slate-300'>
                        <tr><th class='px-3 py-2'>Symbol</th><th class='px-3 py-2'>Strategy</th><th class='px-3 py-2'>Metric</th><th class='px-3 py-2'>Value</th></tr>
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
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class='bg-slate-900 text-slate-100'>
  <div class='max-w-6xl mx-auto py-8 px-4 space-y-4'>
    <header>
      <h1 class='text-2xl font-bold'>Run Comparison</h1>
      <a class='text-sky-400 underline text-sm' href='/'>Back to runs</a>
    </header>
    {cards}
  </div>
</body>
</html>
        """.format(cards="".join(run_cards))
        return HTMLResponse(html)

    return app
