from __future__ import annotations

from pathlib import Path
from typing import Any


class HealthReporter:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def export(self, failures: list[dict[str, Any]]):
        if not failures:
            # Still emit a tiny note
            (self.out_dir / "health.md").write_text("# Data Health\n\nNo data fetch failures.\n")
            return

        lines: list[str] = []
        lines.append("# Data Health")
        lines.append("")
        lines.append(f"Failures: {len(failures)}")
        lines.append("")
        lines.append("| Collection | Symbol | Timeframe | Source | Error |")
        lines.append("|---|---|---|---|---|")
        for f in failures:
            lines.append(
                f"| {f.get('collection', '')} | {f.get('symbol', '')} | {f.get('timeframe', '')} | {f.get('source', '')} | {str(f.get('error', '')).replace('|', ' ').strip()} |"
            )
        (self.out_dir / "health.md").write_text("\n".join(lines) + "\n")
