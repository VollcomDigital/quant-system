from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

ENGINE_VERSION = "1"


class ResultsCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "results.sqlite"
        self._ensure()

    def _ensure(self):
        con = sqlite3.connect(self.db_path)
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS results (
                    collection TEXT,
                    symbol TEXT,
                    timeframe TEXT,
                    strategy TEXT,
                    params_json TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    stats_json TEXT,
                    data_fingerprint TEXT,
                    fees REAL,
                    slippage REAL,
                    run_id TEXT,
                    engine_version TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(collection, symbol, timeframe, strategy, params_json, metric_name, data_fingerprint, fees, slippage, engine_version)
                )
                """
            )
            # Backward-compat: ensure run_id column exists
            try:
                con.execute("ALTER TABLE results ADD COLUMN run_id TEXT")
            except Exception:
                pass
            con.commit()
        finally:
            con.close()

    def get(
        self,
        *,
        collection: str,
        symbol: str,
        timeframe: str,
        strategy: str,
        params: dict[str, Any],
        metric_name: str,
        data_fingerprint: str,
        fees: float,
        slippage: float,
        run_id: str | None = None,
    ) -> dict[str, Any] | None:
        params_json = json.dumps(params, sort_keys=True)
        con = sqlite3.connect(self.db_path)
        try:
            cur = con.execute(
                """
                SELECT metric_value, stats_json FROM results
                WHERE collection=? AND symbol=? AND timeframe=? AND strategy=?
                  AND params_json=? AND metric_name=? AND data_fingerprint=?
                  AND fees=? AND slippage=? AND engine_version=?
                """,
                (
                    collection,
                    symbol,
                    timeframe,
                    strategy,
                    params_json,
                    metric_name,
                    data_fingerprint,
                    fees,
                    slippage,
                    ENGINE_VERSION,
                ),
            )
            row = cur.fetchone()
            if not row:
                return None
            metric_value, stats_json = row
            return {
                "metric_value": float(metric_value),
                "stats": json.loads(stats_json),
            }
        finally:
            con.close()

    def set(
        self,
        *,
        collection: str,
        symbol: str,
        timeframe: str,
        strategy: str,
        params: dict[str, Any],
        metric_name: str,
        metric_value: float,
        stats: dict[str, Any],
        data_fingerprint: str,
        fees: float,
        slippage: float,
        run_id: str | None = None,
    ) -> None:
        params_json = json.dumps(params, sort_keys=True)
        con = sqlite3.connect(self.db_path)
        try:
            con.execute(
                """
                INSERT OR REPLACE INTO results
                (collection, symbol, timeframe, strategy, params_json, metric_name, metric_value, stats_json, data_fingerprint, fees, slippage, run_id, engine_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    collection,
                    symbol,
                    timeframe,
                    strategy,
                    params_json,
                    metric_name,
                    float(metric_value),
                    json.dumps(stats, sort_keys=True),
                    data_fingerprint,
                    fees,
                    slippage,
                    run_id,
                    ENGINE_VERSION,
                ),
            )
            con.commit()
        finally:
            con.close()

    def list_by_run(self, run_id: str) -> list[dict[str, Any]]:
        con = sqlite3.connect(self.db_path)
        try:
            cur = con.execute(
                """
                SELECT collection, symbol, timeframe, strategy, params_json, metric_name, metric_value, stats_json
                FROM results WHERE run_id = ? ORDER BY collection, symbol, timeframe, strategy
                """,
                (run_id,),
            )
            rows = []
            for r in cur.fetchall():
                (
                    collection,
                    symbol,
                    timeframe,
                    strategy,
                    params_json,
                    metric_name,
                    metric_value,
                    stats_json,
                ) = r
                rows.append(
                    {
                        "collection": collection,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "strategy": strategy,
                        "params": json.loads(params_json),
                        "metric": metric_name,
                        "metric_value": float(metric_value),
                        "stats": json.loads(stats_json),
                    }
                )
            return rows
        finally:
            con.close()
