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
            self._create_results_table(con)
            # Backward-compat: ensure run_id column exists
            try:
                con.execute("ALTER TABLE results ADD COLUMN run_id TEXT")
            except Exception:
                pass
            try:
                con.execute(
                    "ALTER TABLE results ADD COLUMN evaluation_mode TEXT DEFAULT 'backtest'"
                )
            except Exception:
                pass
            try:
                con.execute("ALTER TABLE results ADD COLUMN mode_config_hash TEXT DEFAULT ''")
            except Exception:
                pass
            self._migrate_legacy_primary_key(con)
            con.commit()
        finally:
            con.close()

    @staticmethod
    def _create_results_table(con: sqlite3.Connection) -> None:
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
                evaluation_mode TEXT DEFAULT 'backtest',
                mode_config_hash TEXT DEFAULT '',
                engine_version TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(
                    collection,
                    symbol,
                    timeframe,
                    strategy,
                    params_json,
                    metric_name,
                    data_fingerprint,
                    fees,
                    slippage,
                    evaluation_mode,
                    mode_config_hash,
                    engine_version
                )
            )
            """
        )

    @staticmethod
    def _primary_key_columns(con: sqlite3.Connection, table: str) -> list[str]:
        rows = con.execute(f"PRAGMA table_info({table})").fetchall()
        pk_rows = sorted((int(row[5]), str(row[1])) for row in rows if int(row[5]) > 0)
        return [name for _, name in pk_rows]

    def _migrate_legacy_primary_key(self, con: sqlite3.Connection) -> None:
        pk_columns = self._primary_key_columns(con, "results")
        if "evaluation_mode" in pk_columns and "mode_config_hash" in pk_columns:
            return
        con.execute("ALTER TABLE results RENAME TO results_legacy")
        self._create_results_table(con)
        con.execute(
            """
            INSERT INTO results
            (
                collection,
                symbol,
                timeframe,
                strategy,
                params_json,
                metric_name,
                metric_value,
                stats_json,
                data_fingerprint,
                fees,
                slippage,
                run_id,
                evaluation_mode,
                mode_config_hash,
                engine_version,
                created_at
            )
            SELECT
                collection,
                symbol,
                timeframe,
                strategy,
                params_json,
                metric_name,
                metric_value,
                stats_json,
                data_fingerprint,
                fees,
                slippage,
                run_id,
                COALESCE(evaluation_mode, 'backtest'),
                COALESCE(mode_config_hash, ''),
                engine_version,
                created_at
            FROM results_legacy
            """
        )
        con.execute("DROP TABLE results_legacy")

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
        evaluation_mode: str = "backtest",
        mode_config_hash: str = "",
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
                  AND COALESCE(evaluation_mode, 'backtest')=?
                  AND COALESCE(mode_config_hash, '')=?
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
                    evaluation_mode,
                    mode_config_hash,
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
        evaluation_mode: str = "backtest",
        mode_config_hash: str = "",
    ) -> None:
        params_json = json.dumps(params, sort_keys=True)
        con = sqlite3.connect(self.db_path)
        try:
            con.execute(
                """
                INSERT OR REPLACE INTO results
                (
                    collection,
                    symbol,
                    timeframe,
                    strategy,
                    params_json,
                    metric_name,
                    metric_value,
                    stats_json,
                    data_fingerprint,
                    fees,
                    slippage,
                    run_id,
                    evaluation_mode,
                    mode_config_hash,
                    engine_version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    evaluation_mode,
                    mode_config_hash,
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
