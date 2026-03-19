from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ENGINE_VERSION = "1"


def _normalize_evaluation_mode(value: Any) -> str:
    mode = str(value or "backtest").strip().lower()
    return mode or "backtest"


@dataclass(frozen=True)
class ResultsCacheRecord:
    collection: str
    symbol: str
    timeframe: str
    strategy: str
    params: dict[str, Any]
    metric_name: str
    metric_value: float
    stats: dict[str, Any]
    data_fingerprint: str
    fees: float
    slippage: float
    run_id: str | None = None
    evaluation_mode: str = "backtest"
    mode_config_hash: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ResultsCacheRecord":
        required = (
            "collection",
            "symbol",
            "timeframe",
            "strategy",
            "params",
            "metric_name",
            "metric_value",
            "stats",
            "data_fingerprint",
            "fees",
            "slippage",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Missing cache record fields: {', '.join(sorted(missing))}")
        return cls(
            collection=str(payload["collection"]),
            symbol=str(payload["symbol"]),
            timeframe=str(payload["timeframe"]),
            strategy=str(payload["strategy"]),
            params=dict(payload["params"]),
            metric_name=str(payload["metric_name"]),
            metric_value=float(payload["metric_value"]),
            stats=dict(payload["stats"]),
            data_fingerprint=str(payload["data_fingerprint"]),
            fees=float(payload["fees"]),
            slippage=float(payload["slippage"]),
            run_id=str(payload["run_id"]) if payload.get("run_id") is not None else None,
            evaluation_mode=_normalize_evaluation_mode(payload.get("evaluation_mode", "backtest")),
            mode_config_hash=str(payload.get("mode_config_hash", "")),
        )


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
            existing_columns = self._get_results_table_columns(con)
            # Backward-compat: add columns only when missing (avoid exception-driven control flow).
            if "run_id" not in existing_columns:
                con.execute("ALTER TABLE results ADD COLUMN run_id TEXT")
            if "evaluation_mode" not in existing_columns:
                con.execute("ALTER TABLE results ADD COLUMN evaluation_mode TEXT DEFAULT 'backtest'")
            if "mode_config_hash" not in existing_columns:
                con.execute("ALTER TABLE results ADD COLUMN mode_config_hash TEXT DEFAULT ''")
            self._migrate_legacy_primary_key(con)
            con.commit()
        finally:
            con.close()

    @staticmethod
    def _get_results_table_columns(con: sqlite3.Connection) -> set[str]:
        cursor = con.execute("PRAGMA table_info(results)")
        return {str(row[1]) for row in cursor.fetchall()}

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
        normalized_mode = _normalize_evaluation_mode(evaluation_mode)
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
                    normalized_mode,
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
        record: ResultsCacheRecord | None = None,
        **payload: Any,
    ) -> None:
        if record is None:
            record = ResultsCacheRecord.from_mapping(payload)
        elif payload:
            raise ValueError("Pass either record or keyword payload to ResultsCache.set, not both")
        params_json = json.dumps(record.params, sort_keys=True)
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
                    record.collection,
                    record.symbol,
                    record.timeframe,
                    record.strategy,
                    params_json,
                    record.metric_name,
                    float(record.metric_value),
                    json.dumps(record.stats, sort_keys=True),
                    record.data_fingerprint,
                    record.fees,
                    record.slippage,
                    record.run_id,
                    record.evaluation_mode,
                    record.mode_config_hash,
                    ENGINE_VERSION,
                ),
            )
            con.commit()
        finally:
            con.close()

    def list_by_run(self, run_id: str) -> list[dict[str, Any]]:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            cur = con.execute(
                """
                SELECT collection, symbol, timeframe, strategy, params_json, metric_name, metric_value, stats_json
                FROM results WHERE run_id = ? ORDER BY collection, symbol, timeframe, strategy
                """,
                (run_id,),
            )
            rows = []
            for row in cur.fetchall():
                rows.append(
                    {
                        "collection": row["collection"],
                        "symbol": row["symbol"],
                        "timeframe": row["timeframe"],
                        "strategy": row["strategy"],
                        "params": json.loads(row["params_json"]),
                        "metric": row["metric_name"],
                        "metric_value": float(row["metric_value"]),
                        "stats": json.loads(row["stats_json"]),
                    }
                )
            return rows
        finally:
            con.close()
