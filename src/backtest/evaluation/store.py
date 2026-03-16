from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import EvaluationModeConfig, ResultRecord, EvaluationCacheRecord

EVALUATION_SCHEMA_VERSION = "1"


class EvaluationCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "evaluation_cache.sqlite"
        self._ensure()

    def _ensure(self) -> None:
        con = sqlite3.connect(self.db_path)
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluation_cache (
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
                    evaluation_mode TEXT,
                    mode_config_hash TEXT,
                    validation_config_hash TEXT,
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
                        validation_config_hash,
                        engine_version
                    )
                )
                """
            )
            con.commit()
        finally:
            con.close()

    @staticmethod
    def hash_mode_config(mode_config: EvaluationModeConfig) -> str:
        payload = {
            "mode": mode_config.mode,
            "payload": mode_config.payload,
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

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
        evaluation_mode: str,
        mode_config_hash: str,
        validation_config_hash: str,
    ) -> dict[str, Any] | None:
        params_json = json.dumps(params, sort_keys=True)
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            cur = con.execute(
                """
                SELECT metric_value, stats_json
                FROM evaluation_cache
                WHERE collection=? AND symbol=? AND timeframe=? AND strategy=?
                  AND params_json=? AND metric_name=? AND data_fingerprint=?
                  AND fees=? AND slippage=?
                  AND evaluation_mode=? AND mode_config_hash=? AND validation_config_hash=?
                  AND engine_version=?
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
                    evaluation_mode,
                    mode_config_hash,
                    validation_config_hash,
                    EVALUATION_SCHEMA_VERSION,
                ),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {
                "metric_value": float(row["metric_value"]),
                "stats": json.loads(row["stats_json"]),
            }
        finally:
            con.close()

    def set(self, *, record: EvaluationCacheRecord) -> None:
        params_json = json.dumps(record.params, sort_keys=True)
        con = sqlite3.connect(self.db_path)
        try:
            con.execute(
                """
                INSERT OR REPLACE INTO evaluation_cache
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
                    evaluation_mode,
                    mode_config_hash,
                    validation_config_hash,
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
                    record.evaluation_mode,
                    record.mode_config_hash,
                    record.validation_config_hash,
                    EVALUATION_SCHEMA_VERSION,
                ),
            )
            con.commit()
        finally:
            con.close()


class ResultStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "result_store.sqlite"
        self._ensure()

    def _ensure(self) -> None:
        con = sqlite3.connect(self.db_path)
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS result_records (
                    run_id TEXT,
                    evaluation_mode TEXT,
                    collection TEXT,
                    symbol TEXT,
                    timeframe TEXT,
                    source TEXT,
                    strategy TEXT,
                    params_json TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    stats_json TEXT,
                    data_fingerprint TEXT,
                    fees REAL,
                    slippage REAL,
                    mode_config_hash TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            con.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_result_records_run
                ON result_records(run_id, collection, symbol, timeframe, strategy)
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS run_metadata (
                    run_id TEXT PRIMARY KEY,
                    evaluation_mode TEXT,
                    mode_config_hash TEXT,
                    validation_profile_json TEXT,
                    active_gates_json TEXT,
                    inactive_gates_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            con.commit()
        finally:
            con.close()

    def insert(self, record: ResultRecord) -> None:
        con = sqlite3.connect(self.db_path)
        try:
            con.execute(
                """
                INSERT INTO result_records
                (
                    run_id,
                    evaluation_mode,
                    collection,
                    symbol,
                    timeframe,
                    source,
                    strategy,
                    params_json,
                    metric_name,
                    metric_value,
                    stats_json,
                    data_fingerprint,
                    fees,
                    slippage,
                    mode_config_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.evaluation_mode,
                    record.collection,
                    record.symbol,
                    record.timeframe,
                    record.source,
                    record.strategy,
                    json.dumps(record.params, sort_keys=True),
                    record.metric_name,
                    float(record.metric_value),
                    json.dumps(record.stats, sort_keys=True),
                    record.data_fingerprint,
                    record.fees,
                    record.slippage,
                    record.mode_config_hash,
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
                SELECT
                    run_id,
                    evaluation_mode,
                    collection,
                    symbol,
                    timeframe,
                    source,
                    strategy,
                    params_json,
                    metric_name,
                    metric_value,
                    stats_json,
                    data_fingerprint,
                    fees,
                    slippage,
                    mode_config_hash
                FROM result_records
                WHERE run_id = ?
                ORDER BY collection, symbol, timeframe, strategy
                """,
                (run_id,),
            )
            return [
                {
                    "run_id": row["run_id"],
                    "evaluation_mode": row["evaluation_mode"],
                    "collection": row["collection"],
                    "symbol": row["symbol"],
                    "timeframe": row["timeframe"],
                    "source": row["source"],
                    "strategy": row["strategy"],
                    "params": json.loads(row["params_json"]),
                    "metric": row["metric_name"],
                    "metric_value": float(row["metric_value"]),
                    "stats": json.loads(row["stats_json"]),
                    "data_fingerprint": row["data_fingerprint"],
                    "fees": float(row["fees"]),
                    "slippage": float(row["slippage"]),
                    "mode_config_hash": row["mode_config_hash"],
                }
                for row in cur.fetchall()
            ]
        finally:
            con.close()

    def upsert_run_metadata(
        self,
        *,
        run_id: str,
        evaluation_mode: str,
        mode_config_hash: str,
        validation_profile: dict[str, Any],
        active_gates: list[str],
        inactive_gates: list[str],
    ) -> None:
        con = sqlite3.connect(self.db_path)
        try:
            con.execute(
                """
                INSERT INTO run_metadata
                (
                    run_id,
                    evaluation_mode,
                    mode_config_hash,
                    validation_profile_json,
                    active_gates_json,
                    inactive_gates_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    evaluation_mode=excluded.evaluation_mode,
                    mode_config_hash=excluded.mode_config_hash,
                    validation_profile_json=excluded.validation_profile_json,
                    active_gates_json=excluded.active_gates_json,
                    inactive_gates_json=excluded.inactive_gates_json
                """,
                (
                    run_id,
                    evaluation_mode,
                    mode_config_hash,
                    json.dumps(validation_profile, sort_keys=True),
                    json.dumps(active_gates, sort_keys=True),
                    json.dumps(inactive_gates, sort_keys=True),
                ),
            )
            con.commit()
        finally:
            con.close()

    def get_run_metadata(self, run_id: str) -> dict[str, Any] | None:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            row = con.execute(
                """
                SELECT
                    run_id,
                    evaluation_mode,
                    mode_config_hash,
                    validation_profile_json,
                    active_gates_json,
                    inactive_gates_json
                FROM run_metadata
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "run_id": row["run_id"],
                "evaluation_mode": row["evaluation_mode"],
                "mode_config_hash": row["mode_config_hash"],
                "validation_profile": json.loads(row["validation_profile_json"]),
                "active_gates": list(json.loads(row["active_gates_json"])),
                "inactive_gates": list(json.loads(row["inactive_gates_json"])),
            }
        finally:
            con.close()
