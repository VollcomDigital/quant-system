"""
Unified Cache Manager - Consolidates all caching functionality.
Supports data, backtest results, and optimization caching with intelligent management.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import pickle
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

# Optional Redis for recent overlay cache
try:
    import redis as _redis  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional
    _redis = None


@dataclass
class CacheEntry:
    """Cache entry metadata."""

    key: str
    cache_type: str  # 'data', 'backtest', 'optimization'
    symbol: str
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime | None
    size_bytes: int
    source: str | None = None
    interval: str | None = None
    data_type: str | None = None  # 'spot', 'futures', etc.
    parameters_hash: str | None = None
    version: str = "1.0"


class UnifiedCacheManager:
    """
    Unified cache manager that consolidates all caching functionality.
    Handles data caching, backtest results, and optimization results.
    """

    def __init__(self, cache_dir: str = "cache", max_size_gb: float = 10.0):
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.lock = threading.RLock()

        # Create directory structure
        self.data_dir = self.cache_dir / "data"
        self.backtest_dir = self.cache_dir / "backtests"
        self.optimization_dir = self.cache_dir / "optimizations"
        self.metadata_db = self.cache_dir / "cache.db"

        for dir_path in [self.data_dir, self.backtest_dir, self.optimization_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self._init_database()
        self.logger = logging.getLogger(__name__)

        # Optional Redis client for recent overlay layer
        self.redis_client = None
        try:
            use_redis = os.getenv("USE_REDIS_RECENT", "false").lower() == "true"
            redis_url = os.getenv("REDIS_URL", "")
            if use_redis and _redis is not None and redis_url:
                self.redis_client = _redis.from_url(redis_url, decode_responses=False)
                # ping to verify
                try:
                    self.redis_client.ping()
                    self.logger.info("Redis recent overlay enabled (%s)", redis_url)
                except Exception:
                    self.redis_client = None
        except Exception:
            self.redis_client = None

    def _init_database(self) -> None:
        """Initialize SQLite database for metadata."""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    cache_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    expires_at TEXT,
                    size_bytes INTEGER NOT NULL,
                    source TEXT,
                    interval TEXT,
                    data_type TEXT,
                    parameters_hash TEXT,
                    version TEXT DEFAULT '1.0',
                    file_path TEXT NOT NULL
                )
            """
            )

            # Create indexes for performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_cache_type ON cache_entries (cache_type)",
                "CREATE INDEX IF NOT EXISTS idx_symbol ON cache_entries (symbol)",
                "CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries (expires_at)",
                "CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries (last_accessed)",
                "CREATE INDEX IF NOT EXISTS idx_source ON cache_entries (source)",
                "CREATE INDEX IF NOT EXISTS idx_data_type ON cache_entries (data_type)",
            ]

            for index_sql in indexes:
                conn.execute(index_sql)

    def cache_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        interval: str = "1d",
        source: str | None = None,
        data_type: str | None = None,
        ttl_hours: int = 48,
    ) -> str:
        """
        Cache market data.

        Args:
            symbol: Symbol identifier
            data: DataFrame with OHLCV data
            interval: Data interval
            source: Data source name
            data_type: Data type ('spot', 'futures', etc.)
            ttl_hours: Time to live in hours

        Returns:
            Cache key
        """
        with self.lock:
            key = self._generate_key(
                "data",
                symbol=symbol,
                interval=interval,
                source=source,
                data_type=data_type,
            )

            file_path = self._get_file_path("data", key)
            compressed_data = self._compress_data(data)

            # Write compressed data
            file_path.write_bytes(compressed_data)

            # Create cache entry
            now = datetime.now()
            entry = CacheEntry(
                key=key,
                cache_type="data",
                symbol=symbol,
                created_at=now,
                last_accessed=now,
                expires_at=now + timedelta(hours=ttl_hours),
                size_bytes=len(compressed_data),
                source=source,
                interval=interval,
                data_type=data_type,
            )

            self._save_entry(entry, file_path)
            self._cleanup_if_needed()

            return key

    def get_data(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
        source: str | None = None,
        data_type: str | None = None,
    ) -> pd.DataFrame | None:
        """
        Retrieve cached market data.

        Args:
            symbol: Symbol identifier
            start_date: Optional start date filter
            end_date: Optional end date filter
            interval: Data interval
            source: Optional source filter
            data_type: Optional data type filter

        Returns:
            DataFrame or None if not found/expired
        """
        with self.lock:
            # Find matching cache entries
            entries = self._find_entries(
                "data",
                symbol=symbol,
                interval=interval,
                source=source,
                data_type=data_type,
            )

            if not entries:
                return None

            # Get the most recent non-expired entry
            valid_entries = [e for e in entries if not self._is_expired(e)]
            if not valid_entries:
                # Clean up expired entries
                for entry in entries:
                    self._remove_entry(entry.key)
                return None

            # Sort by creation date (most recent first)
            valid_entries.sort(key=lambda x: x.created_at, reverse=True)
            entry = valid_entries[0]

            # Load and decompress data
            file_path = self._get_file_path("data", entry.key)
            if not file_path.exists():
                self._remove_entry(entry.key)
                return None

            try:
                compressed_data = file_path.read_bytes()
                data = self._decompress_data(compressed_data)

                # Update access time
                self._update_access_time(entry.key)

                # Filter by date range if specified
                if start_date or end_date:
                    if start_date:
                        start = pd.to_datetime(start_date, utc=True)
                        # If data index is timezone-aware, ensure comparison consistency
                        if hasattr(data.index, "tz") and data.index.tz is not None:
                            if start.tz is None:
                                start = start.tz_localize("UTC")
                        else:
                            # If data index is timezone-naive but start is aware, make start naive
                            if start.tz is not None:
                                start = start.tz_localize(None)
                        data = data[data.index >= start]
                    if end_date:
                        end = pd.to_datetime(end_date, utc=True)
                        # If data index is timezone-aware, ensure comparison consistency
                        if hasattr(data.index, "tz") and data.index.tz is not None:
                            if end.tz is None:
                                end = end.tz_localize("UTC")
                        else:
                            # If data index is timezone-naive but end is aware, make end naive
                            if end.tz is not None:
                                end = end.tz_localize(None)
                        data = data[data.index <= end]

                return data if not data.empty else None

            except Exception as e:
                self.logger.warning("Failed to load cached data for %s: %s", symbol, e)
                self._remove_entry(entry.key)
                return None

    def cache_backtest_result(
        self,
        symbol: str,
        strategy: str,
        parameters: dict[str, Any],
        result: dict[str, Any],
        interval: str = "1d",
        ttl_days: int = 30,
    ) -> str:
        """Cache backtest result."""
        with self.lock:
            params_hash = self._hash_parameters(parameters)
            key = self._generate_key(
                "backtest",
                symbol=symbol,
                strategy=strategy,
                parameters_hash=params_hash,
                interval=interval,
            )

            file_path = self._get_file_path("backtest", key)

            # Add metadata to result
            result_with_meta = {
                "result": result,
                "symbol": symbol,
                "strategy": strategy,
                "parameters": parameters,
                "interval": interval,
                "cached_at": datetime.now().isoformat(),
            }

            compressed_data = self._compress_data(result_with_meta)
            file_path.write_bytes(compressed_data)

            # Create cache entry
            now = datetime.now()
            entry = CacheEntry(
                key=key,
                cache_type="backtest",
                symbol=symbol,
                created_at=now,
                last_accessed=now,
                expires_at=now + timedelta(days=ttl_days),
                size_bytes=len(compressed_data),
                interval=interval,
                parameters_hash=params_hash,
            )

            self._save_entry(entry, file_path)
            self._cleanup_if_needed()

            return key

    def get_backtest_result(
        self,
        symbol: str,
        strategy: str,
        parameters: dict[str, Any],
        interval: str = "1d",
    ) -> dict[str, Any] | None:
        """Retrieve cached backtest result."""
        with self.lock:
            params_hash = self._hash_parameters(parameters)
            entries = self._find_entries(
                "backtest",
                symbol=symbol,
                parameters_hash=params_hash,
                interval=interval,
            )

            if not entries:
                return None

            # Get the most recent non-expired entry
            valid_entries = [e for e in entries if not self._is_expired(e)]
            if not valid_entries:
                for entry in entries:
                    self._remove_entry(entry.key)
                return None

            entry = valid_entries[0]
            file_path = self._get_file_path("backtest", entry.key)

            if not file_path.exists():
                self._remove_entry(entry.key)
                return None

            try:
                compressed_data = file_path.read_bytes()
                cached_data = self._decompress_data(compressed_data)

                self._update_access_time(entry.key)
                result = cached_data.get("result")
                return result if result is not None else {}

            except Exception as e:
                self.logger.warning("Failed to load cached backtest: %s", e)
                self._remove_entry(entry.key)
                return None

    def cache_optimization_result(
        self,
        symbol: str,
        strategy: str,
        optimization_config: dict[str, Any],
        result: dict[str, Any],
        interval: str = "1d",
        ttl_days: int = 60,
    ) -> str:
        """Cache optimization result."""
        with self.lock:
            config_hash = self._hash_parameters(optimization_config)
            key = self._generate_key(
                "optimization",
                symbol=symbol,
                strategy=strategy,
                parameters_hash=config_hash,
                interval=interval,
            )

            file_path = self._get_file_path("optimization", key)

            result_with_meta = {
                "result": result,
                "symbol": symbol,
                "strategy": strategy,
                "optimization_config": optimization_config,
                "interval": interval,
                "cached_at": datetime.now().isoformat(),
            }

            compressed_data = self._compress_data(result_with_meta)
            file_path.write_bytes(compressed_data)

            # Create cache entry
            now = datetime.now()
            entry = CacheEntry(
                key=key,
                cache_type="optimization",
                symbol=symbol,
                created_at=now,
                last_accessed=now,
                expires_at=now + timedelta(days=ttl_days),
                size_bytes=len(compressed_data),
                interval=interval,
                parameters_hash=config_hash,
            )

            self._save_entry(entry, file_path)
            self._cleanup_if_needed()

            return key

    def get_optimization_result(
        self,
        symbol: str,
        strategy: str,
        optimization_config: dict[str, Any],
        interval: str = "1d",
    ) -> dict[str, Any] | None:
        """Retrieve cached optimization result."""
        with self.lock:
            config_hash = self._hash_parameters(optimization_config)
            entries = self._find_entries(
                "optimization",
                symbol=symbol,
                parameters_hash=config_hash,
                interval=interval,
            )

            if not entries:
                return None

            valid_entries = [e for e in entries if not self._is_expired(e)]
            if not valid_entries:
                for entry in entries:
                    self._remove_entry(entry.key)
                return None

            entry = valid_entries[0]
            file_path = self._get_file_path("optimization", entry.key)

            if not file_path.exists():
                self._remove_entry(entry.key)
                return None

            try:
                compressed_data = file_path.read_bytes()
                cached_data = self._decompress_data(compressed_data)

                self._update_access_time(entry.key)
                result = cached_data.get("result")
                return result if result is not None else {}

            except Exception as e:
                self.logger.warning("Failed to load cached optimization: %s", e)
                self._remove_entry(entry.key)
                return None

    def clear_cache(
        self,
        cache_type: str | None = None,
        symbol: str | None = None,
        source: str | None = None,
        older_than_days: int | None = None,
    ) -> None:
        """Clear cache entries based on filters."""
        with self.lock:
            conditions = []
            params = []

            if cache_type:
                conditions.append("cache_type = ?")
                params.append(cache_type)

            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)

            if source:
                conditions.append("source = ?")
                params.append(source)

            if older_than_days:
                cutoff = (datetime.now() - timedelta(days=older_than_days)).isoformat()
                conditions.append("created_at < ?")
                params.append(cutoff)

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            with sqlite3.connect(self.metadata_db) as conn:
                # Use parameterized query to prevent SQL injection
                if conditions:
                    query = f"SELECT key, cache_type FROM cache_entries WHERE {where_clause}"  # nosec B608
                else:
                    query = "SELECT key, cache_type FROM cache_entries"
                cursor = conn.execute(query, params)

                entries_to_remove = cursor.fetchall()

                # Remove files
                for key, ct in entries_to_remove:
                    file_path = self._get_file_path(ct, key)
                    if file_path.exists():
                        file_path.unlink()

                # Remove metadata
                if conditions:
                    delete_query = f"DELETE FROM cache_entries WHERE {where_clause}"  # nosec B608
                else:
                    delete_query = "DELETE FROM cache_entries"
                conn.execute(delete_query, params)

            self.logger.info("Cleared %s cache entries", len(entries_to_remove))

    def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        with sqlite3.connect(self.metadata_db) as conn:
            # Overall stats
            cursor = conn.execute(
                """
                SELECT
                    cache_type,
                    COUNT(*) as count,
                    SUM(size_bytes) as total_size,
                    AVG(size_bytes) as avg_size,
                    MIN(created_at) as oldest,
                    MAX(created_at) as newest
                FROM cache_entries
                GROUP BY cache_type
            """
            )

            stats_by_type = {}
            total_size = 0

            for row in cursor:
                cache_type, count, size_sum, avg_size, oldest, newest = row
                size_sum = size_sum or 0
                total_size += size_sum

                stats_by_type[cache_type] = {
                    "count": count,
                    "total_size_bytes": size_sum,
                    "total_size_mb": size_sum / 1024**2,
                    "avg_size_bytes": avg_size or 0,
                    "oldest": oldest,
                    "newest": newest,
                }

            # Source distribution for data cache
            cursor = conn.execute(
                """
                SELECT source, COUNT(*), SUM(size_bytes)
                FROM cache_entries
                WHERE cache_type = 'data' AND source IS NOT NULL
                GROUP BY source
            """
            )

            source_stats = {}
            for source, count, size_sum in cursor:
                source_stats[source] = {"count": count, "size_bytes": size_sum or 0}

            return {
                "total_size_bytes": total_size,
                "total_size_mb": total_size / 1024**2,
                "total_size_gb": total_size / 1024**3,
                "max_size_gb": self.max_size_bytes / 1024**3,
                "utilization_percent": (total_size / self.max_size_bytes) * 100,
                "by_type": stats_by_type,
                "by_source": source_stats,
            }

    def _generate_key(self, cache_type: str, **kwargs: Any) -> str:
        """Generate unique cache key."""
        key_parts = [cache_type]
        for k, v in sorted(kwargs.items()):
            if v is not None:
                key_parts.append(f"{k}={v}")

        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _get_file_path(self, cache_type: str, key: str) -> Path:
        """Get file path for cache entry."""
        if cache_type == "data":
            return self.data_dir / f"{key}.gz"
        if cache_type == "backtest":
            return self.backtest_dir / f"{key}.gz"
        if cache_type == "optimization":
            return self.optimization_dir / f"{key}.gz"
        msg = f"Unknown cache type: {cache_type}"
        raise ValueError(msg)

    def _compress_data(self, data: Any) -> bytes:
        """Compress data using gzip."""
        serialized = pickle.dumps(data)

        return gzip.compress(serialized)

    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress data."""
        decompressed = gzip.decompress(compressed_data)
        # Note: pickle.loads() can be unsafe with untrusted data
        # In production, consider using safer serialization formats
        return pickle.loads(decompressed)  # nosec B301

    # -------- Optional Redis recent overlay helpers ---------
    def _redis_recent_key(self, symbol: str, interval: str) -> str:
        return f"data:recent:{symbol}:{interval}"

    def get_recent_overlay_from_redis(
        self, symbol: str, interval: str
    ) -> pd.DataFrame | None:
        try:
            if not self.redis_client:
                return None
            key = self._redis_recent_key(symbol, interval)
            blob = self.redis_client.get(key)
            if not blob:
                return None
            data = self._decompress_data(blob)
            if isinstance(data, pd.DataFrame) and not data.empty:
                return data
            return None
        except Exception:
            return None

    def set_recent_overlay_to_redis(
        self, symbol: str, interval: str, df: pd.DataFrame, ttl_hours: int = 24
    ) -> None:
        try:
            if not self.redis_client or df is None or df.empty:
                return
            key = self._redis_recent_key(symbol, interval)
            blob = self._compress_data(df)
            self.redis_client.setex(key, int(ttl_hours * 3600), blob)
        except Exception:
            return

    def _hash_parameters(self, parameters: dict[str, Any]) -> str:
        """Generate hash for parameters."""
        params_str = json.dumps(parameters, sort_keys=True)
        return hashlib.sha256(params_str.encode()).hexdigest()[:16]

    def _save_entry(self, entry: CacheEntry, file_path: Path) -> None:
        """Save cache entry metadata."""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache_entries
                (key, cache_type, symbol, created_at, last_accessed, expires_at,
                 size_bytes, source, interval, data_type, parameters_hash, version, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    entry.key,
                    entry.cache_type,
                    entry.symbol,
                    entry.created_at.isoformat(),
                    entry.last_accessed.isoformat(),
                    entry.expires_at.isoformat() if entry.expires_at else None,
                    entry.size_bytes,
                    entry.source,
                    entry.interval,
                    entry.data_type,
                    entry.parameters_hash,
                    entry.version,
                    str(file_path),
                ),
            )

    def _find_entries(self, cache_type: str, **filters: Any) -> list[CacheEntry]:
        """Find cache entries matching filters."""
        conditions = ["cache_type = ?"]
        params = [cache_type]

        for key, value in filters.items():
            if value is not None:
                conditions.append(f"{key} = ?")
                params.append(value)

        where_clause = " AND ".join(conditions)

        with sqlite3.connect(self.metadata_db) as conn:
            # Use parameterized query to prevent SQL injection
            query = f"SELECT * FROM cache_entries WHERE {where_clause}"  # nosec B608
            cursor = conn.execute(query, params)

            entries = []
            for row in cursor:
                entry = CacheEntry(
                    key=row[0],
                    cache_type=row[1],
                    symbol=row[2],
                    created_at=datetime.fromisoformat(row[3]).replace(tzinfo=None),
                    last_accessed=datetime.fromisoformat(row[4]).replace(tzinfo=None),
                    expires_at=datetime.fromisoformat(row[5]).replace(tzinfo=None)
                    if row[5]
                    else None,
                    size_bytes=row[6],
                    source=row[7],
                    interval=row[8],
                    data_type=row[9],
                    parameters_hash=row[10],
                    version=row[11],
                )
                entries.append(entry)

            return entries

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        if not entry.expires_at:
            return False

        from datetime import timezone

        # Always use UTC for consistent comparison
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        expires_at = entry.expires_at

        return now > expires_at

    def _update_access_time(self, key: str) -> None:
        """Update last access time."""
        from datetime import timezone

        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute(
                "UPDATE cache_entries SET last_accessed = ? WHERE key = ?",
                (datetime.now(timezone.utc).replace(tzinfo=None).isoformat(), key),
            )

    def _remove_entry(self, key: str) -> None:
        """Remove cache entry and its file."""
        with sqlite3.connect(self.metadata_db) as conn:
            cursor = conn.execute(
                "SELECT cache_type FROM cache_entries WHERE key = ?", (key,)
            )
            row = cursor.fetchone()

            if row:
                cache_type = row[0]
                file_path = self._get_file_path(cache_type, key)
                if file_path.exists():
                    file_path.unlink()

                conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))

    def _cleanup_if_needed(self) -> None:
        """Clean up cache if size exceeds limit."""
        stats = self.get_cache_stats()
        total_size = stats["total_size_bytes"]

        if total_size > self.max_size_bytes:
            self.logger.info(
                "Cache size (%.2f GB) exceeds limit, cleaning up...",
                total_size / 1024**3,
            )

            # Remove expired entries first
            self._cleanup_expired()

            # If still over limit, remove LRU entries
            stats = self.get_cache_stats()
            if stats["total_size_bytes"] > self.max_size_bytes:
                self._cleanup_lru()

    def _cleanup_expired(self) -> None:
        """Remove expired cache entries."""
        now = datetime.now().isoformat()

        with sqlite3.connect(self.metadata_db) as conn:
            cursor = conn.execute(
                "SELECT key, cache_type FROM cache_entries WHERE expires_at < ?", (now,)
            )

            expired_entries = cursor.fetchall()

            for key, cache_type in expired_entries:
                file_path = self._get_file_path(cache_type, key)
                if file_path.exists():
                    file_path.unlink()

            conn.execute("DELETE FROM cache_entries WHERE expires_at < ?", (now,))

        self.logger.info("Removed %s expired cache entries", len(expired_entries))

    def _cleanup_lru(self) -> None:
        """Remove least recently used entries."""
        target_size = int(self.max_size_bytes * 0.8)  # Clean to 80% of limit

        with sqlite3.connect(self.metadata_db) as conn:
            cursor = conn.execute(
                """
                SELECT key, cache_type, size_bytes
                FROM cache_entries
                ORDER BY last_accessed ASC
            """
            )

            current_size = self.get_cache_stats()["total_size_bytes"]
            removed_count = 0

            for key, cache_type, size_bytes in cursor:
                if current_size <= target_size:
                    break

                file_path = self._get_file_path(cache_type, key)
                if file_path.exists():
                    file_path.unlink()

                conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                current_size -= size_bytes
                removed_count += 1

        self.logger.info("Removed %s LRU cache entries", removed_count)
