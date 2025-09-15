from __future__ import annotations

from pathlib import Path

import pandas as pd


class ParquetCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, source: str, symbol: str, timeframe: str) -> Path:
        sym = symbol.replace("/", "-")
        return self.root / source / f"{sym}_{timeframe}.parquet"

    def load(self, source: str, symbol: str, timeframe: str) -> pd.DataFrame | None:
        p = self._path(source, symbol, timeframe)
        if not p.exists():
            return None
        try:
            df = pd.read_parquet(p)
            return df
        except Exception:
            return None

    def save(self, source: str, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        p = self._path(source, symbol, timeframe)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, compression="zstd")
