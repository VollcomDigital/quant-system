"""
Trades parser: parse a stringified pandas DataFrame (or JSON/CSV-like text) returned by the backtest
engine and normalize into a list of dicts suitable for insertion into the DB.

Strategy:
- Try json.loads first (some engines return JSON)
- Try pandas (if available) to read JSON/CSV
- Try csv.Sniffer with common delimiters
- Last-resort: whitespace heuristic splitting on two-or-more spaces (pandas pretty-print)
- Normalize common column names to a canonical set

Returns a list of dicts with keys such as:
  trade_index, size, entry_bar, exit_bar, entry_price, exit_price, pnl, duration, tag,
  entry_signals, exit_signals
"""

from __future__ import annotations

import csv
import io
import json
import re
from typing import Any, Dict, List, Optional

CANONICAL_COLUMNS = {
    # various possible column names mapped to canonical names
    "index": "trade_index",
    "trade_index": "trade_index",
    "size": "size",
    "qty": "size",
    "quantity": "size",
    "entry_bar": "entry_bar",
    "entrybar": "entry_bar",
    "entry": "entry_bar",
    "entry_index": "entry_bar",
    # entry/exit timestamps
    "entry_time": "entry_time",
    "entrytime": "entry_time",
    "entry timestamp": "entry_time",
    "entry_ts": "entry_time",
    "entry_date": "entry_time",
    "exit_bar": "exit_bar",
    "exitbar": "exit_bar",
    "exit": "exit_bar",
    "exit_index": "exit_bar",
    "exit_time": "exit_time",
    "exittime": "exit_time",
    "exit timestamp": "exit_time",
    "exit_ts": "exit_time",
    "exit_date": "exit_time",
    "entry_price": "entry_price",
    "entryprice": "entry_price",
    "exit_price": "exit_price",
    "exitprice": "exit_price",
    "pnl": "pnl",
    "profit": "pnl",
    "pl": "pnl",
    "duration": "duration",
    "tag": "tag",
    "entry_signals": "entry_signals",
    "exit_signals": "exit_signals",
    "signals": "entry_signals",
}


def _normalize_row(raw: Dict[Any, Any], idx: Optional[int] = None) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    # accept any-hashable keys and coerce to str for normalization
    raw = {str(k): v for k, v in raw.items()}
    # lower-case keys for matching
    mapping = {str(k).lower().strip(): v for k, v in raw.items()}
    # map known names
    for k, v in mapping.items():
        canon = CANONICAL_COLUMNS.get(k)
        if canon:
            normalized[canon] = v
        else:
            # keep unknown columns as-is (but lowercased)
            normalized[k] = v
    # ensure trade_index present
    if "trade_index" not in normalized:
        if idx is not None:
            try:
                normalized["trade_index"] = int(idx)
            except Exception:
                normalized["trade_index"] = idx or 0
        else:
            # try to extract numeric 'index' or fallback to 0
            normalized.setdefault("trade_index", 0)
    return normalized


def _parse_with_pandas(text: str) -> Optional[List[Dict[str, Any]]]:
    try:
        import pandas as pd  # type: ignore[import-not-found]
        from pandas.errors import EmptyDataError  # type: ignore[import-not-found]
    except Exception:
        return None

    # Try read_json (records) then read_csv
    try:
        df = pd.read_json(io.StringIO(text), orient="records")
        if df is not None and not df.empty:
            records = df.to_dict(orient="records")
            return [_normalize_row(dict(r), idx=i) for i, r in enumerate(records)]
    except Exception:
        pass

    try:
        df = pd.read_csv(io.StringIO(text))
        if df is not None and not df.empty:
            records = df.to_dict(orient="records")
            return [_normalize_row(dict(r), idx=i) for i, r in enumerate(records)]
    except EmptyDataError:
        return []
    except Exception:
        # Try python engine with whitespace delimiter heuristics
        try:
            df = pd.read_csv(io.StringIO(text), sep=r"\s{2,}", engine="python")
            if df is not None and not df.empty:
                records = df.to_dict(orient="records")
                return [_normalize_row(dict(r), idx=i) for i, r in enumerate(records)]
        except Exception:
            return None
    return None


def _parse_with_csv(text: str) -> Optional[List[Dict[str, Any]]]:
    sio = io.StringIO(text)
    # sniff delimiter
    try:
        sample = text[:4096]
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        sio.seek(0)
        reader = csv.DictReader(sio, dialect=dialect)
        rows = [dict(r) for r in reader]
        if rows:
            return [_normalize_row(r, idx=i) for i, r in enumerate(rows)]
    except Exception:
        pass

    # fallback: comma
    try:
        sio.seek(0)
        reader = csv.DictReader(sio)
        rows = [dict(r) for r in reader]
        if rows:
            return [_normalize_row(r, idx=i) for i, r in enumerate(rows)]
    except Exception:
        pass

    return None


def _parse_whitespace_table(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parses pretty-printed pandas DataFrame which separates columns by two or more spaces.
    Example:
    index  entry_bar  exit_bar  entry_price  exit_price   pnl
    0      100        120       10.5         12.0         1.5
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    # header detection: first line with word chars and spaces
    header = lines[0]
    # split on 2+ spaces
    cols = re.split(r"\s{2,}", header.strip())
    if len(cols) < 2:
        return None
    data = []
    for ln in lines[1:]:
        parts = re.split(r"\s{2,}", ln.strip())
        if len(parts) != len(cols):
            # if mismatch, skip or try to pad
            continue
        row = dict(zip(cols, parts))
        data.append(row)
    if not data:
        return None
    return [_normalize_row(r, idx=i) for i, r in enumerate(data)]


def parse_trades_from_string(trades_str: Optional[str]) -> List[Dict[str, Any]]:
    """
    Public parser. Returns an empty list for falsy input.

    Steps:
     - Try json.loads
     - Try pandas-based parser
     - Try csv.Sniffer-based parser
     - Try whitespace table parser
     - Fallback: return empty list
    """
    if not trades_str:
        return []

    text = trades_str.strip()

    # 1) JSON
    try:
        obj = json.loads(text)
        # If a dict representing a DF: convert to list
        if isinstance(obj, dict):
            # dict-of-lists or dict-of-dicts? try to convert to records
            # common format: {"0": {...}, "1": {...}} or {"col": [..]}
            if all(isinstance(v, list) for v in obj.values()):
                # dict of columns -> convert to records
                keys = list(obj.keys())
                length = len(next(iter(obj.values()), []))
                records = []
                for i in range(length):
                    rec = {k: obj[k][i] for k in keys}
                    records.append(rec)
                return [_normalize_row(r, idx=i) for i, r in enumerate(records)]
            # dict of records
            if all(isinstance(v, dict) for v in obj.values()):
                records = list(obj.values())
                return [_normalize_row(r, idx=i) for i, r in enumerate(records)]
            # single record
            return [_normalize_row(obj, idx=0)]
        if isinstance(obj, list):
            return [
                _normalize_row(r if isinstance(r, dict) else {"value": r}, idx=i)
                for i, r in enumerate(obj)
            ]
    except Exception:
        pass

    # 2) pandas
    try:
        pd_res = _parse_with_pandas(text)
        if pd_res is not None:
            return pd_res
    except Exception:
        pass

    # 3) csv
    try:
        csv_res = _parse_with_csv(text)
        if csv_res is not None:
            return csv_res
    except Exception:
        pass

    # 4) whitespace table
    try:
        ws = _parse_whitespace_table(text)
        if ws is not None:
            return ws
    except Exception:
        pass

    # 5) Last resort: try splitting lines and commas
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) == 1:
        # single-line value
        return [{"trade_index": 0, "value": lines[0]}]
    # multiple lines: try simple CSV split
    header = lines[0]
    cols = [c.strip() for c in re.split(r"[,\t;|]+", header) if c.strip()]
    if len(cols) >= 2:
        data = []
        for i, ln in enumerate(lines[1:]):
            parts = [p.strip() for p in re.split(r"[,\t;|]+", ln) if p.strip()][
                : len(cols)
            ]
            if len(parts) != len(cols):
                continue
            row = dict(zip(cols, parts))
            data.append(row)
        if data:
            return [_normalize_row(r, idx=i) for i, r in enumerate(data)]
    # If nothing worked, return empty list
    return []
