from __future__ import annotations

import re


def _map_yfinance(symbol: str) -> str:
    s = symbol.strip()
    # Normalize common punctuation to Yahoo style first
    class_share_map = {
        "BRK.B": "BRK-B",
        "BRK.A": "BRK-A",
        "BF.B": "BF-B",
        "BF.A": "BF-A",
        "HEI.A": "HEI-A",
        "HEI.B": "HEI-B",
        "LEN.B": "LEN-B",
    }
    if s.upper() in class_share_map:
        return class_share_map[s.upper()]
    # Indices common aliases -> Yahoo caret codes
    index_map = {
        "SPX": "^GSPC",
        "SP500": "^GSPC",
        "GSPC": "^GSPC",
        "NDX": "^NDX",
        "DJI": "^DJI",
        "RUT": "^RUT",
        "VIX": "^VIX",
        # International common names
        "DAX": "^GDAXI",
        "CAC": "^FCHI",
        "FTSE": "^FTSE",
        "NIKKEI": "^N225",
        "N225": "^N225",
        "HSI": "^HSI",
        "EUROSTOXX50": "^STOXX50E",
        "SX5E": "^STOXX50E",
    }
    if s.upper() in index_map:
        return index_map[s.upper()]

    # Forex: canonical EURUSD or EUR/USD -> EURUSD=X on Yahoo
    if re.fullmatch(r"[A-Z]{6}", s):
        return f"{s}=X"
    if re.fullmatch(r"[A-Z]{3}/[A-Z]{3}", s):
        base, quote = s.split("/")
        return f"{base}{quote}=X"

    # Futures: if user passed root like GC, CL, SI, NG, ZC, ZS, ZW map to =F
    futures_roots = {
        # Metals
        "GC",
        "SI",
        "HG",
        "PL",
        "PA",
        # Energy
        "CL",
        "NG",
        "HO",
        "RB",
        "BZ",
        # Grains/softs/livestock
        "ZC",
        "ZS",
        "ZM",
        "ZL",
        "ZW",
        "ZO",
        "KC",
        "SB",
        "CC",
        "CT",
        "OJ",
        "LE",
        "HE",
        "GF",
        # Rates
        "ZB",
        "ZN",
        "ZF",
        "ZT",
    }
    if s.upper() in futures_roots:
        return f"{s}=F"

    # Crypto: BTCUSD / BTC/USDT / BTCUSDT -> BTC-USD on Yahoo
    up = s.upper()
    if re.fullmatch(r"[A-Z]{2,6}USD(T)?", up):
        base = up[:-4] if up.endswith("USDT") else up[:-3]
        return f"{base}-USD"
    if "/" in up and (up.endswith("/USD") or up.endswith("/USDT")):
        base = up.split("/")[0]
        return f"{base}-USD"

    return s


def _strip_yahoo_decoration(symbol: str) -> str:
    s = symbol.strip()
    # Remove Yahoo-specific adornments for providers that don't use them
    if s.startswith("^"):
        s = s[1:]
    s = re.sub(r"(=F|=X)$", "", s)
    # Convert share class hyphen form back to dot (e.g., BRK-B -> BRK.B)
    if re.fullmatch(r"[A-Z]{1,5}-[A-Z]", s):
        s = s.replace("-", ".")
    return s


def map_symbol(provider: str, symbol: str) -> str:
    p = provider.lower()
    if p == "yfinance":
        return _map_yfinance(symbol)
    if p == "finnhub":
        s = symbol.strip().upper()
        # FX majors: EURUSD -> OANDA:EUR_USD; EUR/USD -> OANDA:EUR_USD
        if re.fullmatch(r"[A-Z]{6}", s):
            return f"OANDA:{s[:3]}_{s[3:]}"
        if re.fullmatch(r"[A-Z]{3}/[A-Z]{3}", s):
            base, quote = s.split("/")
            return f"OANDA:{base}_{quote}"
        return s
    if p == "twelvedata":
        s = symbol.strip().upper()
        # FX: EURUSD -> EUR/USD
        if re.fullmatch(r"[A-Z]{6}", s):
            return f"{s[:3]}/{s[3:]}"
        return s
    if p == "alphavantage":
        # AV handled in-source; keep symbol as-is
        return symbol.strip()
    # polygon/tiingo/alpaca typically don't accept Yahoo adornments
    if p in {"polygon", "tiingo", "alpaca"}:
        return _strip_yahoo_decoration(symbol)
    return symbol
