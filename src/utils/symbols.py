from __future__ import annotations

import time
from dataclasses import dataclass

# Optional import to ease local testing where ccxt may not be available
try:  # pragma: no cover
    import ccxt  # type: ignore
except Exception:  # pragma: no cover
    ccxt = None  # type: ignore


@dataclass
class DiscoverOptions:
    exchange: str = "binance"
    quote: str = "USDT"
    top_n: int = 50
    min_volume: float = 0.0


def discover_ccxt_symbols(opts: DiscoverOptions) -> list[tuple[str, float]]:
    if ccxt is None:
        raise ImportError("ccxt is required to discover symbols")
    ex = getattr(ccxt, opts.exchange)({"enableRateLimit": True})
    # Prefer fetchTickers if supported; otherwise fallback to markets data
    result: list[tuple[str, float]] = []
    backoff = 1.0
    max_backoff = 30.0
    try:
        tickers = None
        for _ in range(5):
            try:
                tickers = ex.fetch_tickers()
                break
            except Exception as e:
                should_backoff = False
                if ccxt is not None:
                    throttling = (
                        getattr(ccxt, "RateLimitExceeded", Exception),
                        getattr(ccxt, "DDoSProtection", Exception),
                        getattr(ccxt, "ExchangeNotAvailable", Exception),
                        getattr(ccxt, "NetworkError", Exception),
                    )
                    should_backoff = isinstance(e, throttling)
                if should_backoff:
                    time.sleep(backoff)
                    backoff = min(max_backoff, backoff * 2)
                    continue
                raise
        if tickers is None:
            tickers = ex.fetch_tickers()
        for symbol, t in tickers.items():
            # Filter by quote currency and spot markets
            if not isinstance(symbol, str) or "/" not in symbol:
                continue
            base, quote = symbol.split("/")
            if quote.upper() != opts.quote.upper():
                continue
            vol = float(t.get("quoteVolume", t.get("baseVolume", 0.0)) or 0.0)
            if vol >= opts.min_volume:
                result.append((symbol, vol))
    except Exception:
        # Fallback to markets when tickers failed
        markets = ex.load_markets()
        for symbol, m in markets.items():
            if m.get("active") is False:
                continue
            if m.get("spot") is not True:
                continue
            if m.get("quote", "").upper() != opts.quote.upper():
                continue
            result.append((symbol, 0.0))

    # Sort by volume desc (zeros at end) and take top_n
    result.sort(key=lambda x: x[1], reverse=True)
    return result[: opts.top_n]
