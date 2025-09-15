from types import SimpleNamespace

from src.utils.symbols import DiscoverOptions, discover_ccxt_symbols


class DummyExchange:
    def __init__(self, tickers=None, markets=None):
        self._tickers = tickers or {}
        self._markets = markets or {}

    def fetch_tickers(self):
        return self._tickers

    def load_markets(self):
        return self._markets


def test_discover_symbols_monkeypatch(monkeypatch):
    # Patch ccxt.binance to our dummy class
    import src.utils.symbols as symbols_mod

    dummy = DummyExchange(
        tickers={
            "AAA/USDT": {"quoteVolume": 200},
            "BBB/USDT": {"quoteVolume": 100},
            "CCC/USD": {"quoteVolume": 999},  # filtered by quote
        }
    )
    monkeypatch.setattr(symbols_mod, "ccxt", SimpleNamespace(binance=lambda cfg: dummy))

    res = discover_ccxt_symbols(DiscoverOptions(exchange="binance", quote="USDT", top_n=2))
    assert len(res) == 2
    assert res[0][0] == "AAA/USDT"
    assert res[1][0] == "BBB/USDT"
