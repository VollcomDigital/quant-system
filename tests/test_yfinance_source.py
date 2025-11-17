from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.data.yfinance_source import YFinanceSource


class DummyTicker:
    def __init__(self, splits=None, dividends=None, actions=None, history_df=None):
        self._splits = splits
        self._dividends = dividends
        self.actions = actions or {}
        self._history = history_df if history_df is not None else pd.DataFrame()

    @property
    def splits(self):
        return self._splits

    @property
    def dividends(self):
        return self._dividends

    def history(self, *args, **kwargs):
        return self._history


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "yfinance_cache.YFCache",
        lambda cache_dir: SimpleNamespace(
            ticker=SimpleNamespace(Ticker=lambda symbol: DummyTicker(history_df=pd.DataFrame()))
        ),
        raising=False,
    )
    return YFinanceSource(tmp_path)


def test_fetch_splits_returns_dataframe(tmp_path, monkeypatch):
    series = pd.Series([2.0], index=pd.to_datetime(["2023-01-01"], utc=True))
    fake = DummyTicker(splits=series)
    monkeypatch.setattr("yfinance.Ticker", lambda symbol: fake)

    src = YFinanceSource(tmp_path)
    df = src.fetch_splits("AAPL")
    assert list(df.columns) == ["ratio"]
    assert df.iloc[0]["ratio"] == pytest.approx(2.0)


def test_fetch_dividends_handles_actions(tmp_path, monkeypatch):
    series = pd.Series([0.5], index=pd.to_datetime(["2022-06-01"], utc=True))
    fake = DummyTicker(dividends=None, actions={"Dividends": series})
    monkeypatch.setattr("yfinance.Ticker", lambda symbol: fake)

    src = YFinanceSource(tmp_path)
    df = src.fetch_dividends("AAPL")
    assert df.iloc[0]["dividend"] == pytest.approx(0.5)


def test_fetch_splits_empty(monkeypatch, tmp_path):
    fake = DummyTicker(splits=pd.Series(dtype=float))
    monkeypatch.setattr("yfinance.Ticker", lambda symbol: fake)
    src = YFinanceSource(tmp_path)
    df = src.fetch_splits("AAPL")
    assert df.empty


def test_fetch_fundamentals(tmp_path, monkeypatch):
    financials = pd.DataFrame({"Revenue": [100.0]}, index=[pd.Timestamp("2023-01-01")])
    fake = DummyTicker(
        splits=pd.Series([2.0], index=pd.to_datetime(["2023-01-01"], utc=True)),
        dividends=pd.Series([0.5], index=pd.to_datetime(["2023-02-01"], utc=True)),
        actions={},
        history_df=pd.DataFrame(),
    )
    fake.financials = financials
    fake.balance_sheet = financials
    fake.cashflow = financials
    fake.info = {"sector": "Technology"}

    monkeypatch.setattr("yfinance.Ticker", lambda symbol: fake)

    src = YFinanceSource(tmp_path)
    data = src.fetch_fundamentals("AAPL")
    assert data["info"]["sector"] == "Technology"
    key = next(iter(data["fundamentals"]["income_statement"].keys()))
    assert data["fundamentals"]["income_statement"][key]["Revenue"] == pytest.approx(100.0)
    assert data["splits"]
    assert data["dividends"]
