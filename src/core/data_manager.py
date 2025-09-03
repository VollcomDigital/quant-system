"""
Unified Data Manager - Consolidates all data fetching and management functionality.
Supports multiple data sources including Bybit for crypto futures.
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .cache_manager import UnifiedCacheManager

warnings.filterwarnings("ignore")


@dataclass
class DataSourceConfig:
    """Configuration for data sources."""

    name: str
    priority: int
    rate_limit: float
    max_retries: int
    timeout: float
    supports_batch: bool = False
    supports_futures: bool = False
    asset_types: List[str] | None = None
    max_symbols_per_request: int = 1


class DataSource(ABC):
    """Abstract base class for all data sources."""

    def __init__(self, config: DataSourceConfig) -> None:
        self.config = config
        self.last_request_time = 0
        self.session = self._create_session()
        self.logger = logging.getLogger(f"{__name__}.{config.name}")

    def transform_symbol(self, symbol: str, asset_type: str | None = None) -> str:
        """Transform symbol to fit this data source's format."""
        return symbol  # Default: no transformation

    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy."""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _rate_limit(self) -> None:
        """Apply rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.config.rate_limit:
            time.sleep(self.config.rate_limit - elapsed)
        self.last_request_time = int(time.time())

    @abstractmethod
    def fetch_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """Fetch data for a single symbol."""

    @abstractmethod
    def fetch_batch_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols."""

    @abstractmethod
    def get_available_symbols(self, asset_type: str | None = None) -> List[str]:
        """Get available symbols for this source."""

    def standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data format across all sources."""
        if df.empty:
            return df

        df = df.copy()

        # Standardize column names
        column_mapping = {
            "Open": "open",
            "open": "open",
            "High": "high",
            "high": "high",
            "Low": "low",
            "low": "low",
            "Close": "close",
            "close": "close",
            "Adj Close": "adj_close",
            "adj_close": "adj_close",
            "Volume": "volume",
            "volume": "volume",
        }

        df.columns = [column_mapping.get(col, col.lower()) for col in df.columns]

        # Ensure required columns exist
        required_cols = ["open", "high", "low", "close"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            msg = f"Missing required columns: {missing_cols}"
            raise ValueError(msg)

        # Convert to numeric
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Sort by date
        df = df.sort_index()

        # Remove invalid data
        df = df.dropna(subset=["close"])
        df = df[
            (df["high"] >= df["low"])
            & (df["high"] >= df["open"])
            & (df["high"] >= df["close"])
            & (df["low"] <= df["open"])
            & (df["low"] <= df["close"])
        ]

        return df


class YahooFinanceSource(DataSource):
    """Yahoo Finance data source - primary for stocks, forex, commodities."""

    def __init__(self) -> None:
        config = DataSourceConfig(
            name="yahoo_finance",
            priority=1,
            rate_limit=1.5,
            max_retries=3,
            timeout=30,
            supports_batch=True,
            supports_futures=True,
            asset_types=["stocks", "forex", "commodities", "indices", "crypto"],
            max_symbols_per_request=100,
        )
        super().__init__(config)

    def transform_symbol(self, symbol: str, asset_type: str | None = None) -> str:
        """Transform symbol for Yahoo Finance format."""
        # Yahoo Finance forex format
        if asset_type == "forex" or "=" in symbol:
            return symbol  # Already in correct format (EURUSD=X)

        # Handle forex pairs without =X
        forex_pairs = [
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "USDCHF",
            "AUDUSD",
            "USDCAD",
            "NZDUSD",
            "EURJPY",
            "GBPJPY",
            "EURGBP",
            "AUDJPY",
            "EURAUD",
            "EURCHF",
            "AUDNZD",
            "GBPAUD",
            "GBPCAD",
        ]
        if symbol in forex_pairs:
            return f"{symbol}=X"

        # Crypto format - Yahoo uses dash format and typically USD quote
        if asset_type == "crypto" or any(
            crypto in symbol.upper() for crypto in ["BTC", "ETH", "ADA", "SOL"]
        ):
            up = symbol.upper()
            if "USDT" in up and "-" not in up:
                # Map USDT quote to Yahoo's USD convention, e.g., IMXUSDT -> IMX-USD
                return up.replace("USDT", "-USD")
            if "USD" in up and "-" not in up:
                return up.replace("USD", "-USD")

        return symbol

    def fetch_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance.

        Supports a 'period' kwarg (e.g. 'max', '1y') which will be preferred over
        start/end if provided. This mirrors yfinance.Ticker.history semantics.
        """
        import yfinance as yf

        self._rate_limit()

        # Transform symbol to Yahoo Finance format
        asset_type = kwargs.get("asset_type")
        transformed_symbol = self.transform_symbol(symbol, asset_type)

        # Allow callers to pass 'period' to request the provider's period-based download
        period = kwargs.get("period") or kwargs.get("period_mode") or None

        try:
            ticker = yf.Ticker(transformed_symbol)
            if period:
                # Use period-based download (yfinance handles interval constraints)
                data = ticker.history(period=period, interval=interval)
            else:
                data = ticker.history(start=start_date, end=end_date, interval=interval)

            if data is None or data.empty:
                return None

            return self.standardize_data(data)

        except Exception as e:
            self.logger.warning("Yahoo Finance fetch failed for %s: %s", symbol, e)
            return None

    def fetch_batch_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch batch data from Yahoo Finance.

        If a 'period' kwarg is provided it will be used instead of start/end
        (matches yfinance.download semantics).
        """
        import yfinance as yf

        self._rate_limit()

        period = kwargs.get("period") or kwargs.get("period_mode") or None

        try:
            if period:
                data = yf.download(
                    symbols,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    progress=False,
                )
            else:
                data = yf.download(
                    symbols,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    group_by="ticker",
                    progress=False,
                )

            result = {}
            if len(symbols) == 1:
                symbol = symbols[0]
                if not getattr(data, "empty", False):
                    result[symbol] = self.standardize_data(data)
            else:
                # yfinance.download returns a DataFrame with a top-level column for each ticker
                for symbol in symbols:
                    try:
                        if symbol in data.columns.levels[0]:
                            symbol_data = data[symbol]
                            if not getattr(symbol_data, "empty", False):
                                result[symbol] = self.standardize_data(symbol_data)
                    except Exception as exc:
                        # some downloads return a flat DataFrame for single-column cases; ignore failures per-symbol
                        self.logger.debug(
                            "Batch fetch postprocess failed for %s: %s", symbol, exc
                        )
                        continue

            return result

        except Exception as e:
            self.logger.warning("Yahoo Finance batch fetch failed: %s", e)
            return {}

    def get_available_symbols(self, asset_type: str | None = None) -> List[str]:
        """Get available symbols - not implemented for Yahoo Finance source."""
        # Yahoo Finance doesn't provide a direct API for symbol listing
        # Would need external data or hardcoded list
        self.logger.warning(
            "get_available_symbols not implemented for Yahoo Finance source"
        )
        return []


class BybitSource(DataSource):
    """Bybit data source - primary for crypto futures trading."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
    ) -> None:
        config = DataSourceConfig(
            name="bybit",
            priority=1,  # Primary for crypto
            rate_limit=0.1,  # 10 requests per second
            max_retries=3,
            timeout=30,
            supports_batch=True,
            supports_futures=True,
            asset_types=["crypto", "crypto_futures"],
            max_symbols_per_request=50,
        )
        super().__init__(config)

        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

        # Bybit endpoints
        if testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"

    def fetch_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        category: str = "linear",
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from Bybit.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            start_date: Start date
            end_date: End date
            interval: Kline interval ('1', '3', '5', '15', '30', '60', '120', '240',
                     '360', '720', 'D', 'W', 'M')
            category: Product category ('spot', 'linear', 'inverse', 'option')
        """
        self._rate_limit()

        try:
            # Convert interval to Bybit format
            bybit_interval = self._convert_interval(interval)
            if not bybit_interval:
                self.logger.error("Unsupported interval: %s", interval)
                return None

            # Convert dates to timestamps (robust to strings and tokens)
            start_dt = pd.to_datetime(start_date, errors="coerce")
            end_dt = pd.to_datetime(end_date, errors="coerce")
            if pd.isna(end_dt):
                end_dt = pd.Timestamp.utcnow()
            if pd.isna(start_dt):
                # Default window based on interval
                try:
                    if interval in {"1m", "3m", "5m", "15m", "30m"}:
                        start_dt = end_dt - pd.Timedelta(days=7)
                    elif interval in {"1h", "2h", "4h", "6h", "12h"}:
                        start_dt = end_dt - pd.Timedelta(days=90)
                    else:
                        start_dt = end_dt - pd.Timedelta(days=365)
                except Exception:
                    start_dt = end_dt - pd.Timedelta(days=90)

            start_ts = int(start_dt.timestamp() * 1000)
            end_ts = int(end_dt.timestamp() * 1000)

            # Fetch kline data
            url = f"{self.base_url}/v5/market/kline"
            params = {
                "category": category,
                "symbol": symbol,
                "interval": bybit_interval,
                "start": start_ts,
                "end": end_ts,
                "limit": 1000,
            }

            all_data = []
            current_end = end_ts

            # Fetch data in chunks (Bybit returns max 1000 records per request)
            while current_end > start_ts:
                params["end"] = current_end

                response = self.session.get(
                    url, params=params, timeout=self.config.timeout
                )
                response.raise_for_status()

                data = response.json()

                if data.get("retCode") != 0:
                    self.logger.error("Bybit API error: %s", data.get("retMsg"))
                    break

                klines = data.get("result", {}).get("list", [])
                if not klines:
                    break

                all_data.extend(klines)

                # Update end timestamp for next iteration
                current_end = int(klines[-1][0]) - 1

                # Rate limit between requests
                time.sleep(self.config.rate_limit)

            if not all_data:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(
                all_data,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "turnover",
                ],
            )

            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            df.set_index("timestamp", inplace=True)
            df = df.sort_index()

            # Convert to numeric
            numeric_cols = ["open", "high", "low", "close", "volume", "turnover"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            return self.standardize_data(df)

        except Exception as e:
            self.logger.warning("Bybit fetch failed for %s: %s", symbol, e)
            return None

    def fetch_batch_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch batch data from Bybit (sequential due to rate limits)."""
        result = {}

        for symbol in symbols:
            data = self.fetch_data(symbol, start_date, end_date, interval, **kwargs)
            if data is not None:
                result[symbol] = data

        return result

    def get_available_symbols(self, asset_type: str = "linear") -> List[str]:
        """Get available trading symbols from Bybit."""
        try:
            url = f"{self.base_url}/v5/market/instruments-info"
            params = {"category": asset_type}

            response = self.session.get(url, params=params, timeout=self.config.timeout)
            response.raise_for_status()

            data = response.json()

            if data.get("retCode") != 0:
                self.logger.error("Bybit API error: %s", data.get("retMsg"))
                return []

            instruments = data.get("result", {}).get("list", [])
            symbols = [
                inst.get("symbol")
                for inst in instruments
                if inst.get("status") == "Trading"
            ]

            return symbols

        except Exception as e:
            self.logger.error("Failed to fetch Bybit symbols: %s", e)
            return []

    def get_futures_symbols(self) -> List[str]:
        """Get crypto futures symbols."""
        return self.get_available_symbols("linear")

    def get_spot_symbols(self) -> List[str]:
        """Get crypto spot symbols."""
        return self.get_available_symbols("spot")

    def _convert_interval(self, interval: str) -> Optional[str]:
        """Convert standard interval to Bybit format."""
        mapping = {
            "1m": "1",
            "3m": "3",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "2h": "120",
            "4h": "240",
            "6h": "360",
            "12h": "720",
            "1d": "D",
            "1w": "W",
            "1M": "M",
        }
        return mapping.get(interval)


class AlphaVantageSource(DataSource):
    """Alpha Vantage source for additional stock data."""

    def __init__(self, api_key: str) -> None:
        config = DataSourceConfig(
            name="alpha_vantage",
            priority=3,
            rate_limit=12,  # 5 requests per minute
            max_retries=3,
            timeout=30,
            supports_batch=False,
            asset_types=["stocks", "forex", "commodities"],
            max_symbols_per_request=1,
        )
        super().__init__(config)
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def fetch_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage."""
        self._rate_limit()

        try:
            function = self._get_function(interval)
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": self.api_key,
                "outputsize": "full",
                "datatype": "json",
            }

            if interval not in ["1d", "1w", "1M"]:
                params["interval"] = self._convert_interval(interval)

            response = self.session.get(
                self.base_url, params=params, timeout=self.config.timeout
            )
            data = response.json()

            # Find time series data
            time_series_key = None
            for key in data.keys():
                if "Time Series" in key:
                    time_series_key = key
                    break

            if not time_series_key:
                return None

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data[time_series_key], orient="index")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Standardize column names
            df.columns = [
                col.split(". ")[-1].lower().replace(" ", "_") for col in df.columns
            ]

            # Filter by date range using UTC timezone
            start = pd.to_datetime(start_date, utc=True)
            end = pd.to_datetime(end_date, utc=True)

            # Convert data index to UTC for consistent comparison
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")

            df = df[(df.index >= start) & (df.index <= end)]

            return self.standardize_data(df) if not df.empty else None

        except Exception as e:
            self.logger.warning("Alpha Vantage fetch failed for %s: %s", symbol, e)
            return None

    def fetch_batch_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """Sequential fetch for Alpha Vantage."""
        result = {}
        for symbol in symbols:
            data = self.fetch_data(symbol, start_date, end_date, interval, **kwargs)
            if data is not None:
                result[symbol] = data
        return result

    def get_available_symbols(self, asset_type: str | None = None) -> List[str]:
        """Get available symbols - not implemented for Alpha Vantage source."""
        # Alpha Vantage doesn't provide a direct API for symbol listing
        # Would require subscription to premium endpoints or external data
        self.logger.warning(
            "get_available_symbols not implemented for Alpha Vantage source"
        )
        return []

    def _get_function(self, interval: str) -> str:
        """Get Alpha Vantage function name."""
        if interval in ["1m", "5m", "15m", "30m", "60m"]:
            return "TIME_SERIES_INTRADAY"
        if interval == "1d":
            return "TIME_SERIES_DAILY_ADJUSTED"
        if interval == "1w":
            return "TIME_SERIES_WEEKLY_ADJUSTED"
        if interval == "1M":
            return "TIME_SERIES_MONTHLY_ADJUSTED"
        return "TIME_SERIES_DAILY_ADJUSTED"

    def _convert_interval(self, interval: str) -> str:
        """Convert to Alpha Vantage format."""
        mapping = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "60min",
        }
        return mapping.get(interval, "1min")


class UnifiedDataManager:
    """
    Unified data manager that consolidates all data fetching functionality.
    Automatically routes requests to appropriate data sources based on asset type.
    """

    def __init__(self, cache_manager: UnifiedCacheManager | None = None) -> None:
        self.cache_manager = cache_manager or UnifiedCacheManager()
        self.sources: dict[str, DataSource] = {}
        self.logger = logging.getLogger(__name__)

        # Initialize default sources
        self._initialize_sources()

    def _initialize_sources(self) -> None:
        """Initialize available data sources."""
        import os

        # Yahoo Finance (always available - fallback)
        self.add_source(YahooFinanceSource())

        # Enhanced Alpha Vantage (good for stocks/forex/crypto)
        av_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if av_key:
            try:
                self.add_source(EnhancedAlphaVantageSource())
            except Exception as e:
                self.logger.warning("Could not add Enhanced Alpha Vantage: %s", e)
                # Fallback to existing implementation
                try:
                    self.add_source(AlphaVantageSource(av_key))
                except:
                    pass

        # Twelve Data (excellent coverage)
        twelve_key = os.getenv("TWELVE_DATA_API_KEY")
        if twelve_key:
            try:
                self.add_source(TwelveDataSource())
            except Exception as e:
                self.logger.warning("Could not add Twelve Data: %s", e)

        # Bybit for crypto futures (specialized)
        bybit_key = os.getenv("BYBIT_API_KEY")
        bybit_secret = os.getenv("BYBIT_API_SECRET")
        testnet = os.getenv("BYBIT_TESTNET", "false").lower() == "true"

        self.add_source(BybitSource(bybit_key, bybit_secret, testnet))

    def add_source(self, source: DataSource):
        """Add a data source."""
        self.sources[source.config.name] = source
        self.logger.debug("Added data source: %s", source.config.name)

    def get_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        use_cache: bool = True,
        asset_type: str | None = None,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """
        Get data for a symbol with intelligent source routing.

        Args:
            symbol: Symbol to fetch
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval
            use_cache: Whether to use cached data
            asset_type: Asset type hint ('crypto', 'stocks', 'forex', etc.)
            **kwargs: Additional parameters for specific sources
        """
        # If a native provider period was requested (e.g., period='max'), skip cache reads to ensure
        # we fetch the full available history from the source. We'll still write-through to cache below.
        period_requested = kwargs.get("period") or kwargs.get("period_mode")

        # Check cache first (only when no explicit provider period was requested)
        if use_cache and not period_requested:
            # Legacy fast-path: return any single cached hit immediately (maintains test expectations)
            legacy_cached = self.cache_manager.get_data(
                symbol, start_date, end_date, interval
            )
            if legacy_cached is not None:
                self.logger.debug("Cache hit (legacy) for %s", symbol)
                return legacy_cached

            # Split cache: attempt to merge a full snapshot with a recent overlay
            # Try Redis overlay first if available
            full_df = self.cache_manager.get_data(
                symbol, start_date, end_date, interval, data_type="full"
            )
            recent_df = None
            try:
                recent_df = self.cache_manager.get_recent_overlay_from_redis(
                    symbol, interval
                )
            except Exception:
                recent_df = None
            if recent_df is None:
                recent_df = self.cache_manager.get_data(
                    symbol, start_date, end_date, interval, data_type="recent"
                )
            merged = None
            if (
                full_df is not None
                and not full_df.empty
                and recent_df is not None
                and not recent_df.empty
            ):
                try:
                    merged = (
                        pd.concat([full_df, recent_df])
                        .sort_index()
                        .loc[lambda df: ~df.index.duplicated(keep="last")]
                    )
                except Exception:
                    merged = full_df
            elif full_df is not None and not full_df.empty:
                merged = full_df
            elif recent_df is not None and not recent_df.empty:
                merged = recent_df

            if merged is not None and not merged.empty:
                # If requested range extends beyond merged coverage, auto-extend by fetching missing windows
                try:
                    req_start = pd.to_datetime(start_date)
                    req_end = pd.to_datetime(end_date)
                    c_start = merged.index[0]
                    c_end = merged.index[-1]
                    need_before = req_start < c_start
                    need_after = req_end > c_end
                except Exception:
                    need_before = need_after = False

                if need_before:
                    try:
                        df_b = self.get_data(
                            symbol,
                            start_date,
                            c_start.strftime("%Y-%m-%d"),
                            interval,
                            use_cache=False,
                            asset_type=asset_type,
                            period_mode=period_requested,
                        )
                        if df_b is not None and not df_b.empty:
                            merged = (
                                pd.concat([df_b, merged])
                                .sort_index()
                                .loc[lambda df: ~df.index.duplicated(keep="last")]
                            )
                    except Exception:
                        pass
                if need_after:
                    try:
                        df_a = self.get_data(
                            symbol,
                            c_end.strftime("%Y-%m-%d"),
                            end_date,
                            interval,
                            use_cache=False,
                            asset_type=asset_type,
                            period_mode=period_requested,
                        )
                        if df_a is not None and not df_a.empty:
                            merged = (
                                pd.concat([merged, df_a])
                                .sort_index()
                                .loc[lambda df: ~df.index.duplicated(keep="last")]
                            )
                    except Exception:
                        pass

                return merged

        # Determine asset type if not provided
        if not asset_type:
            asset_type = self._detect_asset_type(symbol)

        # Get appropriate sources for asset type
        suitable_sources = self._get_sources_for_asset_type(asset_type)

        # Try each source in priority order
        for source in suitable_sources:
            try:
                # Pass asset_type to enable symbol transformation
                kwargs["asset_type"] = asset_type
                data = source.fetch_data(
                    symbol, start_date, end_date, interval, **kwargs
                )
                if data is not None and not data.empty:
                    # Always write-through to cache on a fresh fetch.
                    # Use split-caching: store 'full' when provider period requested, else 'recent'.
                    cache_kind = "full" if period_requested else "recent"
                    ttl_hours = 24 if cache_kind == "recent" else 24 * 30
                    try:
                        self.cache_manager.cache_data(
                            symbol,
                            data,
                            interval,
                            source.config.name,
                            data_type=cache_kind,
                            ttl_hours=ttl_hours,
                        )
                    except Exception as e:
                        self.logger.warning(
                            "Failed to cache data for %s from %s: %s",
                            symbol,
                            source.config.name,
                            e,
                        )

                    self.logger.info(
                        "Successfully fetched %s from %s", symbol, source.config.name
                    )
                    # Freshness check for daily bars
                    if interval == "1d":
                        try:
                            last_bar = data.index[-1].date()
                            from pandas.tseries.offsets import BDay

                            expected = (
                                pd.Timestamp(datetime.utcnow().date()) - BDay(1)
                            ).date()
                            if last_bar < expected:
                                self.logger.warning(
                                    "Data for %s seems stale: last=%s expected>=%s",
                                    symbol,
                                    last_bar,
                                    expected,
                                )
                        except Exception:
                            pass
                    return data

            except Exception as e:
                self.logger.warning(
                    "Source %s failed for %s: %s", source.config.name, symbol, e
                )
                continue

        self.logger.error("All sources failed for %s", symbol)
        return None

    def get_batch_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        use_cache: bool = True,
        asset_type: str | None = None,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """Get data for multiple symbols with intelligent batching and cache-first behavior."""
        result: Dict[str, pd.DataFrame] = {}

        # Group symbols by asset type for optimal source selection
        symbol_groups = self._group_symbols_by_type(symbols, asset_type)

        for group_type, group_symbols in symbol_groups.items():
            sources = self._get_sources_for_asset_type(group_type)

            # If caching enabled, try to satisfy from cache first to avoid external requests
            missing_symbols: List[str] = []
            if use_cache:
                for symbol in list(group_symbols):
                    try:
                        full_df = self.cache_manager.get_data(
                            symbol, start_date, end_date, interval, data_type="full"
                        )
                        recent_df = self.cache_manager.get_data(
                            symbol, start_date, end_date, interval, data_type="recent"
                        )
                        merged = None
                        if (
                            full_df is not None
                            and not full_df.empty
                            and recent_df is not None
                            and not recent_df.empty
                        ):
                            merged = (
                                pd.concat([full_df, recent_df])
                                .sort_index()
                                .loc[lambda df: ~df.index.duplicated(keep="last")]
                            )
                        elif full_df is not None and not full_df.empty:
                            merged = full_df
                        elif recent_df is not None and not recent_df.empty:
                            merged = recent_df

                        if merged is not None and not merged.empty:
                            result[symbol] = merged
                            # Track that we used cache for this symbol
                            continue
                        missing_symbols.append(symbol)
                    except Exception as e:
                        self.logger.warning("Cache lookup failed for %s: %s", symbol, e)
                        missing_symbols.append(symbol)
            else:
                missing_symbols = list(group_symbols)

            # Try batch-capable sources for missing symbols
            for source in sources:
                if not missing_symbols:
                    break

                if source.config.supports_batch and len(missing_symbols) > 1:
                    try:
                        batch_data = source.fetch_batch_data(
                            missing_symbols, start_date, end_date, interval, **kwargs
                        )

                        # Add fetched data to result and update cache
                        fetched_symbols = []
                        for symbol, data in batch_data.items():
                            if data is not None and not data.empty:
                                result[symbol] = data
                                fetched_symbols.append(symbol)
                                if use_cache:
                                    try:
                                        self.cache_manager.cache_data(
                                            symbol, data, interval, source.config.name
                                        )
                                    except Exception as e:
                                        self.logger.warning(
                                            "Failed to cache data for %s from %s: %s",
                                            symbol,
                                            source.config.name,
                                            e,
                                        )

                        # Remove fetched symbols from missing list
                        if fetched_symbols:
                            missing_symbols = [
                                s for s in missing_symbols if s not in fetched_symbols
                            ]

                    except Exception as e:
                        self.logger.warning(
                            "Batch fetch failed from %s: %s", source.config.name, e
                        )

            # Fall back to individual requests for any remaining missing symbols
            for symbol in missing_symbols:
                try:
                    individual_data = self.get_data(
                        symbol,
                        start_date,
                        end_date,
                        interval,
                        use_cache,
                        group_type,
                        **kwargs,
                    )
                    if individual_data is not None:
                        result[symbol] = individual_data
                except Exception as e:
                    self.logger.warning("Individual fetch failed for %s: %s", symbol, e)

        return result

    def get_crypto_futures_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Get crypto futures data specifically from Bybit."""
        bybit_source = self.sources.get("bybit")
        if not bybit_source:
            self.logger.error("Bybit source not available for futures data")
            return None

        # Check cache first
        if use_cache:
            cached_data = self.cache_manager.get_data(
                symbol, start_date, end_date, interval, "futures"
            )
            if cached_data is not None:
                return cached_data

        try:
            data = bybit_source.fetch_data(
                symbol, start_date, end_date, interval, category="linear"
            )

            if data is not None and use_cache:
                self.cache_manager.cache_data(
                    symbol, data, interval, "bybit", data_type="futures"
                )

            return data

        except Exception as e:
            self.logger.error("Failed to fetch futures data for %s: %s", symbol, e)
            return None

    def _detect_asset_type(self, symbol: str) -> str:
        """Detect asset type from symbol."""
        symbol_upper = symbol.upper()

        # Crypto patterns
        if any(
            pattern in symbol_upper for pattern in ["USDT", "BTC", "ETH", "BNB", "ADA"]
        ) or (symbol_upper.endswith("USD") and len(symbol_upper) > 6):
            return "crypto"
        if "-USD" in symbol_upper:
            return "crypto"

        # Forex patterns
        if symbol_upper.endswith("=X") or len(symbol_upper) == 6:
            return "forex"

        # Futures patterns
        if symbol_upper.endswith("=F"):
            return "commodities"

        # Default to stocks
        return "stocks"

    # Global override for source ordering per asset type (process-wide)
    _global_source_order_overrides: dict[str, list[str]] = {}

    @classmethod
    def set_source_order_override(
        cls, asset_type: str, ordered_sources: list[str]
    ) -> None:
        cls._global_source_order_overrides[asset_type] = list(ordered_sources)

    def _get_sources_for_asset_type(self, asset_type: str) -> List[DataSource]:
        """Get appropriate sources for asset type, sorted by priority or override."""
        suitable_sources = []

        for source in self.sources.values():
            if not source.config.asset_types or asset_type in source.config.asset_types:
                suitable_sources.append(source)

        # Optional filtering for crypto: allow disabling Yahoo/AlphaVantage via env,
        # and prefer Bybit/Twelve when available to reduce noisy fallbacks.
        if asset_type == "crypto":
            import os as _os

            disable_yahoo = _os.getenv("DISABLE_YAHOO_CRYPTO", "false").lower() in {
                "1",
                "true",
                "yes",
            }
            disable_av = _os.getenv("DISABLE_AV_CRYPTO", "false").lower() in {
                "1",
                "true",
                "yes",
            }
            names = {s.config.name for s in suitable_sources}
            has_primary = any(n in names for n in {"bybit", "twelve_data"})
            if disable_yahoo or has_primary:
                suitable_sources = [
                    s for s in suitable_sources if s.config.name != "yahoo_finance"
                ]
            if disable_av or has_primary:
                suitable_sources = [
                    s for s in suitable_sources if s.config.name != "alpha_vantage"
                ]

        override = self._global_source_order_overrides.get(asset_type)
        if override:
            order_idx = {name: i for i, name in enumerate(override)}
            suitable_sources.sort(key=lambda x: order_idx.get(x.config.name, 10_000))
        else:
            if asset_type == "crypto":
                suitable_sources.sort(
                    key=lambda x: (0 if x.config.name == "bybit" else x.config.priority)
                )
            else:
                suitable_sources.sort(key=lambda x: x.config.priority)

        return suitable_sources

    def probe_and_set_order(
        self,
        asset_type: str,
        symbols: list[str],
        interval: str = "1d",
        sample_size: int = 5,
    ) -> list[str]:
        """Probe sources for coverage and set a global ordering by longest history.

        Skips cache and uses provider period='max'. Returns ordered source names.
        """
        sym_sample = symbols[: max(1, min(sample_size, len(symbols)))]
        candidates = [s for s in self._get_sources_for_asset_type(asset_type)]
        scores: list[tuple[str, int, pd.Timestamp | None]] = []

        for src in candidates:
            total_rows = 0
            earliest: pd.Timestamp | None = None
            for s in sym_sample:
                try:
                    df = src.fetch_data(
                        s,
                        start_date="1900-01-01",
                        end_date=datetime.utcnow().date().isoformat(),
                        interval=interval,
                        asset_type=asset_type,
                        period="max",
                        period_mode="max",
                    )
                    if df is not None and not df.empty:
                        total_rows += len(df)
                        f = df.index[0]
                        earliest = f if earliest is None or f < earliest else earliest
                except Exception as exc:
                    self.logger.debug(
                        "Probe error for %s via %s: %s", s, src.config.name, exc
                    )
                    continue
            scores.append((src.config.name, total_rows, earliest))

        def _key(t: tuple[str, int, pd.Timestamp | None]):
            name, rows, first = t
            first_val = first.value if hasattr(first, "value") else 2**63 - 1
            return (-rows, first_val)

        ordered = [name for name, *_ in sorted(scores, key=_key)]
        if ordered:
            self.set_source_order_override(asset_type, ordered)
        return ordered

    def _group_symbols_by_type(
        self, symbols: List[str], default_type: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Group symbols by detected asset type."""
        groups: Dict[str, List[str]] = {}

        for symbol in symbols:
            asset_type = default_type or self._detect_asset_type(symbol)
            if asset_type not in groups:
                groups[asset_type] = []
            groups[asset_type].append(symbol)

        return groups

    def get_available_crypto_futures(self) -> List[str]:
        """Get available crypto futures symbols."""
        bybit_source = self.sources.get("bybit")
        if bybit_source:
            return bybit_source.get_futures_symbols()
        return []

    def get_source_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all data sources."""
        status = {}
        for name, source in self.sources.items():
            status[name] = {
                "priority": source.config.priority,
                "rate_limit": source.config.rate_limit,
                "supports_batch": source.config.supports_batch,
                "supports_futures": source.config.supports_futures,
                "asset_types": source.config.asset_types,
                "max_symbols_per_request": source.config.max_symbols_per_request,
            }
        return status


# Additional Data Sources


class EnhancedAlphaVantageSource(DataSource):
    """Enhanced Alpha Vantage data source - excellent for stocks, forex, crypto."""

    def __init__(self) -> None:
        config = DataSourceConfig(
            name="alpha_vantage_enhanced",
            priority=2,
            rate_limit=5.0,  # 5 calls per minute for free tier
            max_retries=3,
            timeout=30.0,
            supports_batch=False,
            asset_types=["stock", "forex", "crypto", "commodity"],
        )
        super().__init__(config)
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
        self.base_url = "https://www.alphavantage.co/query"

    def transform_symbol(self, symbol: str, asset_type: str | None = None) -> str:
        """Transform symbol for Alpha Vantage format."""
        # Alpha Vantage forex format (no =X suffix)
        if "=X" in symbol:
            return symbol.replace("=X", "")

        # Alpha Vantage crypto format (no dash)
        if "-USD" in symbol:
            return symbol.replace("-USD", "USD")

        return symbol

    def fetch_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage."""
        try:
            self._rate_limit()

            # Transform symbol to Alpha Vantage format
            asset_type = kwargs.get("asset_type")
            transformed_symbol = self.transform_symbol(symbol, asset_type)

            # Map intervals
            av_interval = self._map_interval(interval)
            function = self._get_function(transformed_symbol, interval)

            params = {
                "function": function,
                "symbol": transformed_symbol,
                "apikey": self.api_key,
                "outputsize": "full",
                "datatype": "json",
            }

            if interval in ["1min", "5min", "15min", "30min", "60min"]:
                params["interval"] = av_interval

            response = self.session.get(
                self.base_url, params=params, timeout=self.config.timeout
            )
            response.raise_for_status()

            data = response.json()

            # Check for API errors
            if "Error Message" in data:
                self.logger.error("Alpha Vantage error: %s", data["Error Message"])
                return None

            if "Note" in data:
                self.logger.warning("Alpha Vantage rate limit: %s", data["Note"])
                return None

            # Parse data
            time_series_key = self._get_time_series_key(data)
            if not time_series_key:
                return None

            df = self._parse_time_series(data[time_series_key])
            if df is not None:
                df = self._filter_date_range(df, start_date, end_date)

            return df

        except Exception as e:
            self.logger.error("Error fetching %s from Alpha Vantage: %s", symbol, e)
            return None

    def _map_interval(self, interval: str) -> str:
        """Map internal intervals to Alpha Vantage intervals."""
        mapping = {
            "1min": "1min",
            "5min": "5min",
            "15min": "15min",
            "30min": "30min",
            "1h": "60min",
            "1d": "daily",
        }
        return mapping.get(interval, "daily")

    def _get_function(self, symbol: str, interval: str) -> str:
        """Get appropriate Alpha Vantage function."""
        if "/" in symbol:  # Forex
            if interval == "1d":
                return "FX_DAILY"
            return "FX_INTRADAY"
        if any(crypto in symbol.upper() for crypto in ["BTC", "ETH", "LTC", "XRP"]):
            if interval == "1d":
                return "DIGITAL_CURRENCY_DAILY"
            return "CRYPTO_INTRADAY"
        # Stocks
        if interval == "1d":
            return "TIME_SERIES_DAILY"
        return "TIME_SERIES_INTRADAY"

    def _get_time_series_key(self, data: dict) -> Optional[str]:
        """Find the time series key in the response."""
        for key in data:
            if "Time Series" in key:
                return key
        return None

    def _parse_time_series(self, time_series: dict) -> Optional[pd.DataFrame]:
        """Parse time series data into DataFrame."""
        try:
            df = pd.DataFrame.from_dict(time_series, orient="index")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Standardize column names
            column_mapping = {}
            for col in df.columns:
                if "open" in col.lower():
                    column_mapping[col] = "Open"
                elif "high" in col.lower():
                    column_mapping[col] = "High"
                elif "low" in col.lower():
                    column_mapping[col] = "Low"
                elif "close" in col.lower():
                    column_mapping[col] = "Close"
                elif "volume" in col.lower():
                    column_mapping[col] = "Volume"

            df = df.rename(columns=column_mapping)

            # Convert to numeric
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            return df

        except Exception as e:
            self.logger.error("Error parsing Alpha Vantage data: %s", e)
            return None

    def fetch_batch_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        **kwargs: Any,
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols."""
        result = {}
        for symbol in symbols:
            data = self.fetch_data(symbol, start_date, end_date, interval, **kwargs)
            if data is not None:
                result[symbol] = data
        return result

    def get_available_symbols(self, asset_type: str | None = None) -> list[str]:
        """Get available symbols for this source."""
        # Alpha Vantage doesn't provide a comprehensive symbol list
        # Return common symbols based on asset type
        if asset_type == "stock":
            return ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META"]
        if asset_type == "forex":
            return ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF"]
        if asset_type == "crypto":
            return ["BTC/USD", "ETH/USD", "LTC/USD", "XRP/USD"]
        return []


class TwelveDataSource(DataSource):
    """Twelve Data source - excellent coverage for stocks, forex, crypto, indices."""

    def __init__(self) -> None:
        config = DataSourceConfig(
            name="twelve_data",
            priority=2,
            rate_limit=1.0,  # 8 requests per minute for free tier
            max_retries=3,
            timeout=30.0,
            supports_batch=True,
            max_symbols_per_request=8,
            asset_types=["stock", "forex", "crypto", "index", "etf"],
        )
        super().__init__(config)
        self.api_key = os.getenv("TWELVE_DATA_API_KEY", "demo")
        self.base_url = "https://api.twelvedata.com"

    def transform_symbol(self, symbol: str, asset_type: str | None = None) -> str:
        """Transform symbol for Twelve Data format."""
        # Twelve Data forex format (use slash format)
        if "=X" in symbol:
            base_symbol = symbol.replace("=X", "")
            if len(base_symbol) == 6:  # EURUSD -> EUR/USD
                return f"{base_symbol[:3]}/{base_symbol[3:]}"

        # Twelve Data crypto format (no dash)
        up = symbol.upper()
        if "-USD" in up:
            return up.replace("-USD", "USD")
        if "-USDT" in up:
            up = up.replace("-USDT", "/USDT")
            # fallthrough to exchange append below
            # return here was removed to allow exchange tagging
        # IMXUSDT -> IMX/USDT
        if up.endswith("USDT") and "/" not in up and "-" not in up:
            up = f"{up[:-4]}/USDT"

        # Optional exchange routing for crypto, e.g., IMX/USDT:BINANCE
        try:
            import os as _os

            exch = _os.getenv("TWELVE_DATA_CRYPTO_EXCHANGE", "").strip()
            if exch and ":" not in up and (asset_type == "crypto" or "/" in up):
                up = f"{up}:{exch.upper()}"
        except Exception:
            pass

        return up

    def fetch_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Twelve Data."""
        try:
            self._rate_limit()

            # Transform symbol to Twelve Data format
            asset_type = kwargs.get("asset_type")
            transformed_symbol = self.transform_symbol(symbol, asset_type)

            # Coerce dates (supports strings or datetime-like)
            start_dt = pd.to_datetime(start_date, errors="coerce")
            end_dt = pd.to_datetime(end_date, errors="coerce")
            if pd.isna(end_dt):
                end_dt = pd.Timestamp.utcnow()
            if pd.isna(start_dt):
                # default to one year window for daily; smaller for intraday
                if interval in {"1m", "5m", "15m", "30m", "1h", "4h"}:
                    start_dt = end_dt - pd.Timedelta(days=30)
                else:
                    start_dt = end_dt - pd.Timedelta(days=365)

            params = {
                "symbol": transformed_symbol,
                "interval": self._map_interval(interval),
                "start_date": start_dt.strftime("%Y-%m-%d"),
                "end_date": end_dt.strftime("%Y-%m-%d"),
                "apikey": self.api_key,
                "format": "JSON",
                "outputsize": 5000,
            }

            url = f"{self.base_url}/time_series"
            response = self.session.get(url, params=params, timeout=self.config.timeout)
            response.raise_for_status()

            data = response.json()

            if "code" in data and data["code"] != 200:
                self.logger.error(
                    "Twelve Data error: %s", data.get("message", "Unknown error")
                )
                return None

            if "values" not in data:
                self.logger.warning("No data returned for %s", symbol)
                return None

            return self._parse_twelve_data(data["values"])

        except Exception as e:
            self.logger.error("Error fetching %s from Twelve Data: %s", symbol, e)
            return None

    def _map_interval(self, interval: str) -> str:
        """Map internal intervals to Twelve Data intervals."""
        mapping = {
            "1min": "1min",
            "5min": "5min",
            "15min": "15min",
            "30min": "30min",
            "1h": "1h",
            "4h": "4h",
            "1d": "1day",
            "1wk": "1week",
        }
        return mapping.get(interval, "1day")

    def _parse_twelve_data(self, values: list) -> Optional[pd.DataFrame]:
        """Parse Twelve Data values into DataFrame."""
        try:
            df = pd.DataFrame(values)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")

            # Convert to numeric and rename columns
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            )

            # Select standard columns
            columns = ["Open", "High", "Low", "Close"]
            if "Volume" in df.columns:
                columns.append("Volume")

            df = df[columns]
            return df.sort_index()

        except Exception as e:
            self.logger.error("Error parsing Twelve Data: %s", e)
            return None

    def fetch_batch_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        **kwargs: Any,
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols."""
        result = {}
        for symbol in symbols:
            data = self.fetch_data(symbol, start_date, end_date, interval, **kwargs)
            if data is not None:
                result[symbol] = data
        return result

    def get_available_symbols(self, asset_type: str | None = None) -> list[str]:
        """Get available symbols for this source."""
        # Twelve Data doesn't provide a comprehensive symbol list in free tier
        # Return common symbols based on asset type
        if asset_type == "stock":
            return ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX"]
        if asset_type == "forex":
            return ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD"]
        if asset_type == "crypto":
            return ["BTC/USD", "ETH/USD", "LTC/USD", "XRP/USD", "ADA/USD", "DOT/USD"]
        return []


# (end of module)
