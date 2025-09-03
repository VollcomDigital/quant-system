#!/usr/bin/env python3
"""
Run direct backtest comparisons for a short sample of crypto assets.
Writes JSON output to exports/comparison_sample_Q3_2025.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from src.core.direct_backtest import run_strategy_comparison
from src.core.external_strategy_loader import get_strategy_loader

# Ensure external loader points at quant-strategies algorithms python directory
project_root = Path(__file__).resolve().parent.parent
quant_path = project_root / "quant-strategies" / "algorithms" / "python"
get_strategy_loader(str(quant_path))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("direct-sample")

# Strategies to test: use all external strategies discovered by StrategyFactory (fallback to default set)
from src.core.strategy import StrategyFactory

_external_strats = StrategyFactory.list_strategies().get("external", [])
STRATEGIES = (
    _external_strats
    if _external_strats
    else ["BuyAndHold", "rsi", "macd", "bollinger_bands"]
)

# Sample 10 representative assets (mix of BTC/ETH/large caps/mid/small caps)
SYMBOLS = [
    "BTCUSD",
    "ETHUSD",
    "SOLUSDT",
    "BNBUSDT",
    "AVAXUSDT",
    "DOGEUSDT",
    "SUIUSDT",
    "RNDRUSDT",
    "AGIXUSDT",
    "AAVEUSDT",
]

# Date range - using a broad crypto range (adjust if your local data differs)
START_DATE = "1970-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
# Comprehensive timeframes to test (user requested): 1m,5m,15m,1h,4h,1d,1wk
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d", "1wk"]
INITIAL_CAPITAL = 10000.0

results = {}
for timeframe in TIMEFRAMES:
    logger.info("Running comparisons for timeframe %s", timeframe)
    for symbol in SYMBOLS:
        logger.info("  Running strategy comparison for %s (%s)", symbol, timeframe)
        try:
            comp = run_strategy_comparison(
                symbol=symbol,
                strategies=STRATEGIES,
                start_date=START_DATE,
                end_date=END_DATE,
                timeframe=timeframe,
                initial_capital=INITIAL_CAPITAL,
            )
            # Store per-symbol per-timeframe
            results.setdefault(symbol, {})[timeframe] = comp
            logger.info(
                "    Completed %s %s: total_strategies=%s, successful=%s",
                symbol,
                timeframe,
                comp.get("total_strategies"),
                comp.get("successful_strategies"),
            )
        except Exception as e:
            logger.error("    Failed for %s %s: %s", symbol, timeframe, e)
            results.setdefault(symbol, {})[timeframe] = {"error": str(e)}

# Ensure output dir exists (should already)
output_path = Path("exports/comparison_sample_Q3_2025.json")
with output_path.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str)

print(f"Saved sample comparison results to {output_path}")
