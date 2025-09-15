from __future__ import annotations

import importlib

import pandas as pd

from ..base import BaseStrategy


class CtorSignalsAdapter(BaseStrategy):
    """Adapter for external classes that:

    - accept parameters via __init__(**parameters)
    - expose generate_signals(self, data: pd.DataFrame) -> pd.Series with values {1,-1,0}

    The adapter converts the 1/-1/0 signal series into entry/exit boolean series
    compatible with the backtesting engine.

    Config example:

    strategies:
      - name: bitcoin_strategy
        module: src.strategies.adapters.ctor_signals_adapter
        class: CtorSignalsAdapter
        params:
          external_module: my_pkg.bitcoin_strategy
          external_class: BitcoinStrategy
          grid:
            lookback_period: [20, 50]
            sma_period: [20, 50]
    """

    name = "external_ctor_signals"

    def param_grid(self) -> dict:
        # Grid is supplied via config under params.grid and handled by the runner
        return {}

    def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
        mod_name = params.get("external_module")
        cls_name = params.get("external_class")
        if not mod_name or not cls_name:
            raise ValueError("external_module and external_class are required in params")

        mod = importlib.import_module(mod_name)
        Cls = getattr(mod, cls_name)

        # Remove adapter-specific keys and pass the rest to constructor
        ctor_kwargs = {
            k: v
            for k, v in params.items()
            if k not in {"external_module", "external_class", "grid"}
        }
        obj = Cls(**ctor_kwargs)

        # External generate_signals returns a Series in {1, 0, -1}
        sig = obj.generate_signals(df)
        if not isinstance(sig, pd.Series):
            raise TypeError("External strategy must return a pandas Series of signals")
        entries = (sig == 1).fillna(False)
        exits = (sig == -1).fillna(False)
        return entries, exits
