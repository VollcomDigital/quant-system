from __future__ import annotations

import importlib

import pandas as pd

from ..base import BaseStrategy


class GenericExternalAdapter(BaseStrategy):
    """Adapter to wrap an external strategy without modifying its code.

    Usage in config (example):

    strategies:
      - name: ext_rsi
        module: src.strategies.adapters.generic_adapter
        class: GenericExternalAdapter
        params:
          external_module: my_repo.rsi.module
          external_class: RSIStrategy   # or use external_function: rsi_generate
          grid:
            rsi_window: [14, 21]
            rsi_buy: [30]
            rsi_sell: [70]

    The runner will pass a merged params dict to generate_signals that includes both
    static adapter params (external_module/class/function) and the current grid values.
    """

    name = "external_generic"

    def param_grid(self) -> dict:
        # Grid is supplied via config under params.grid and handled by the runner
        return {}

    def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
        mod_name = params.get("external_module")
        cls_name = params.get("external_class")
        fn_name = params.get("external_function")
        if not mod_name:
            raise ValueError("external_module is required in params for GenericExternalAdapter")

        mod = importlib.import_module(mod_name)

        if cls_name:
            Cls = getattr(mod, cls_name)
            obj = Cls()
            if hasattr(obj, "generate_signals"):
                return obj.generate_signals(df, params)
            # Callable class instance
            return obj(df, params)
        if fn_name:
            func = getattr(mod, fn_name)
            return func(df, params)

        raise ValueError("Provide either external_class or external_function in params")
