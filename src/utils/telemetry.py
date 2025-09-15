from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from typing import Any


def get_logger(name: str = "quant") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def log_json(logger: logging.Logger, event: str, **kwargs: Any) -> None:
    payload = {"event": event, **kwargs}
    logger.info(json.dumps(payload, default=str))


@contextmanager
def time_block(logger: logging.Logger, event: str, **kwargs: Any):
    start = time.perf_counter()
    try:
        yield
    finally:
        dur = time.perf_counter() - start
        log_json(logger, event, duration_sec=round(dur, 4), **kwargs)
