from __future__ import annotations

import logging
import time
from threading import Lock


class RateLimiter:
    """Simple time-based rate limiter to space out API calls.

    Ensures at least `min_interval` seconds between successive `acquire()` calls.
    """

    def __init__(self, min_interval: float = 1.0, logger: logging.Logger | None = None):
        self.min_interval = float(min_interval)
        self._last = 0.0
        self._lock = Lock()
        self._logger = logger or logging.getLogger("quant.rate_limiter")
        self.last_wait = 0.0

    def acquire(self) -> float:
        with self._lock:
            now = time.time()
            delta = now - self._last
            wait = max(0.0, self.min_interval - delta)
            self.last_wait = wait
            if wait > 0:
                self._logger.debug(
                    "rate-limit: sleeping %.3fs (min_interval=%.3fs)", wait, self.min_interval
                )
                time.sleep(wait)
            self._last = time.time()
            return wait
