from __future__ import annotations

import time
from threading import Lock


class RateLimiter:
    """Simple time-based rate limiter to space out API calls.

    Ensures at least `min_interval` seconds between successive `acquire()` calls.
    """

    def __init__(self, min_interval: float = 1.0):
        self.min_interval = float(min_interval)
        self._last = 0.0
        self._lock = Lock()

    def acquire(self):
        with self._lock:
            now = time.time()
            delta = now - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.time()
