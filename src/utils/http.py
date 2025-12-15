from __future__ import annotations

import logging
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class LoggingHTTPAdapter(HTTPAdapter):
    """HTTPAdapter that logs request attempts and durations."""

    def __init__(
        self,
        *args,
        logger: logging.Logger | None = None,
        log_level: int = logging.DEBUG,
        **kwargs,
    ):
        self.logger = logger or logging.getLogger("quant.http")
        self.log_level = log_level
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):  # type: ignore[override]
        start = time.perf_counter()
        try:
            response = super().send(request, **kwargs)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            self.logger.log(
                self.log_level,
                "http-error %s %s after %.3fs: %s",
                getattr(request, "method", "UNKNOWN"),
                getattr(request, "url", "<unknown>"),
                elapsed,
                exc,
            )
            raise

        elapsed = time.perf_counter() - start
        attempts = 1
        raw = getattr(response, "raw", None)
        history = getattr(getattr(raw, "retries", None), "history", None)
        if history:
            attempts = len(history) + 1

        self.logger.log(
            self.log_level,
            "http %s %s -> %s in %.3fs (attempt=%s)",
            getattr(request, "method", "UNKNOWN"),
            getattr(request, "url", "<unknown>"),
            getattr(response, "status_code", "?"),
            elapsed,
            attempts,
        )
        return response


def create_retry_session(
    total: int = 5,
    backoff_factor: float = 0.5,
    status_forcelist: tuple = (429, 500, 502, 503, 504),
    allowed_methods: frozenset | None = None,
) -> requests.Session:
    if allowed_methods is None:
        allowed_methods = frozenset(["HEAD", "GET", "OPTIONS"])  # idempotent
    retry = Retry(
        total=total,
        read=total,
        connect=total,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=allowed_methods,
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = LoggingHTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


__all__ = ["create_retry_session", "LoggingHTTPAdapter"]
