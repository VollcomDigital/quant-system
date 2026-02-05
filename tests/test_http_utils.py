from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import requests
from requests.adapters import HTTPAdapter

from src.utils.http import LoggingHTTPAdapter, create_retry_session


def test_logging_http_adapter_logs_success(monkeypatch, caplog):
    response = requests.Response()
    response.status_code = 200
    response.raw = SimpleNamespace(retries=SimpleNamespace(history=[1, 2]))

    def fake_send(self, request, **kwargs):
        return response

    monkeypatch.setattr(HTTPAdapter, "send", fake_send)

    logger = logging.getLogger("quant.http.test")
    adapter = LoggingHTTPAdapter(logger=logger, log_level=logging.INFO)
    with caplog.at_level(logging.INFO, logger=logger.name):
        req = SimpleNamespace(method="GET", url="https://example.com")
        got = adapter.send(req)

    assert got is response
    assert "http GET" in caplog.text
    assert "attempt=3" in caplog.text


def test_logging_http_adapter_logs_error(monkeypatch, caplog):
    def fake_send(self, request, **kwargs):
        raise requests.RequestException("boom")

    monkeypatch.setattr(HTTPAdapter, "send", fake_send)

    logger = logging.getLogger("quant.http.error")
    adapter = LoggingHTTPAdapter(logger=logger, log_level=logging.WARNING)
    with caplog.at_level(logging.WARNING, logger=logger.name):
        with pytest.raises(requests.RequestException):
            adapter.send(SimpleNamespace(method="POST", url="https://example.com"))

    assert "http-error" in caplog.text


def test_create_retry_session_builds_adapter():
    session = create_retry_session(total=2, backoff_factor=0.1)
    adapter = session.adapters["https://"]
    assert isinstance(adapter, LoggingHTTPAdapter)
    assert adapter.max_retries.total == 2
