import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests
from requests.adapters import HTTPAdapter

from src.utils.http import LoggingHTTPAdapter, create_retry_session


def test_create_retry_session():
    s = create_retry_session()
    # Ensure adapters are mounted
    assert "http://" in s.adapters
    assert "https://" in s.adapters
    assert isinstance(s.get_adapter("https://"), LoggingHTTPAdapter)


def test_logging_adapter_logs_success(caplog):
    adapter = LoggingHTTPAdapter()
    request = requests.Request("GET", "https://example.com").prepare()
    fake_response = requests.Response()
    fake_response.status_code = 200
    fake_response.raw = SimpleNamespace(retries=SimpleNamespace(history=[1, 2]))

    with caplog.at_level(logging.DEBUG, logger="quant.http"):
        with patch.object(HTTPAdapter, "send", return_value=fake_response):
            adapter.send(request)

    assert "attempt=3" in caplog.text
    assert "-> 200" in caplog.text


def test_logging_adapter_logs_errors(caplog):
    adapter = LoggingHTTPAdapter()
    request = requests.Request("GET", "https://example.com").prepare()

    with caplog.at_level(logging.DEBUG, logger="quant.http"):
        with patch.object(HTTPAdapter, "send", side_effect=requests.ConnectionError("boom")):
            with pytest.raises(requests.ConnectionError):
                adapter.send(request)

    assert "http-error" in caplog.text
