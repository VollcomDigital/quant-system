import logging

import pytest

from src.data.ratelimiter import RateLimiter


def test_rate_limiter_logs_sleep(monkeypatch, caplog):
    times = iter([1000.0, 1000.0, 1000.05, 1000.2, 1000.3, 1000.4])
    monkeypatch.setattr("src.data.ratelimiter.time.time", lambda: next(times))
    slept = []
    monkeypatch.setattr("src.data.ratelimiter.time.sleep", lambda duration: slept.append(duration))

    limiter = RateLimiter(min_interval=0.1)

    caplog.set_level(logging.DEBUG, logger="quant.rate_limiter")

    wait1 = limiter.acquire()
    wait2 = limiter.acquire()

    assert wait1 == pytest.approx(0.0)
    assert wait2 == pytest.approx(0.05)
    assert len(slept) == 1
    assert slept[0] == pytest.approx(0.05)
    assert "rate-limit" in caplog.text
    assert limiter.last_wait == pytest.approx(0.05)
