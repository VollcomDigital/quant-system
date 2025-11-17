from __future__ import annotations

import json
from types import SimpleNamespace

from src.config import NotificationsConfig, SlackNotificationConfig
from src.reporting.notifications import notify_all


class DummyResult(SimpleNamespace):
    pass


def _best_result(metric_value: float) -> DummyResult:
    return DummyResult(
        collection="crypto",
        symbol="BTC/USDT",
        timeframe="1d",
        strategy="stratA",
        params={"x": 1},
        metric_name="sharpe",
        metric_value=metric_value,
    )


def test_slack_notification_sent(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=10):
        captured["payload"] = json.loads(req.data.decode())
        return SimpleNamespace()

    monkeypatch.setattr("src.reporting.notifications.request.urlopen", fake_urlopen)

    cfg = NotificationsConfig(
        slack=SlackNotificationConfig(
            webhook_url="https://slack.example/hook",
            metric="sharpe",
            threshold=1.0,
        )
    )
    events = notify_all([_best_result(1.5)], cfg, run_id="run-1")
    assert events[0]["sent"] is True
    assert "stratA" in captured["payload"]["text"]


def test_slack_notification_below_threshold(monkeypatch):
    monkeypatch.setattr(
        "src.reporting.notifications.request.urlopen", lambda *args, **kwargs: SimpleNamespace()
    )
    cfg = NotificationsConfig(
        slack=SlackNotificationConfig(
            webhook_url="https://slack.example/hook",
            metric="sharpe",
            threshold=2.0,
        )
    )
    events = notify_all([_best_result(1.5)], cfg, run_id="run-1")
    assert events[0]["sent"] is False
    assert events[0]["reason"] == "below_threshold"
