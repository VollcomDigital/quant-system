from __future__ import annotations

import json
import logging
from typing import Any
from urllib import error, request

from ..backtest.runner import BestResult
from ..config import NotificationsConfig, SlackNotificationConfig


def notify_all(
    results: list[BestResult],
    notifications: NotificationsConfig | None,
    run_id: str,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not notifications:
        return events
    if notifications.slack:
        events.append(_notify_slack(results, notifications.slack, run_id))
    return events


def _notify_slack(
    results: list[BestResult], slack_cfg: SlackNotificationConfig, run_id: str
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "channel": "slack",
        "metric": slack_cfg.metric,
        "sent": False,
    }
    if not results:
        event["reason"] = "no_results"
        return event

    metric_target = slack_cfg.metric.lower()
    filtered = [r for r in results if r.metric_name.lower() == metric_target]
    candidates = filtered or results
    if not candidates:
        event["reason"] = "no_candidates"
        return event

    best = max(candidates, key=lambda r: r.metric_value)
    event.update(
        {
            "collection": best.collection,
            "symbol": best.symbol,
            "strategy": best.strategy,
            "timeframe": best.timeframe,
            "value": best.metric_value,
        }
    )

    if slack_cfg.threshold is not None and best.metric_value < slack_cfg.threshold:
        event["reason"] = "below_threshold"
        return event

    text_lines = [
        f":tada: {best.collection} / {best.symbol} — {best.strategy} ({best.timeframe})",
        f"{best.metric_name.upper()} = {best.metric_value:.4f}",
        f"Params: {best.params}",
        f"Run ID: {run_id}",
    ]
    payload = {
        "text": "\n".join(text_lines),
    }
    if slack_cfg.username:
        payload["username"] = slack_cfg.username
    if slack_cfg.channel:
        payload["channel"] = slack_cfg.channel

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        slack_cfg.webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        request.urlopen(req, timeout=10)
        event["sent"] = True
        return event
    except error.URLError as exc:  # pragma: no cover - network failure path
        logging.warning("slack_notification_failed", exc_info=exc)
        event["reason"] = str(exc)
        return event
