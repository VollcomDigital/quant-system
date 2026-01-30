from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def sanitize_for_json(data: Any) -> Any:
    if is_dataclass(data):
        data = asdict(data)
    if isinstance(data, dict):
        return {key: sanitize_for_json(value) for key, value in data.items()}
    if isinstance(data, (list, tuple, set)):
        return [sanitize_for_json(value) for value in data]
    if isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    if isinstance(data, Path):
        return str(data)
    return data


def safe_json_dumps(data: Any, **kwargs: Any) -> str:
    sanitized = sanitize_for_json(data)
    return json.dumps(sanitized, default=json_default, allow_nan=False, **kwargs)
