"""Shared pydantic base class and validators."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator


class Schema(BaseModel):
    """Base for every cross-package contract."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        frozen=True,
        extra="forbid",
        validate_assignment=True,
    )


def require_aware(value: datetime) -> datetime:
    """Reject naive datetimes - every contract timestamp must be tz-aware."""
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware (UTC preferred)")
    return value


def aware_datetime_validator(field: str):  # noqa: ANN201 - used as decorator factory
    return field_validator(field)(lambda cls, v: require_aware(v))
