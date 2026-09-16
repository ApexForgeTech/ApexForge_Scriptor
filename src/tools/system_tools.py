"""Safe local system-information tools used by the assistant.

This module deliberately exposes only deterministic, read-only information.
It never executes arbitrary shell commands supplied by a model or user.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from config import settings

_AZ_WEEKDAYS = (
    "Bazar ertəsi", "Çərşənbə axşamı", "Çərşənbə", "Cümə axşamı",
    "Cümə", "Şənbə", "Bazar",
)
_AZ_MONTHS = (
    "yanvar", "fevral", "mart", "aprel", "may", "iyun",
    "iyul", "avqust", "sentyabr", "oktyabr", "noyabr", "dekabr",
)


def now() -> datetime:
    """Return the current local datetime using the configured IANA timezone."""
    try:
        timezone = ZoneInfo(settings.timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown timezone: {settings.timezone}") from exc
    return datetime.now(timezone)


def current_time(language: str = "az") -> str:
    """Return a human-readable current time, date, and weekday.

    ``language`` currently supports Azerbaijani (the default) and English.
    """
    moment = now()
    if language == "en":
        return (
            f"Current local time: {moment:%H:%M}. Today is "
            f"{moment:%A, %B %-d, %Y} ({settings.timezone})."
        )
    return (
        f"İndi saat {moment:%H:%M}-dır. Bu gün "
        f"{moment.day} {_AZ_MONTHS[moment.month - 1]} {moment.year}, "
        f"{_AZ_WEEKDAYS[moment.weekday()]} günüdür ({settings.timezone})."
    )


def temporal_context() -> str:
    """A compact factual timestamp that can safely be included in RAG prompts."""
    moment = now()
    return f"Current local date and time: {moment:%Y-%m-%d %H:%M} ({settings.timezone})."
