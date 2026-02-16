"""StatCan CPI release calendar ingestion.

Primary source: StatCan release calendar.
Fallback: deterministic seeded upcoming CPI event used when network parsing fails.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from .common import fetch_url, utc_now_iso

STATCAN_CALENDAR_URL = "https://www150.statcan.gc.ca/n1/dai-quo/cal2-eng.htm"


def _to_utc_iso_from_et(date_str: str, hour: int = 8, minute: int = 30) -> str:
    # Treat ET as UTC-5 baseline for deterministic output in this lightweight pipeline.
    dt_et = datetime.fromisoformat(f"{date_str}T{hour:02d}:{minute:02d}:00")
    dt_utc = dt_et.replace(tzinfo=timezone(timedelta(hours=-5))).astimezone(timezone.utc)
    return dt_utc.replace(microsecond=0).isoformat()


def fetch_release_events() -> dict:
    fetched_at = utc_now_iso()
    events: list[dict] = []
    errors: list[str] = []
    try:
        html = fetch_url(STATCAN_CALENDAR_URL, timeout=30, retries=1)
        # Minimal resilient matcher: capture dates adjacent to CPI label.
        # Format example in page text often includes yyyy-mm-dd.
        cpi_chunks = re.findall(r"(?:Consumer Price Index|Indice des prix.*?)(.{0,400})", html, flags=re.IGNORECASE | re.DOTALL)
        dates: list[str] = []
        for chunk in cpi_chunks:
            dates.extend(re.findall(r"(20\d{2}-\d{2}-\d{2})", chunk))
        # Keep unique sorted
        for day in sorted(set(dates)):
            events.append(
                {
                    "event_date": day,
                    "release_at_et": f"{day} 08:30 ET",
                    "release_at_utc": _to_utc_iso_from_et(day),
                    "series": "Canada CPI",
                    "source_url": STATCAN_CALENDAR_URL,
                    "status": "scheduled",
                }
            )
    except Exception as err:  # pragma: no cover - network dependent
        errors.append(str(err))

    if not events:
        # Seeded fallback for continuity (known date for Jan-2026 CPI publication).
        fallback_day = "2026-02-17"
        events = [
            {
                "event_date": fallback_day,
                "release_at_et": f"{fallback_day} 08:30 ET",
                "release_at_utc": _to_utc_iso_from_et(fallback_day),
                "series": "Canada CPI",
                "source_url": STATCAN_CALENDAR_URL,
                "status": "scheduled_fallback",
            }
        ]

    return {
        "as_of": fetched_at,
        "source_url": STATCAN_CALENDAR_URL,
        "events": events,
        "errors": errors,
    }

