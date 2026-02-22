"""Free-source CPI consensus extraction.

Sources:
- TradingEconomics
- FXStreet economic calendar
"""
from __future__ import annotations

import re

from .common import fetch_url, utc_now_iso

TRADING_ECONOMICS_URL = "https://tradingeconomics.com/canada/inflation-cpi"
FXSTREET_CALENDAR_URL = "https://www.fxstreet.com/economic-calendar"


def _extract_percent_candidates(text: str) -> list[float]:
    vals: list[float] = []
    for raw in re.findall(r"(-?\d{1,2}(?:\.\d{1,2})?)\s*%", text):
        try:
            vals.append(float(raw))
        except ValueError:
            continue
    return vals


def fetch_consensus_estimate() -> dict:
    as_of = utc_now_iso()
    source_rows: list[dict] = []
    consensus_candidates: list[float] = []
    errors: list[str] = []

    for name, url in (
        ("tradingeconomics", TRADING_ECONOMICS_URL),
        ("fxstreet", FXSTREET_CALENDAR_URL),
    ):
        try:
            text = fetch_url(url, timeout=25, retries=1)
            percents = _extract_percent_candidates(text)
            inferred = None
            # Heuristic for headline YoY in plausible range.
            plausible = [v for v in percents if -5.0 <= v <= 15.0]
            if plausible:
                inferred = plausible[0]
                consensus_candidates.append(inferred)
            field_conf = "none"
            if inferred is not None:
                # Plausible Canada CPI YoY zone receives medium confidence.
                field_conf = "medium" if 1.0 <= inferred <= 5.0 else "low"
            source_rows.append(
                {
                    "source": name,
                    "url": url,
                    "retrieved_at": as_of,
                    "headline_yoy_candidate": inferred,
                    "field_confidence": field_conf,
                }
            )
        except Exception as err:  # pragma: no cover - network dependent
            errors.append(f"{name}: {err}")
            source_rows.append(
                {
                    "source": name,
                    "url": url,
                    "retrieved_at": as_of,
                    "headline_yoy_candidate": None,
                    "field_confidence": "none",
                }
            )

    headline_yoy = None
    confidence = "low"
    if consensus_candidates:
        headline_yoy = round(sum(consensus_candidates) / len(consensus_candidates), 3)
        confidence = "medium" if len(consensus_candidates) > 1 else "low"

    return {
        "as_of": as_of,
        "headline_yoy": headline_yoy,
        "headline_mom": None,
        "core_notes": "Free-source consensus; unofficial and may be delayed.",
        "source_count": sum(1 for row in source_rows if row.get("headline_yoy_candidate") is not None),
        "confidence": confidence,
        "sources": source_rows,
        "errors": errors,
        "method_version": "v1.6.0",
    }
