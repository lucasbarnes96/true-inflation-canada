"""Supplemental recreation/education proxies from public fee datasets/pages."""
from __future__ import annotations

from datetime import datetime, timezone

from .common import fetch_url, parse_floats_from_text, utc_now_iso
from .types import Quote, SourceHealth

PARKS_FEES_URL = "https://parks.canada.ca/voyage-travel/tarifs-fees"
STATCAN_EDU_PORTAL_URL = "https://www.statcan.gc.ca/en/subjects-start/education_training_and_learning"


def scrape_recreation_education_public() -> tuple[list[Quote], list[SourceHealth]]:
    quotes: list[Quote] = []
    health: list[SourceHealth] = []
    observed = datetime.now(timezone.utc).date()

    for source, url in (
        ("parkscanada_fees", PARKS_FEES_URL),
        ("statcan_education_portal", STATCAN_EDU_PORTAL_URL),
    ):
        try:
            html = fetch_url(url, timeout=20, retries=1)
            values = [v for v in parse_floats_from_text(html) if 1 <= v <= 1000][:10]
            for idx, value in enumerate(values):
                quotes.append(
                    Quote(
                        category="recreation_education",
                        item_id=f"{source}_{idx}",
                        value=value,
                        observed_at=observed,
                        source=source,
                    )
                )
            health.append(
                SourceHealth(
                    source=source,
                    category="recreation_education",
                    tier=2,
                    status="stale" if values else "missing",
                    last_success_timestamp=utc_now_iso() if values else None,
                    detail=f"Collected {len(values)} supplemental recreation/education points.",
                    last_observation_period=None,
                )
            )
        except Exception as err:
            health.append(
                SourceHealth(
                    source=source,
                    category="recreation_education",
                    tier=2,
                    status="missing",
                    last_success_timestamp=None,
                    detail=f"Fetch failed: {err}",
                    last_observation_period=None,
                )
            )

    return quotes, health

