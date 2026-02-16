"""Supplemental communication price proxies from public telecom trackers."""
from __future__ import annotations

from datetime import datetime, timezone

from .common import fetch_url, parse_floats_from_text, utc_now_iso
from .types import Quote, SourceHealth

ISED_MOBILE_PLANS_URL = "https://ised-isde.canada.ca/site/mobile-plans/en"
CRTC_CMR_URL = "https://crtc.gc.ca/eng/publications/reports/policymonitoring/2024/index.htm"


def scrape_communication_public() -> tuple[list[Quote], list[SourceHealth]]:
    quotes: list[Quote] = []
    health: list[SourceHealth] = []
    observed = datetime.now(timezone.utc).date()

    for source, url in (
        ("ised_mobile_plan_tracker", ISED_MOBILE_PLANS_URL),
        ("crtc_cmr_report", CRTC_CMR_URL),
    ):
        try:
            html = fetch_url(url, timeout=20, retries=1)
            values = [v for v in parse_floats_from_text(html) if 10 <= v <= 200][:8]
            for idx, value in enumerate(values):
                quotes.append(
                    Quote(
                        category="communication",
                        item_id=f"{source}_{idx}",
                        value=value,
                        observed_at=observed,
                        source=source,
                    )
                )
            health.append(
                SourceHealth(
                    source=source,
                    category="communication",
                    tier=2,
                    status="stale" if values else "missing",
                    last_success_timestamp=utc_now_iso() if values else None,
                    detail=f"Collected {len(values)} supplemental communication points.",
                    last_observation_period=None,
                )
            )
        except Exception as err:
            health.append(
                SourceHealth(
                    source=source,
                    category="communication",
                    tier=2,
                    status="missing",
                    last_success_timestamp=None,
                    detail=f"Fetch failed: {err}",
                    last_observation_period=None,
                )
            )

    return quotes, health

