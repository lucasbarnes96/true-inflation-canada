"""Supplemental health/personal proxies from public references."""
from __future__ import annotations

from datetime import datetime, timezone

from .common import fetch_url, parse_floats_from_text, utc_now_iso
from .types import Quote, SourceHealth

HEALTH_DPD_URL = "https://health-products.canada.ca/dpd-bdpp/index-eng.jsp"
PMPRB_REPORTS_URL = "https://www.canada.ca/en/patented-medicine-prices-review/services/reports-studies.html"


def scrape_health_public() -> tuple[list[Quote], list[SourceHealth]]:
    quotes: list[Quote] = []
    health: list[SourceHealth] = []
    observed = datetime.now(timezone.utc).date()

    for source, url in (
        ("healthcanada_dpd", HEALTH_DPD_URL),
        ("pmprb_reports", PMPRB_REPORTS_URL),
    ):
        try:
            html = fetch_url(url, timeout=20, retries=1)
            values = [v for v in parse_floats_from_text(html) if 1 <= v <= 500][:8]
            for idx, value in enumerate(values):
                quotes.append(
                    Quote(
                        category="health_personal",
                        item_id=f"{source}_{idx}",
                        value=value,
                        observed_at=observed,
                        source=source,
                    )
                )
            health.append(
                SourceHealth(
                    source=source,
                    category="health_personal",
                    tier=2,
                    status="stale" if values else "missing",
                    last_success_timestamp=utc_now_iso() if values else None,
                    detail=f"Collected {len(values)} supplemental health/personal points.",
                    last_observation_period=None,
                )
            )
        except Exception as err:
            health.append(
                SourceHealth(
                    source=source,
                    category="health_personal",
                    tier=2,
                    status="missing",
                    last_success_timestamp=None,
                    detail=f"Fetch failed: {err}",
                    last_observation_period=None,
                )
            )

    return quotes, health

