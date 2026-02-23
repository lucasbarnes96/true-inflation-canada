"""Supplemental health/personal proxies from public references."""
from __future__ import annotations

from datetime import datetime, timezone

from .common import fetch_url, parse_floats_from_text, utc_now_iso
from .types import Quote, SourceHealth

HEALTH_DPD_URL = "https://health-products.canada.ca/dpd-bdpp/index-eng.jsp"
PMPRB_REPORTS_URLS = (
    "https://www.pmprb-cepmb.gc.ca/en/reporting/market-intelligence",
    "https://www.pmprb-cepmb.gc.ca/en",
    "https://www.canada.ca/en/patented-medicine-prices-review/services/reports-studies.html",
)


def _fetch_health_source(url: str) -> tuple[str, str]:
    return url, fetch_url(url, timeout=20, retries=1)


def _fetch_pmprb_with_fallback() -> tuple[str, str]:
    last_error: Exception | None = None
    for url in PMPRB_REPORTS_URLS:
        try:
            return url, fetch_url(url, timeout=20, retries=1)
        except Exception as err:
            last_error = err
    raise RuntimeError(f"All PMPRB URLs failed: {last_error}")


def scrape_health_public() -> tuple[list[Quote], list[SourceHealth]]:
    quotes: list[Quote] = []
    health: list[SourceHealth] = []
    observed = datetime.now(timezone.utc).date()

    for source in ("healthcanada_dpd", "pmprb_reports"):
        try:
            fetched_url, html = (
                _fetch_pmprb_with_fallback() if source == "pmprb_reports" else _fetch_health_source(HEALTH_DPD_URL)
            )
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
                    status="fresh" if values else "missing",
                    last_success_timestamp=utc_now_iso() if values else None,
                    detail=f"Collected {len(values)} supplemental health/personal points from {fetched_url}.",
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
