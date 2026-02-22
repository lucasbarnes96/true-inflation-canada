"""Supplemental communication price proxies from public telecom trackers."""
from __future__ import annotations

from datetime import datetime, timezone

from .common import fetch_url, parse_floats_from_text, utc_now_iso
from .types import Quote, SourceHealth

ISED_MOBILE_PLANS_URL = "https://ised-isde.canada.ca/site/mobile-plans/en"
CRTC_CMR_URLS = [
    "https://crtc.gc.ca/eng/publications/reports/policymonitoring/2025/index.htm",
    "https://crtc.gc.ca/eng/publications/reports/policymonitoring/2024/index.htm",
    "https://crtc.gc.ca/eng/publications/reports/policymonitoring/2023/index.htm",
]
CRTC_INSECURE_HOSTS = {"crtc.gc.ca"}


def scrape_communication_public() -> tuple[list[Quote], list[SourceHealth]]:
    quotes: list[Quote] = []
    health: list[SourceHealth] = []
    observed = datetime.now(timezone.utc).date()

    for source, url in (("ised_mobile_plan_tracker", ISED_MOBILE_PLANS_URL),):
        try:
            verify = True
            html = fetch_url(
                url,
                timeout=20,
                retries=1,
                verify=verify,
                allowed_insecure_hosts=CRTC_INSECURE_HOSTS if not verify else None,
            )
            values = [v for v in parse_floats_from_text(html) if 10 <= v <= 200][:8]
            mode_note = " TLS verify disabled for pinned CRTC host." if not verify else ""
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
                    detail=f"Collected {len(values)} supplemental communication points.{mode_note}",
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

    crtc_errors: list[str] = []
    for url in CRTC_CMR_URLS:
        try:
            html = fetch_url(
                url,
                timeout=20,
                retries=1,
                verify=False,
                allowed_insecure_hosts=CRTC_INSECURE_HOSTS,
            )
            values = [v for v in parse_floats_from_text(html) if 10 <= v <= 200][:8]
            mode_note = " TLS verify disabled for pinned CRTC host."
            for idx, value in enumerate(values):
                quotes.append(
                    Quote(
                        category="communication",
                        item_id=f"crtc_cmr_report_{idx}",
                        value=value,
                        observed_at=observed,
                        source="crtc_cmr_report",
                    )
                )
            health.append(
                SourceHealth(
                    source="crtc_cmr_report",
                    category="communication",
                    tier=2,
                    status="stale" if values else "missing",
                    last_success_timestamp=utc_now_iso() if values else None,
                    detail=f"Collected {len(values)} supplemental communication points from {url}.{mode_note}",
                    last_observation_period=None,
                )
            )
            break
        except Exception as err:
            crtc_errors.append(f"{url}: {err}")
    else:
        health.append(
            SourceHealth(
                source="crtc_cmr_report",
                category="communication",
                tier=2,
                status="missing",
                last_success_timestamp=None,
                detail="Fetch failed across CRTC URLs: " + " | ".join(crtc_errors[:3]),
                last_observation_period=None,
            )
        )

    return quotes, health
