"""Housing listings scraper — Rentals.ca National Rent Report.

Parses Rentals.ca content to extract the latest national average asking rent.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from .common import fetch_url, utc_now_iso
from .types import Quote, SourceHealth

# Prefer official Rentals domains. Google Web Cache was retired and is no longer reliable.
RENTALS_CA_URLS = [
    "https://press.rentals.ca/releases/national-rent-report",
    "https://rentals.ca/national-rent-report",
]


def scrape_housing_listings() -> tuple[list[Quote], list[SourceHealth]]:
    quotes: list[Quote] = []
    health: list[SourceHealth] = []
    price_pattern = re.compile(r"\$(\d{1,2}(?:,\d{3}))")
    errors: list[str] = []

    for url in RENTALS_CA_URLS:
        try:
            html = fetch_url(url)
        except Exception as err:  # pragma: no cover - network dependent
            msg = str(err)
            if "403" in msg:
                msg = "Blocked by anti-bot protection (403 Forbidden)"
            errors.append(f"{url}: {msg}")
            continue

        candidates: list[float] = []
        for paragraph in html.split("<p>"):
            clean_p = paragraph.split("</p>")[0]
            lower = clean_p.lower()
            if "average asking rent" in lower and "canada" in lower:
                match = price_pattern.search(clean_p)
                if not match:
                    continue
                try:
                    val = float(match.group(1).replace(",", ""))
                except ValueError:
                    continue
                if 1000 < val < 4000:
                    candidates.append(val)

        if not candidates:
            prices = price_pattern.findall(html)
            for p in prices:
                try:
                    val = float(p.replace(",", ""))
                except ValueError:
                    continue
                if 1500 < val < 3500:
                    candidates.append(val)
                    break

        if not candidates:
            errors.append(f"{url}: could not find average asking rent pattern")
            continue

        rent_value = candidates[0]
        observed = datetime.now(timezone.utc).date()
        quotes.append(
            Quote(
                category="housing",
                item_id="average_asking_rent_canada",
                value=rent_value,
                observed_at=observed,
                source="rentals_ca_scrape",
            )
        )
        health.append(
            SourceHealth(
                source="rentals_ca_scrape",
                category="housing",
                tier=2,
                status="fresh",
                last_success_timestamp=utc_now_iso(),
                detail=f"Parsed average asking rent ${rent_value} from {url}.",
                last_observation_period=None,
            )
        )
        return quotes, health

    health.append(
        SourceHealth(
            source="rentals_ca_scrape",
            category="housing",
            tier=2,
            status="missing",
            last_success_timestamp=None,
            detail="Fetch failed across Rentals sources: " + " | ".join(errors[:3]) if errors else "No Rentals sources available.",
            last_observation_period=None,
        )
    )

    return quotes, health
