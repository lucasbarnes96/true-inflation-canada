"""Energy category scraper — OEB rates + StatCan energy fallback.

Combines:
1. OEB (Ontario Energy Board) rate scraping (existing, fragile)
2. StatCan monthly CPI energy aggregate fallback (reliable)
"""
from __future__ import annotations

import csv
import io
import urllib.request
import zipfile
from datetime import datetime, timezone

from .common import fetch_url, parse_floats_from_text, utc_now_iso, USER_AGENT
from .types import Quote, SourceHealth

OEB_RATES_URL = (
    "https://www.oeb.ca/consumer-information-and-protection/electricity-rates"
)

STATCAN_CPI_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/18100004-eng.zip"


def _scrape_oeb() -> tuple[list[Quote], list[SourceHealth]]:
    """Scrape OEB electricity rates (Ontario-specific, fragile)."""
    quotes: list[Quote] = []
    health: list[SourceHealth] = []
    try:
        html = fetch_url(OEB_RATES_URL)
        values = [v for v in parse_floats_from_text(html) if 1 <= v <= 50]
        values = values[:12]
        observed = datetime.now(timezone.utc).date()
        for i, value in enumerate(values):
            quotes.append(
                Quote(
                    category="energy",
                    item_id=f"oeb_rate_{i}",
                    value=value,
                    observed_at=observed,
                    source="oeb_scrape",
                )
            )
        health.append(
            SourceHealth(
                source="oeb_scrape",
                category="energy",
                tier=2,
                status="fresh" if quotes else "missing",
                last_success_timestamp=utc_now_iso() if quotes else None,
                detail=f"Collected {len(quotes)} OEB rate observations.",
                last_observation_period=None,
            )
        )
    except Exception as err:
        health.append(
            SourceHealth(
                source="oeb_scrape",
                category="energy",
                tier=2,
                status="missing",
                last_success_timestamp=None,
                detail=f"OEB fetch failed: {err}",
                last_observation_period=None,
            )
        )
    return quotes, health


def _scrape_statcan_energy() -> tuple[list[Quote], list[SourceHealth]]:
    """Monthly StatCan CPI energy aggregate fallback."""
    quotes: list[Quote] = []
    health: list[SourceHealth] = []
    try:
        req = urllib.request.Request(
            STATCAN_CPI_URL,
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=45) as response:
            data = response.read()

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            csv_name = next(name for name in zf.namelist() if name.endswith(".csv"))
            with zf.open(csv_name) as handle:
                decoded = io.TextIOWrapper(handle, encoding="utf-8-sig", errors="ignore")
                rows = list(csv.DictReader(decoded))

        latest_ref = None
        latest_val = None
        for row in rows:
            if row.get("GEO") != "Canada":
                continue
            product = (row.get("Products and product groups") or "").strip().lower()
            if product != "energy":
                continue
            value_raw = row.get("VALUE")
            ref_date = row.get("REF_DATE")
            if not value_raw or not ref_date:
                continue
            try:
                value = float(value_raw)
            except ValueError:
                continue
            if latest_ref is None or ref_date > latest_ref:
                latest_ref = ref_date
                latest_val = value

        if latest_val is not None:
            quotes.append(
                Quote(
                    category="energy",
                    item_id="statcan_energy_index",
                    value=latest_val,
                    observed_at=datetime.now(timezone.utc).date(),
                    source="statcan_energy_cpi_csv",
                )
            )

        health.append(
            SourceHealth(
                source="statcan_energy_cpi_csv",
                category="energy",
                tier=1,
                status="stale" if quotes else "missing",
                last_success_timestamp=utc_now_iso() if quotes else None,
                detail=f"Collected {len(quotes)} StatCan energy CPI fallback points.",
                last_observation_period=latest_ref,
            )
        )
    except Exception as err:
        health.append(
            SourceHealth(
                source="statcan_energy_cpi_csv",
                category="energy",
                tier=1,
                status="missing",
                last_success_timestamp=None,
                detail=f"StatCan energy fallback failed: {err}",
                last_observation_period=None,
            )
        )
    return quotes, health


def scrape_energy() -> tuple[list[Quote], list[SourceHealth]]:
    """Scrape energy data from multiple sources."""
    all_quotes: list[Quote] = []
    all_health: list[SourceHealth] = []

    # Try both active sources, accumulate results.
    # NOTE: legacy IESO HOEP endpoint was retired and removed from active scraping.
    for scraper in (_scrape_oeb, _scrape_statcan_energy):
        q, h = scraper()
        all_quotes.extend(q)
        all_health.extend(h)

    return all_quotes, all_health
