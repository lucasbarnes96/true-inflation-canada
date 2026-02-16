"""Health and personal care category scraper using StatCan CPI table 18-10-0004."""
from __future__ import annotations

import csv
import io
import urllib.request
import zipfile
from datetime import datetime, timezone

from .common import USER_AGENT, utc_now_iso
from .types import Quote, SourceHealth

STATCAN_CPI_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/18100004-eng.zip"
TARGET_KEYWORDS = ["health and personal care", "personal care", "health care"]


def scrape_health_personal() -> tuple[list[Quote], list[SourceHealth]]:
    quotes: list[Quote] = []
    health: list[SourceHealth] = []
    try:
        req = urllib.request.Request(STATCAN_CPI_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            csv_name = next(name for name in zf.namelist() if name.endswith(".csv"))
            with zf.open(csv_name) as handle:
                decoded = io.TextIOWrapper(handle, encoding="utf-8-sig", errors="ignore")
                rows = list(csv.DictReader(decoded))

        latest_by_product: dict[str, tuple[str, float]] = {}
        for row in rows:
            if row.get("GEO") != "Canada":
                continue
            product = (row.get("Products and product groups") or "").strip()
            product_lower = product.lower()
            if not any(kw in product_lower for kw in TARGET_KEYWORDS):
                continue
            value_raw = row.get("VALUE")
            ref_date = row.get("REF_DATE")
            if not value_raw or not ref_date:
                continue
            try:
                value = float(value_raw)
            except ValueError:
                continue
            prev = latest_by_product.get(product)
            if prev is None or ref_date > prev[0]:
                latest_by_product[product] = (ref_date, value)

        observed = datetime.now(timezone.utc).date()
        latest_period = None
        for product, (period, value) in latest_by_product.items():
            latest_period = period if latest_period is None or period > latest_period else latest_period
            quotes.append(
                Quote(
                    category="health_personal",
                    item_id=product.lower().replace(" ", "_"),
                    value=value,
                    observed_at=observed,
                    source="statcan_cpi_csv",
                )
            )

        health.append(
            SourceHealth(
                source="statcan_cpi_csv",
                category="health_personal",
                tier=1,
                status="stale" if quotes else "missing",
                last_success_timestamp=utc_now_iso() if quotes else None,
                detail=f"Collected {len(quotes)} health/personal CPI proxies from StatCan CSV.",
                last_observation_period=latest_period,
            )
        )
    except Exception as err:
        health.append(
            SourceHealth(
                source="statcan_cpi_csv",
                category="health_personal",
                tier=1,
                status="missing",
                last_success_timestamp=None,
                detail=f"Fetch failed: {err}",
                last_observation_period=None,
            )
        )
    return quotes, health
