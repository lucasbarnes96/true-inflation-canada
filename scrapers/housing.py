"""Housing category scraper — StatCan CPI table 18-10-0004 CSV download.

Downloads the full CPI table as a ZIP/CSV from Statistics Canada and
extracts the latest Shelter and Rent index values for Canada.
Falls back gracefully if the download fails.
"""
from __future__ import annotations

import csv
import io
import urllib.request
import zipfile
from datetime import datetime, timezone

from .common import utc_now_iso, USER_AGENT
from .types import Quote, SourceHealth

# StatCan WDS returns a download URL for this table
STATCAN_CSV_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/18100004-eng.zip"

# Products we care about for the housing category
TARGET_PRODUCTS = {"Shelter", "Rented accommodation", "Owned accommodation"}


def scrape_housing() -> tuple[list[Quote], list[SourceHealth]]:
    quotes: list[Quote] = []
    health: list[SourceHealth] = []

    try:
        # Fetch the ZIP file directly as bytes
        req = urllib.request.Request(
            STATCAN_CSV_URL,
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            csv_name = next(
                name for name in zf.namelist() if name.endswith(".csv")
            )
            with zf.open(csv_name) as handle:
                decoded = io.TextIOWrapper(
                    handle, encoding="utf-8-sig", errors="ignore"
                )
                rows = list(csv.DictReader(decoded))

        # Find the latest value for each target product in Canada
        latest_by_product: dict[str, tuple[str, float]] = {}
        for row in rows:
            if row.get("GEO") != "Canada":
                continue
            product = (row.get("Products and product groups") or "").strip()
            if product not in TARGET_PRODUCTS:
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
                    category="housing",
                    item_id=product.lower().replace(" ", "_"),
                    value=value,
                    observed_at=observed,
                    source="statcan_cpi_csv",
                )
            )

        health.append(
            SourceHealth(
                source="statcan_cpi_csv",
                category="housing",
                tier=1,
                status="stale" if quotes else "missing",
                last_success_timestamp=utc_now_iso() if quotes else None,
                detail=f"Collected {len(quotes)} CPI housing proxies from StatCan CSV.",
                last_observation_period=latest_period,
            )
        )
    except Exception as err:
        health.append(
            SourceHealth(
                source="statcan_cpi_csv",
                category="housing",
                tier=1,
                status="missing",
                last_success_timestamp=None,
                detail=f"Fetch failed: {err}",
                last_observation_period=None,
            )
        )

    return quotes, health
