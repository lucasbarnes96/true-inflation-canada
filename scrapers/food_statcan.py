"""StatCan retail food prices — Table 18-10-0245-01.

Monthly average retail prices for selected products by province.
Free CSV download, no API key required.
"""
from __future__ import annotations

import csv
import io
import urllib.request
import zipfile
from datetime import datetime, timezone

from .common import utc_now_iso, USER_AGENT
from .types import Quote, SourceHealth

STATCAN_FOOD_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/18100245-eng.zip"

# Target food items that serve as good basket proxies
TARGET_FOOD_ITEMS = {
    "Eggs, 1 dozen",
    "Milk, partly skimmed (2%), 2 litres",
    "Butter, 454 grams",
    "Bread, white, 675 grams",
    "Chicken, whole",
    "Ground beef, regular",
    "Apples, per kilogram",
    "Bananas, per kilogram",
    "Potatoes, per kilogram",
    "Tomatoes, per kilogram",
    "Onions, per kilogram",
    "Rice, white, 2 kilograms",
    "Sugar, white, 2 kilograms",
    "Flour, white, all purpose, 2.5 kilograms",
    "Canned soup, 284 millilitres",
    "Macaroni or spaghetti, 500 grams",
    "Orange juice, frozen concentrate, 355 millilitres",
    "Wieners, 450 grams",
    "Bacon, 500 grams",
    "Cheddar cheese, 250 grams",
    "Evaporated milk, 385 millilitres",
    "Coffee, roasted, 300 grams",
}


def scrape_food_statcan() -> tuple[list[Quote], list[SourceHealth]]:
    """Scrape official monthly retail food prices from StatCan."""
    quotes: list[Quote] = []
    health: list[SourceHealth] = []

    try:
        req = urllib.request.Request(
            STATCAN_FOOD_URL,
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=120) as response:
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

        # Find the latest price for each target product in Canada
        latest_by_product: dict[str, tuple[str, float]] = {}
        for row in rows:
            if row.get("GEO") != "Canada":
                continue
            product = (row.get("Products") or "").strip()
            if product not in TARGET_FOOD_ITEMS:
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
            # Create a clean item_id from the product name
            item_id = product.lower().split(",")[0].strip().replace(" ", "_")
            quotes.append(
                Quote(
                    category="food",
                    item_id=item_id,
                    value=value,
                    observed_at=observed,
                    source="statcan_food_prices",
                )
            )

        health.append(
            SourceHealth(
                source="statcan_food_prices",
                category="food",
                tier=1,
                status="stale" if quotes else "missing",
                last_success_timestamp=utc_now_iso() if quotes else None,
                detail=f"Collected {len(quotes)} retail food price observations from StatCan.",
                last_observation_period=latest_period,
            )
        )
    except Exception as err:
        health.append(
            SourceHealth(
                source="statcan_food_prices",
                category="food",
                tier=1,
                status="missing",
                last_success_timestamp=None,
                detail=f"Fetch failed: {err}",
                last_observation_period=None,
            )
        )

    return quotes, health
