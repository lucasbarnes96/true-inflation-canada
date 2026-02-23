"""Energy fuel scraper — NRCAN Weekly Average Retail Prices.

Scrapes the Government of Canada (NRCAN) weekly average retail fuel prices.
While not "daily", it is an official and reliable weekly source that leads the monthly CPI.
This is a "Scrappy" implementation that parses the HTML table.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from .common import fetch_url, utc_now_iso
from .types import Quote, SourceHealth

# NRCAN Weekly Average Retail Prices (National)
# The URL displays a table. We want the most recent "Canada" average for "Regular" gasoline.
NRCAN_FUEL_URL = "https://www2.nrcan.gc.ca/eneene/sources/pripri/prices_bycity_e.cfm?productID=1&locationID=66,17,39,11,8,59,73,35,46,2,7,4,66&frequency=W&priceYear=0"


def scrape_energy_fuel() -> tuple[list[Quote], list[SourceHealth]]:
    quotes: list[Quote] = []
    health: list[SourceHealth] = []

    try:
        html = fetch_url(NRCAN_FUEL_URL)

        # NRCAN structure: Each week is a <tr>. The first <td> is the date.
        # We want the last row containing "headerDate" to get the latest week's data.
        rows = html.split("<tr")
        latest_row = None
        for row in rows:
            if 'headers="headerDate' in row:
                latest_row = row

        price_val = None
        if latest_row:
            # Canada is the first data column after the date.
            # headers="header4_1_1 header3_1 header1">139.6</td>
            match = re.search(r'<td[^>]*header3_1[^>]*>([\d\.]+)</td>', latest_row)
            if match:
                try:
                    price_val = float(match.group(1))
                except ValueError:
                    pass

            # Fallback if specific header fails: just take the first number cell after date
            if price_val is None:
                matches = re.findall(r'<td[^>]*>([\d\.]+)</td>', latest_row)
                for m in matches:
                    try:
                        val = float(m)
                        if 100.0 <= val <= 300.0:
                            price_val = val
                            break
                    except ValueError:
                        pass

        if price_val:
            observed = datetime.now(timezone.utc).date()
            quotes.append(
                Quote(
                    category="transport",
                    item_id="gasoline_regular_canada_avg",
                    value=price_val,
                    observed_at=observed,
                    source="nrcan_fuel_scrape",
                )
            )
            health.append(
                SourceHealth(
                    source="nrcan_fuel_scrape",
                    category="transport",
                    tier=2,
                    status="fresh",
                    last_success_timestamp=utc_now_iso(),
                    detail=f"Parsed Canada average gas price {price_val} c/L from NRCAN.",
                    last_observation_period=None,
                )
            )
        else:
            health.append(
                SourceHealth(
                    source="nrcan_fuel_scrape",
                    category="transport",
                    tier=2,
                    status="missing",
                    last_success_timestamp=None,
                    detail="Could not parse a valid price pattern from NRCAN HTML.",
                    last_observation_period=None,
                )
            )

    except Exception as err:
        msg = str(err)
        if "403" in msg:
            msg = "Blocked by anti-bot protection (403 Forbidden)"

        health.append(
            SourceHealth(
                source="nrcan_fuel_scrape",
                category="transport",
                tier=2,
                status="missing",
                last_success_timestamp=None,
                detail=f"Fetch failed: {msg}",
                last_observation_period=None,
            )
        )

    return quotes, health
