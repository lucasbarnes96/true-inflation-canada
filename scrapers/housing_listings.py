"""Housing listings scraper — Rentals.ca National Rent Report.

Parses the monthly "National Rent Report" from Rentals.ca to get the latest
"average asking rent" for all property types in Canada. This is a leading indicator
compared to the official StatCan CPI (which tracks paid rents, including rent-controlled units).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from .common import fetch_url, utc_now_iso
from .types import Quote, SourceHealth

# The report URL is stable
RENTALS_CA_URL = "https://rentals.ca/national-rent-report"


def scrape_housing_listings() -> tuple[list[Quote], list[SourceHealth]]:
    quotes: list[Quote] = []
    health: list[SourceHealth] = []
    
    try:
        html = fetch_url(RENTALS_CA_URL)
        
        # We are looking for a string like:
        # "average asking rent for all residential property types in Canada increased 9.3% year-over-year to $2,178"
        # or "average asking rent in Canada reached $2,193"
        # This is "scrappy" text parsing.
        
        # Regex to find "$2,xxx" near "average asking rent"
        # Matches: "average asking rent... $2,000" (simplistic but robust for this specific report structure)
        
        # Look for the specific "National Overview" section or just the first occurence of the national average
        # Usually in the first few paragraphs.
        
        # Pattern:  $X,XXX
        price_pattern = re.compile(r"\$(\d{1,2}(?:,\d{3}))")
        
        # Find all prices
        prices = price_pattern.findall(html)
        
        # The first few prices in the text are usually the national average.
        # We need to be careful not to pick up an ad or a specific city.
        # The report usually starts with "The average asking rent for all residential property types in Canada..."
        
        # refined strategy: Look for the paragraph containing "average asking rent" and "Canada"
        # and extract the price from *that* paragraph.
        
        candidates = []
        for paragraph in html.split("<p>"):
            clean_p = paragraph.split("</p>")[0]
            if "average asking rent" in clean_p.lower() and "canada" in clean_p.lower():
                # Extract price
                match = price_pattern.search(clean_p)
                if match:
                    val_str = match.group(1).replace(",", "")
                    try:
                        val = float(val_str)
                        if 1000 < val < 4000: # Sanity check for broad monthly rent range
                            candidates.append(val)
                    except ValueError:
                        pass
        
        if not candidates:
             # Fallback: just look for the first reasonable price in the whole body (scrappy!)
             for p in prices:
                 val_str = p.replace(",", "")
                 try:
                     val = float(val_str)
                     if 1500 < val < 3500: # Tighter sanity check for national average
                         candidates.append(val)
                         break # Take the first one (usually the headline)
                 except ValueError:
                     pass

        if candidates:
            # We take the first one as it's likely the headline number
            rent_value = candidates[0]
            
            observed = datetime.now(timezone.utc).date()
            
            # Rent is "housing" category
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
                    tier=2, # Tier 2 because it's a scraper
                    status="fresh",
                    last_success_timestamp=utc_now_iso(),
                    detail=f"Parsed average asking rent ${rent_value} from Rentals.ca report.",
                    last_observation_period=None, # It's "current"
                )
            )
        else:
             health.append(
                SourceHealth(
                    source="rentals_ca_scrape",
                    category="housing",
                    tier=2,
                    status="missing",
                    last_success_timestamp=None,
                    detail="Could not find 'average asking rent' in Rentals.ca HTML.",
                    last_observation_period=None,
                )
            )

    except Exception as err:
        msg = str(err)
        status = "missing"
        if "403" in msg:
            msg = "Blocked by anti-bot protection (403 Forbidden)"

        health.append(
            SourceHealth(
                source="rentals_ca_scrape",
                category="housing",
                tier=2,
                status=status,
                last_success_timestamp=None,
                detail=f"Fetch failed: {msg}",
                last_observation_period=None,
            )
        )

    return quotes, health
