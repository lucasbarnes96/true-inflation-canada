import sys
import os
sys.path.append(os.getcwd())

from scrapers.housing_listings import scrape_housing_listings
from scrapers.energy_fuel import scrape_energy_fuel
import time

def test():
    print("Testing Housing Listings Scraper...")
    t0 = time.time()
    try:
        quotes, health = scrape_housing_listings()
        print(f"Housing: {len(quotes)} quotes, {len(health)} health records. Time: {time.time()-t0:.2f}s")
        for q in quotes:
            print(f"  - {q.item_id}: {q.value} ({q.source})")
        for h in health:
             print(f"  Health: {h.status} - {h.detail}")
    except Exception as e:
        print(f"Housing Failed: {e}")

    print("\nTesting Energy Fuel Scraper...")
    # DEBUG: Fetch and print snippet to see what we got regardless of scraper success
    try:
        from scrapers.common import fetch_url
        from scrapers.energy_fuel import NRCAN_FUEL_URL
        html = fetch_url(NRCAN_FUEL_URL)
        idx = html.find("Canada")
        print(f"DEBUG SNIPPET: {html[idx:idx+500]}")
    except Exception as err:
        print(f"DEBUG FETCH FAILED: {err}")

    t0 = time.time()
    try:
        quotes, health = scrape_energy_fuel()
        print(f"Energy: {len(quotes)} quotes, {len(health)} health records. Time: {time.time()-t0:.2f}s")
        for q in quotes:
            print(f"  - {q.item_id}: {q.value} ({q.source})")
        for h in health:
             print(f"  Health: {h.status} - {h.detail}")
    except Exception as e:
        print(f"Energy Failed: {e}")
        # DEBUG: Fetch and print snippet to see what we got
        try:
            from scrapers.common import fetch_url, NRCAN_FUEL_URL
            html = fetch_url(NRCAN_FUEL_URL)
            idx = html.find("Canada")
            print(f"DEBUG SNIPPET: {html[idx:idx+500]}")
        except:
            pass

if __name__ == "__main__":
    test()
