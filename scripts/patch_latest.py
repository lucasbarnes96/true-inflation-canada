import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from scrapers.energy_fuel import scrape_energy_fuel
from scrapers.housing_listings import scrape_housing_listings

def patch_latest():
    latest_path = Path("data/latest.json")
    if not latest_path.exists():
        print("data/latest.json not found. Cannot patch.")
        return

    print("Scraping fresh Pulse data...")
    # Get Energy data (works)
    e_quotes, _ = scrape_energy_fuel()
    gas_val = None
    if e_quotes:
        # Sort/find best
        for q in e_quotes:
            if q.item_id == "gasoline_regular_canada_avg":
                gas_val = q.value
                break
    
    # Get Housing data (fails gracefully, but we can try)
    h_quotes, _ = scrape_housing_listings()
    rent_val = None
    if h_quotes:
         for q in h_quotes:
            if q.item_id == "average_asking_rent_canada":
                rent_val = q.value
                break

    print(f"Got Gas: {gas_val}, Rent: {rent_val}")

    with open(latest_path, "r") as f:
        data = json.load(f)

    # Ensure meta exists
    if "meta" not in data:
        data["meta"] = {}

    # Inject indicators
    data["meta"]["indicators"] = {
        "average_asking_rent": rent_val,
        "gasoline_canada_avg": gas_val
    }

    with open(latest_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print("Successfully patched data/latest.json with Pulse indicators.")

if __name__ == "__main__":
    patch_latest()
