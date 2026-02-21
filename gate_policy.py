from __future__ import annotations

from copy import deepcopy

METHOD_VERSION = "v1.3.0"

GATE_POLICY: dict = {
    "apify_max_age_days": 14,
    "required_sources": ["statcan_cpi_csv", "statcan_gas_csv"],
    "energy_required_any_of": ["oeb_scrape", "statcan_energy_cpi_csv"],
    "category_min_points": {
        "food": 5,
        "housing": 2,
        "transport": 1,
        "energy": 1,
        "communication": 1,
        "health_personal": 1,
        "recreation_education": 1,
    },
    "metadata_required": ["official_cpi.latest_release_month"],
    "representativeness_min_fresh_ratio": 0.85,
    "food_gate": {
        "min_fresh_sources": 1,
        "min_usable_sources": 2,
        "preferred_source": "apify_loblaws",
    },
    "apify_retry": {
        "max_attempts": 2,
        "backoff_seconds": 2,
    },
}


def gate_policy_payload() -> dict:
    return deepcopy(GATE_POLICY)
