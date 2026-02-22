from __future__ import annotations

from copy import deepcopy

METHOD_VERSION = "v1.3.1"

BASKET_WEIGHTS: dict[str, float] = {
    "housing": 0.2941,
    "food": 0.1691,
    "transport": 0.1690,
    "recreation_education": 0.1012,
    "energy": 0.0800,
    "health_personal": 0.0505,
    "communication": 0.0350,
}

WEIGHTS_METADATA: dict = {
    "source_table": "Statistics Canada Table 18-10-0007-01",
    "source_url": "https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000701",
    "analysis_reference": "Statistics Canada 62F0014M2025003",
    "analysis_url": "https://www150.statcan.gc.ca/n1/pub/62f0014m/62f0014m2025003-eng.htm",
    "basket_reference_year": 2024,
    "effective_month": "2025-05",
    "tracked_weights": BASKET_WEIGHTS,
    "tracked_share_total": round(sum(BASKET_WEIGHTS.values()), 4),
    "omitted_components": [
        {
            "component": "household_operations_furnishings_equipment",
            "weight_estimate": 0.1325,
            "status": "omitted_in_v1",
            "rationale": "Daily representative scraping remains noisy and high maintenance.",
        },
        {
            "component": "clothing_footwear",
            "weight_estimate": 0.0440,
            "status": "omitted_in_v1",
            "rationale": "High SKU churn and promotion-driven volatility can distort daily proxies.",
        },
        {
            "component": "alcohol_tobacco_cannabis",
            "weight_estimate": 0.0400,
            "status": "omitted_in_v1",
            "rationale": "Regulated pricing cadence is less informative for daily nowcasting.",
        },
    ],
    "mapping_notes": [
        "communication is represented as a proxy mapped within broader official CPI components.",
        "energy is tracked as a high-signal proxy across shelter and transport-related cost pressures.",
    ],
}

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
    "min_source_pass_rate_30d": 0.95,
    "max_imputed_weight_ratio": 0.15,
    "max_cross_source_disagreement_score": {
        "food": 0.35,
        "housing": 0.25,
        "transport": 0.30,
        "energy": 0.35,
        "communication": 0.25,
        "health_personal": 0.25,
        "recreation_education": 0.25,
    },
}


def gate_policy_payload() -> dict:
    return deepcopy(GATE_POLICY)


def weights_payload() -> dict:
    return deepcopy(WEIGHTS_METADATA)
