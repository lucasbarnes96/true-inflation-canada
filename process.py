from __future__ import annotations

import calendar
import json
import sqlite3
import statistics
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from gate_policy import BASKET_WEIGHTS, GATE_POLICY, METHOD_VERSION, weights_payload
from models import NowcastSnapshot
from performance import compute_performance_summary, write_performance_summary
from scrapers.common import dns_preflight
from scrapers import (
    Quote,
    SourceHealth,
    fetch_consensus_estimate,
    fetch_release_events,
    fetch_boc_cpi,
    fetch_official_cpi_series,
    fetch_official_cpi_summary,
    scrape_communication,
    scrape_communication_public,
    scrape_energy,
    scrape_energy_fuel,
    scrape_food,
    scrape_food_statcan,
    scrape_grocery_apify,
    scrape_health_personal,
    scrape_health_public,
    scrape_housing,
    scrape_housing_listings,
    scrape_recreation_education,
    scrape_recreation_education_public,
    scrape_transport,
)

DATA_DIR = Path("data")
RUNS_DIR = DATA_DIR / "runs"
LATEST_PATH = DATA_DIR / "latest.json"
PUBLISHED_LATEST_PATH = DATA_DIR / "published_latest.json"
HISTORICAL_PATH = DATA_DIR / "historical.json"
RELEASE_DB_PATH = DATA_DIR / "releases.db"
PERFORMANCE_SUMMARY_PATH = DATA_DIR / "performance_summary.json"
MODEL_CARD_PATH = DATA_DIR / "model_card_latest.json"
RELEASE_EVENTS_PATH = DATA_DIR / "release_events.json"
CONSENSUS_LATEST_PATH = DATA_DIR / "consensus_latest.json"
SOURCE_CONTRACTS_PATH = DATA_DIR / "source_contracts.json"
QA_DB_PATH = DATA_DIR / "qa_runs.db"

CATEGORY_REGISTRY: dict[str, dict] = {
    "food": {
        "weight": BASKET_WEIGHTS["food"],
        "value_bounds": (0.1, 500.0),
        "outlier_threshold_pct": 60.0,
        "min_points": 5,
    },
    "housing": {
        "weight": BASKET_WEIGHTS["housing"],
        "value_bounds": (1.0, 400.0),
        "outlier_threshold_pct": 30.0,
        "min_points": 2,
    },
    "transport": {
        "weight": BASKET_WEIGHTS["transport"],
        "value_bounds": (50.0, 300.0),
        "outlier_threshold_pct": 40.0,
        "min_points": 1,
    },
    "energy": {
        "weight": BASKET_WEIGHTS["energy"],
        "value_bounds": (0.1, 100.0),
        "outlier_threshold_pct": 50.0,
        "min_points": 1,
    },
    "communication": {
        "weight": BASKET_WEIGHTS["communication"],
        "value_bounds": (1.0, 400.0),
        "outlier_threshold_pct": 30.0,
        "min_points": 1,
    },
    "health_personal": {
        "weight": BASKET_WEIGHTS["health_personal"],
        "value_bounds": (1.0, 400.0),
        "outlier_threshold_pct": 25.0,
        "min_points": 1,
    },
    "recreation_education": {
        "weight": BASKET_WEIGHTS["recreation_education"],
        "value_bounds": (1.0, 400.0),
        "outlier_threshold_pct": 30.0,
        "min_points": 1,
    },
}

CATEGORY_WEIGHTS = {name: cfg["weight"] for name, cfg in CATEGORY_REGISTRY.items()}
VALUE_BOUNDS = {name: cfg["value_bounds"] for name, cfg in CATEGORY_REGISTRY.items()}
OUTLIER_THRESHOLD_PCT = {name: cfg["outlier_threshold_pct"] for name, cfg in CATEGORY_REGISTRY.items()}
CATEGORY_MIN_POINTS = dict(GATE_POLICY["category_min_points"])

SCRAPER_REGISTRY = [
    ("food_openfoodfacts", scrape_food),
    ("food_statcan", scrape_food_statcan),
    ("food_apify", scrape_grocery_apify),
    ("transport_statcan", scrape_transport),
    ("transport_fuel_scrappy", scrape_energy_fuel),
    ("housing_statcan", scrape_housing),
    ("housing_listings_scrappy", scrape_housing_listings),
    ("energy_multi", scrape_energy),
    ("communication_statcan", scrape_communication),
    ("communication_public", scrape_communication_public),
    ("health_personal_statcan", scrape_health_personal),
    ("health_public", scrape_health_public),
    ("recreation_education_statcan", scrape_recreation_education),
    ("recreation_education_public", scrape_recreation_education_public),
]

SOURCE_SLA_DAYS = {
    "apify_loblaws": 14,
    "openfoodfacts_api": 2,
    "oeb_scrape": 2,
    "statcan_energy_cpi_csv": 45,
    "statcan_food_prices": 45,
    "statcan_gas_csv": 45,
    "nrcan_fuel_scrape": 8,
    "statcan_cpi_csv": 45,
    "rentals_ca_scrape": 45,
    "ised_mobile_plan_tracker": 60,
    "crtc_cmr_report": 400,
    "healthcanada_dpd": 90,
    "pmprb_reports": 400,
    "parkscanada_fees": 180,
    "statcan_education_portal": 180,
}

METHOD_LABEL = "YoY nowcast from public category proxies with month-to-date prorating"
CORE_GATE_CATEGORIES = ("food", "housing", "transport")
MIN_PLAUSIBLE_CONSENSUS_YOY = 1.0
MAX_PLAUSIBLE_CONSENSUS_YOY = 5.0
MAX_CONSENSUS_SPREAD_PCT = 1.0
SOURCE_TIER_MULTIPLIER = {1: 1.0, 2: 0.85, 3: 0.7}
SOURCE_STATUS_MULTIPLIER = {"fresh": 1.0, "stale": 0.9, "missing": 0.0}
HOUSING_RENT_BLEND_WEIGHT = 0.3
FORECAST_MIN_LIVE_DAYS = 30
CALIBRATION_MIN_LIVE_DAYS = 30
QA_RETRY_OFFSETS_MINUTES = [15, 60, 360]
CROSS_SOURCE_COMPARABLE_CATEGORIES = {"food", "transport"}

DEFAULT_SOURCE_CONTRACTS: dict[str, dict] = {
    "openfoodfacts_api": {
        "required_fields": ["category", "item_id", "value", "observed_at", "source"],
        "min_records": 25,
        "max_records": 500,
        "allowed_value_range": [0.1, 500.0],
        "max_stale_hours": 48.0,
        "max_daily_median_jump_pct": 20.0,
    },
    "statcan_food_prices": {
        "required_fields": ["category", "item_id", "value", "observed_at", "source"],
        "min_records": 1,
        "max_records": 500,
        "allowed_value_range": [0.1, 500.0],
        "max_stale_hours": 1100.0,
        "max_daily_median_jump_pct": 15.0,
    },
    "apify_loblaws": {
        "required_fields": ["category", "item_id", "value", "observed_at", "source"],
        "min_records": 0,
        "max_records": 1000,
        "allowed_value_range": [0.1, 500.0],
        "max_stale_hours": 336.0,
        "max_daily_median_jump_pct": 20.0,
    },
    "statcan_gas_csv": {
        "required_fields": ["category", "item_id", "value", "observed_at", "source"],
        "min_records": 1,
        "max_records": 300,
        "allowed_value_range": [50.0, 300.0],
        "max_stale_hours": 1100.0,
        "max_daily_median_jump_pct": 15.0,
    },
    "transport_fuel_scrappy": {
        "required_fields": ["category", "item_id", "value", "observed_at", "source"],
        "min_records": 1,
        "max_records": 300,
        "allowed_value_range": [50.0, 300.0],
        "max_stale_hours": 48.0,
        "max_daily_median_jump_pct": 20.0,
    },
    "nrcan_fuel_scrape": {
        "required_fields": ["category", "item_id", "value", "observed_at", "source"],
        "min_records": 1,
        "max_records": 300,
        "allowed_value_range": [50.0, 300.0],
        "max_stale_hours": 192.0,
        "max_daily_median_jump_pct": 20.0,
    },
    "statcan_cpi_csv": {
        "required_fields": ["category", "item_id", "value", "observed_at", "source"],
        "min_records": 3,
        "max_records": 500,
        "allowed_value_range": [1.0, 400.0],
        "max_stale_hours": 1100.0,
        "max_daily_median_jump_pct": 10.0,
    },
    "housing_listings_scrappy": {
        "required_fields": ["category", "item_id", "value", "observed_at", "source"],
        "min_records": 1,
        "max_records": 200,
        "allowed_value_range": [500.0, 10000.0],
        "max_stale_hours": 48.0,
        "max_daily_median_jump_pct": 15.0,
    },
    "rentals_ca_scrape": {
        "required_fields": ["category", "item_id", "value", "observed_at", "source"],
        "min_records": 1,
        "max_records": 200,
        "allowed_value_range": [500.0, 10000.0],
        "max_stale_hours": 1100.0,
        "max_daily_median_jump_pct": 15.0,
    },
    "oeb_scrape": {
        "required_fields": ["category", "item_id", "value", "observed_at", "source"],
        "min_records": 1,
        "max_records": 100,
        "allowed_value_range": [0.1, 100.0],
        "max_stale_hours": 48.0,
        "max_daily_median_jump_pct": 20.0,
    },
    "statcan_energy_cpi_csv": {
        "required_fields": ["category", "item_id", "value", "observed_at", "source"],
        "min_records": 1,
        "max_records": 200,
        "allowed_value_range": [50.0, 300.0],
        "max_stale_hours": 1100.0,
        "max_daily_median_jump_pct": 15.0,
    },
    "ised_mobile_plan_tracker": {
        "required_fields": ["category", "item_id", "value", "observed_at", "source"],
        "min_records": 1,
        "max_records": 200,
        "allowed_value_range": [1.0, 400.0],
        "max_stale_hours": 1440.0,
        "max_daily_median_jump_pct": 20.0,
    },
    "crtc_cmr_report": {
        "required_fields": ["category", "item_id", "value", "observed_at", "source"],
        "min_records": 0,
        "max_records": 200,
        "allowed_value_range": [1.0, 400.0],
        "max_stale_hours": 9600.0,
        "max_daily_median_jump_pct": 20.0,
    },
    "healthcanada_dpd": {
        "required_fields": ["category", "item_id", "value", "observed_at", "source"],
        "min_records": 1,
        "max_records": 200,
        "allowed_value_range": [1.0, 400.0],
        "max_stale_hours": 2160.0,
        "max_daily_median_jump_pct": 20.0,
    },
    "pmprb_reports": {
        "required_fields": ["category", "item_id", "value", "observed_at", "source"],
        "min_records": 1,
        "max_records": 200,
        "allowed_value_range": [1.0, 400.0],
        "max_stale_hours": 9600.0,
        "max_daily_median_jump_pct": 20.0,
    },
    "parkscanada_fees": {
        "required_fields": ["category", "item_id", "value", "observed_at", "source"],
        "min_records": 1,
        "max_records": 200,
        "allowed_value_range": [1.0, 400.0],
        "max_stale_hours": 4320.0,
        "max_daily_median_jump_pct": 20.0,
    },
    "statcan_education_portal": {
        "required_fields": ["category", "item_id", "value", "observed_at", "source"],
        "min_records": 1,
        "max_records": 200,
        "allowed_value_range": [1.0, 1000.0],
        "max_stale_hours": 4320.0,
        "max_daily_median_jump_pct": 20.0,
    },
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def round_or_none(value: float | None, places: int = 3) -> float | None:
    if value is None:
        return None
    return round(value, places)


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def source_age_days(last_success_timestamp: str | None, now: datetime | None = None) -> int | None:
    stamp = parse_iso_datetime(last_success_timestamp)
    if stamp is None:
        return None
    if now is None:
        now = utc_now()
    return max(0, (now.date() - stamp.date()).days)


def source_age_hours(last_success_timestamp: str | None, now: datetime | None = None) -> float | None:
    stamp = parse_iso_datetime(last_success_timestamp)
    if stamp is None:
        return None
    if now is None:
        now = utc_now()
    delta = now - stamp
    return round(max(0.0, delta.total_seconds() / 3600.0), 2)


def human_age(age_days: int | None) -> str:
    if age_days is None:
        return "unknown"
    if age_days == 0:
        return "updated today"
    if age_days == 1:
        return "updated 1 day ago"
    return f"updated {age_days} days ago"


def load_json(path: Path, default: dict | list) -> dict | list:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def load_historical() -> dict:
    data = load_json(HISTORICAL_PATH, {})
    return data if isinstance(data, dict) else {}


def load_source_contracts() -> dict[str, dict]:
    payload = load_json(SOURCE_CONTRACTS_PATH, DEFAULT_SOURCE_CONTRACTS)
    if not isinstance(payload, dict):
        return DEFAULT_SOURCE_CONTRACTS
    merged = dict(DEFAULT_SOURCE_CONTRACTS)
    for source, contract in payload.items():
        if isinstance(contract, dict):
            merged[source] = {**merged.get(source, {}), **contract}
    return merged


def load_previous_source_success() -> dict[str, str]:
    by_source: dict[str, str] = {}
    for path in (PUBLISHED_LATEST_PATH, LATEST_PATH):
        payload = load_json(path, {})
        if not isinstance(payload, dict):
            continue
        rows = payload.get("source_health", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            source = row.get("source")
            ts = row.get("last_success_timestamp")
            if isinstance(source, str) and isinstance(ts, str) and ts:
                if source not in by_source:
                    by_source[source] = ts
    return by_source


def dedupe_quotes(quotes: list[Quote]) -> list[Quote]:
    deduped: dict[str, Quote] = {}
    for quote in quotes:
        key = f"{quote.source}|{quote.item_id}|{quote.observed_at.isoformat()}"
        deduped[key] = quote
    return list(deduped.values())


def apply_range_checks(quotes: list[Quote]) -> tuple[list[Quote], int]:
    valid: list[Quote] = []
    rejected = 0
    for quote in quotes:
        bounds = VALUE_BOUNDS.get(quote.category)
        if bounds is None:
            continue
        lower, upper = bounds
        if quote.value <= 0 or quote.value < lower or quote.value > upper:
            rejected += 1
            continue
        valid.append(quote)
    return valid, rejected


def previous_category_median(historical: dict, category: str) -> float | None:
    if not historical:
        return None
    latest_day = sorted(historical.keys())[-1]
    value = historical.get(latest_day, {}).get("categories", {}).get(category, {}).get("proxy_level")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def apply_outlier_filter(quotes: list[Quote], historical: dict) -> tuple[list[Quote], int]:
    by_category: dict[str, list[Quote]] = defaultdict(list)
    for quote in quotes:
        by_category[quote.category].append(quote)

    kept: list[Quote] = []
    anomalies = 0
    for category, cat_quotes in by_category.items():
        median_today = statistics.median(q.value for q in cat_quotes)
        median_prev = previous_category_median(historical, category)
        if median_prev is None or median_prev <= 0:
            kept.extend(cat_quotes)
            continue

        delta_pct = abs((median_today / median_prev - 1) * 100)
        threshold = OUTLIER_THRESHOLD_PCT.get(category, 50.0)
        if delta_pct > threshold:
            anomalies += len(cat_quotes)
            continue

        kept.extend(cat_quotes)

    return kept, anomalies


def recompute_source_health(raw_health: list[SourceHealth], now: datetime) -> list[dict]:
    computed: list[dict] = []
    previous_success = load_previous_source_success()
    for entry in raw_health:
        payload = asdict(entry)
        ts = entry.last_success_timestamp
        if not ts:
            prev_ts = previous_success.get(entry.source)
            if prev_ts:
                ts = prev_ts
                payload["last_success_timestamp"] = prev_ts
                detail = payload.get("detail", "")
                payload["detail"] = f"{detail} Using last successful timestamp from prior run.".strip()

        age_days = source_age_days(ts, now=now)
        sla_days = SOURCE_SLA_DAYS.get(entry.source)
        if age_days is None:
            status = "missing"
        elif sla_days is not None and age_days <= sla_days:
            status = "fresh"
        else:
            status = "stale"

        payload["status"] = status
        payload["age_days"] = age_days
        payload["run_age_hours"] = source_age_hours(ts, now=now)
        payload["updated_days_ago"] = human_age(age_days)
        computed.append(payload)
    return computed


def source_contract_name(scraper_name: str, source_name: str) -> str:
    return source_name if source_name in DEFAULT_SOURCE_CONTRACTS else scraper_name


def validate_source_contract(
    contract_name: str,
    source_name: str,
    quotes: list[Quote],
    source_health: dict | None,
    historical: dict,
    now: datetime,
    source_contracts: dict[str, dict],
) -> dict:
    contract = source_contracts.get(contract_name) or {}
    checks: list[dict] = []
    required_fields = contract.get("required_fields", ["category", "item_id", "value", "observed_at", "source"])
    field_ok = True
    missing_field = None
    for required in required_fields:
        if not quotes:
            break
        if any(getattr(q, required, None) is None for q in quotes):
            field_ok = False
            missing_field = required
            break
    checks.append({"name": "required_fields", "passed": field_ok, "detail": missing_field})

    min_records = int(contract.get("min_records", 0))
    max_records = int(contract.get("max_records", 10_000))
    count = len(quotes)

    max_stale_hours = float(contract.get("max_stale_hours", 24 * 45))
    ts = source_health.get("last_success_timestamp") if isinstance(source_health, dict) else None
    if not ts:
        ts = load_previous_source_success().get(source_name)
    age_hours = source_age_hours(ts, now=now) if ts else None
    stale_within_sla = isinstance(age_hours, (int, float)) and float(age_hours) <= max_stale_hours

    count_ok = min_records <= count <= max_records
    if count == 0 and stale_within_sla:
        count_ok = True
    checks.append({"name": "record_count", "passed": count_ok, "detail": f"{count} in [{min_records}, {max_records}]"})

    value_bounds = contract.get("allowed_value_range") or [0.0, 10**9]
    low, high = float(value_bounds[0]), float(value_bounds[1])
    if quotes:
        range_ok = all(low <= float(q.value) <= high for q in quotes)
    else:
        range_ok = (min_records == 0) or stale_within_sla
    checks.append({"name": "value_range", "passed": range_ok, "detail": f"[{low}, {high}]"})

    freshness_ok = stale_within_sla if count > 0 else True
    checks.append({"name": "freshness", "passed": freshness_ok, "detail": f"{age_hours} <= {max_stale_hours}"})

    prev_median = None
    category = quotes[0].category if quotes else (source_health.get("category") if isinstance(source_health, dict) else None)
    if isinstance(category, str):
        prev_median = previous_category_median(historical, category)
    max_jump = float(contract.get("max_daily_median_jump_pct", 50.0))
    if quotes and prev_median not in (None, 0):
        current_median = statistics.median(float(q.value) for q in quotes)
        jump_pct = abs((current_median / float(prev_median) - 1) * 100)
        jump_ok = jump_pct <= max_jump
        jump_detail = round_or_none(jump_pct, 3)
    else:
        jump_ok = True
        jump_detail = None
    checks.append({"name": "median_jump", "passed": jump_ok, "detail": jump_detail})

    passed = all(check["passed"] for check in checks)
    return {
        "source": source_name,
        "contract": contract_name,
        "attempts": 1,
        "passed": passed,
        "checks": checks,
        "retry_schedule_minutes": QA_RETRY_OFFSETS_MINUTES,
    }


def retry_with_contracts(
    scraper_name: str,
    scraper,
    historical: dict,
    now: datetime,
    source_contracts: dict[str, dict],
) -> tuple[list[Quote], list[SourceHealth], list[dict]]:
    best_quotes: list[Quote] = []
    best_health: list[SourceHealth] = []
    checks_out: list[dict] = []
    max_attempts = len(QA_RETRY_OFFSETS_MINUTES) + 1
    previous_failure_signature: set[tuple[str, str]] | None = None
    for attempt in range(1, max_attempts + 1):
        quotes, health = scraper()
        best_quotes = quotes
        best_health = health
        source_rows = [asdict(h) for h in health]
        by_source_quotes: dict[str, list[Quote]] = defaultdict(list)
        for q in quotes:
            by_source_quotes[q.source].append(q)
        attempt_checks: list[dict] = []
        for row in source_rows:
            source = row.get("source")
            if not isinstance(source, str):
                continue
            contract_name = source_contract_name(scraper_name, source)
            check = validate_source_contract(
                contract_name=contract_name,
                source_name=source,
                quotes=by_source_quotes.get(source, []),
                source_health=row,
                historical=historical,
                now=now,
                source_contracts=source_contracts,
            )
            check["attempts"] = attempt
            attempt_checks.append(check)
        checks_out = attempt_checks
        if attempt_checks and all(c["passed"] for c in attempt_checks):
            break
        # Retry only on likely transient failures; skip repeated static failures.
        failed_pairs: set[tuple[str, str]] = set()
        transient_failed = False
        for check in attempt_checks:
            for item in check.get("checks", []):
                if item.get("passed"):
                    continue
                name = str(item.get("name"))
                failed_pairs.add((check.get("source", ""), name))
                if name in {"freshness", "required_fields"}:
                    transient_failed = True
        if not transient_failed:
            break
        if previous_failure_signature is not None and failed_pairs == previous_failure_signature:
            break
        previous_failure_signature = failed_pairs
    return best_quotes, best_health, checks_out


def reconcile_cross_source_quotes(
    quotes: list[Quote],
    computed_health: list[dict],
    historical: dict,
) -> tuple[list[Quote], dict]:
    by_category_source_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for quote in quotes:
        by_category_source_values[quote.category][quote.source].append(float(quote.value))

    max_disagreement_policy = GATE_POLICY.get("max_cross_source_disagreement_score", {})
    source_tier = {row.get("source"): int(row.get("tier") or 3) for row in computed_health if isinstance(row, dict)}
    official_sources = {
        "statcan_cpi_csv",
        "statcan_food_prices",
        "statcan_gas_csv",
        "statcan_energy_cpi_csv",
    }
    quarantine_sources: set[str] = set()
    disagreement_by_category: dict[str, float] = {}
    reasons: list[dict] = []

    for category, source_values in by_category_source_values.items():
        source_means = {src: statistics.mean(vals) for src, vals in source_values.items() if vals}
        if not source_means:
            continue
        if category not in CROSS_SOURCE_COMPARABLE_CATEGORIES:
            disagreement_by_category[category] = 0.0
            continue
        cat_threshold = float(max_disagreement_policy.get(category, 0.35))
        if len(source_means) >= 2:
            med = statistics.median(source_means.values())
            if med > 0:
                disagreements = {src: abs((val / med) - 1) for src, val in source_means.items()}
                max_disagreement = max(disagreements.values())
                disagreement_by_category[category] = round(max_disagreement, 4)
                if max_disagreement > cat_threshold:
                    worst_ranked = sorted(disagreements.items(), key=lambda item: item[1], reverse=True)
                    quarantine_candidate = None
                    for source, _score in worst_ranked:
                        if source in official_sources or source_tier.get(source, 3) <= 1:
                            continue
                        quarantine_candidate = source
                        break
                    if quarantine_candidate:
                        quarantine_sources.add(quarantine_candidate)
                        reasons.append(
                            {
                                "category": category,
                                "source": quarantine_candidate,
                                "reason": "cross_source_outlier",
                                "score": round_or_none(max_disagreement, 4),
                                "threshold": cat_threshold,
                            }
                        )
                    else:
                        disagreement_by_category[category] = round(cat_threshold * 0.95, 4)
                        reasons.append(
                            {
                                "category": category,
                                "source": None,
                                "reason": "official_source_disagreement_exempt",
                                "score": round_or_none(max_disagreement, 4),
                                "threshold": cat_threshold,
                            }
                        )
        else:
            source, value = next(iter(source_means.items()))
            prev = previous_category_median(historical, category)
            if prev not in (None, 0):
                jump = abs((value / float(prev)) - 1)
                disagreement_by_category[category] = round(jump, 4)
                threshold = min(cat_threshold, 0.15)
                if jump > threshold:
                    if source in official_sources or source_tier.get(source, 3) <= 1:
                        disagreement_by_category[category] = round(threshold * 0.95, 4)
                        reasons.append(
                            {
                                "category": category,
                                "source": source,
                                "reason": "single_source_official_exempt",
                                "score": round_or_none(jump, 4),
                                "threshold": threshold,
                            }
                        )
                    else:
                        quarantine_sources.add(source)
                        reasons.append(
                            {
                                "category": category,
                                "source": source,
                                "reason": "single_source_jump",
                                "score": round_or_none(jump, 4),
                                "threshold": threshold,
                            }
                        )

    filtered = [q for q in quotes if q.source not in quarantine_sources]
    for row in computed_health:
        if row.get("source") in quarantine_sources:
            row["status"] = "missing"
            detail = row.get("detail", "")
            row["detail"] = f"{detail} Quarantined by QA reconciliation.".strip()

    overall = max(disagreement_by_category.values()) if disagreement_by_category else 0.0
    return filtered, {
        "cross_source_disagreement_score": round_or_none(overall, 4),
        "cross_source_disagreement_by_category": disagreement_by_category,
        "quarantine_sources": sorted(quarantine_sources),
        "quarantine_reasons": reasons,
    }


def impute_missing_categories(categories: dict, historical: dict) -> dict:
    diagnostics = {
        "imputation_used": False,
        "imputed_categories": [],
        "imputed_weight_ratio": 0.0,
    }
    if not historical:
        return diagnostics

    ordered = sorted(historical.keys(), reverse=True)
    total_weight = sum(CATEGORY_WEIGHTS.values()) or 1.0
    imputed_weight = 0.0
    for category, payload in categories.items():
        if payload.get("proxy_level") is not None:
            continue

        imputed = None
        method = None
        for day in ordered[:3]:
            prev = historical.get(day, {}).get("categories", {}).get(category, {}).get("proxy_level")
            if isinstance(prev, (int, float)):
                imputed = float(prev)
                method = "last_good_decay"
                break

        if imputed is None:
            for day in ordered:
                prev = historical.get(day, {}).get("categories", {}).get(category, {}).get("proxy_level")
                if isinstance(prev, (int, float)):
                    imputed = float(prev)
                    method = "official_proxy_carry_forward"
                    break

        if imputed is None:
            continue

        payload["proxy_level"] = round_or_none(imputed, 4)
        payload["status"] = "stale"
        diagnostics["imputation_used"] = True
        diagnostics["imputed_categories"].append({"category": category, "method": method})
        imputed_weight += float(payload.get("weight", 0.0))

    diagnostics["imputed_weight_ratio"] = round_or_none(imputed_weight / total_weight, 4)
    return diagnostics


def source_effective_weight(source_row: dict | None) -> float:
    if not isinstance(source_row, dict):
        return 0.0
    tier = int(source_row.get("tier") or 3)
    status = source_row.get("status") or "missing"
    return SOURCE_TIER_MULTIPLIER.get(tier, 0.7) * SOURCE_STATUS_MULTIPLIER.get(status, 0.0)


def summarize_categories(quotes: list[Quote], source_health: list[dict]) -> tuple[dict, dict]:
    by_category: dict[str, list[Quote]] = defaultdict(list)
    for quote in quotes:
        by_category[quote.category].append(quote)

    source_by_name = {s["source"]: s for s in source_health}

    summary: dict[str, dict] = {}
    signal_inputs: dict[str, list[dict]] = {}
    for category, weight in CATEGORY_WEIGHTS.items():
        cat_quotes = by_category.get(category, [])
        per_source_values: dict[str, list[float]] = defaultdict(list)
        for quote in cat_quotes:
            per_source_values[quote.source].append(quote.value)

        weighted_sum = 0.0
        effective_weight = 0.0
        source_rows: list[dict] = []
        for source, values in per_source_values.items():
            source_row = source_by_name.get(source, {})
            source_weight = source_effective_weight(source_row)
            source_mean = statistics.mean(values)
            if source_weight > 0:
                weighted_sum += source_mean * source_weight
                effective_weight += source_weight
            source_rows.append(
                {
                    "source": source,
                    "status": source_row.get("status", "missing"),
                    "tier": source_row.get("tier", 3),
                    "points": len(values),
                    "source_mean": round_or_none(float(source_mean), 4),
                    "effective_weight": round_or_none(float(source_weight), 3),
                }
            )
        level = round_or_none(weighted_sum / effective_weight, 4) if effective_weight > 0 else None

        status = "missing"
        if cat_quotes:
            category_statuses = [h["status"] for h in source_health if h["category"] == category]
            status = "fresh" if "fresh" in category_statuses else "stale"

        summary[category] = {
            "proxy_level": level,
            "daily_change_pct": None,
            "weight": weight,
            "points": len(cat_quotes),
            "status": status,
        }
        signal_inputs[category] = sorted(source_rows, key=lambda row: row["source"])

    return summary, signal_inputs


def previous_indicator_value(key: str) -> float | None:
    for path in (PUBLISHED_LATEST_PATH, LATEST_PATH):
        payload = load_json(path, {})
        if not isinstance(payload, dict):
            continue
        value = payload.get("meta", {}).get("indicators", {}).get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def compute_daily_changes(categories: dict, historical: dict) -> None:
    if not historical:
        return

    latest_day = sorted(historical.keys())[-1]
    prev_categories = historical.get(latest_day, {}).get("categories", {})

    for category, payload in categories.items():
        current = payload.get("proxy_level")
        prev = prev_categories.get(category, {}).get("proxy_level")
        if current is None or prev in (None, 0):
            payload["daily_change_pct"] = None
            continue
        payload["daily_change_pct"] = round_or_none(((float(current) / float(prev)) - 1) * 100)


def apply_housing_signal_overlay(categories: dict, indicators: dict[str, float | None]) -> dict:
    diagnostics = {
        "applied": False,
        "blend_weight": HOUSING_RENT_BLEND_WEIGHT,
        "rent_delta_pct": None,
        "previous_rent": None,
        "current_rent": None,
        "reason": None,
    }
    current = indicators.get("average_asking_rent")
    previous = previous_indicator_value("average_asking_rent")
    diagnostics["current_rent"] = current
    diagnostics["previous_rent"] = previous
    if not isinstance(current, (int, float)) or not isinstance(previous, (int, float)) or previous <= 0:
        diagnostics["reason"] = "missing_or_invalid_indicator_history"
        return diagnostics

    rent_delta = ((float(current) / float(previous)) - 1) * 100
    rent_delta = max(min(rent_delta, 5.0), -5.0)
    diagnostics["rent_delta_pct"] = round_or_none(rent_delta, 4)

    housing = categories.get("housing")
    if not isinstance(housing, dict):
        diagnostics["reason"] = "housing_category_missing"
        return diagnostics

    base_change = housing.get("daily_change_pct")
    if base_change is None:
        housing["daily_change_pct"] = round_or_none(rent_delta, 4)
    else:
        blended = (float(base_change) * (1.0 - HOUSING_RENT_BLEND_WEIGHT)) + (rent_delta * HOUSING_RENT_BLEND_WEIGHT)
        housing["daily_change_pct"] = round_or_none(blended, 4)
    diagnostics["applied"] = True
    return diagnostics


def compute_category_contributions(categories: dict) -> dict:
    contributions: dict[str, float | None] = {}
    for category, payload in categories.items():
        change = payload.get("daily_change_pct")
        weight = payload.get("weight", 0.0)
        if change is None:
            contributions[category] = None
            continue
        contributions[category] = round_or_none(float(change) * float(weight), 4)
    return contributions


def compute_coverage(categories: dict) -> float:
    covered = 0.0
    for payload in categories.values():
        if payload["status"] in {"fresh", "stale"} and payload["proxy_level"] is not None:
            covered += payload["weight"]
    total = sum(CATEGORY_WEIGHTS.values())
    return round(covered / total, 4) if total else 0.0


def compute_representativeness(categories: dict) -> float:
    # Share of planned basket with fresh data only.
    fresh = 0.0
    total = sum(CATEGORY_WEIGHTS.values())
    for payload in categories.values():
        if payload["status"] == "fresh" and payload["proxy_level"] is not None:
            fresh += payload["weight"]
    return round(fresh / total, 4) if total else 0.0


def category_source_diversity(quotes: list[Quote]) -> dict[str, int]:
    by_category: dict[str, set[str]] = defaultdict(set)
    for quote in quotes:
        by_category[quote.category].add(quote.source)
    return {category: len(sources) for category, sources in by_category.items()}


def compute_nowcast_mom(categories: dict, historical: dict) -> float | None:
    weighted_change = 0.0
    effective_weight = 0.0
    for category, payload in categories.items():
        category_change = payload.get("daily_change_pct")
        weight = payload["weight"]
        if category_change is None:
            continue
        weighted_change += float(category_change) * weight
        effective_weight += weight

    if effective_weight == 0:
        return None

    normalized_change = weighted_change / effective_weight
    return round_or_none(normalized_change)


def month_key(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def prev_month(year: int, month: int) -> tuple[int, int]:
    if month == 1:
        return year - 1, 12
    return year, month - 1


def next_month(year: int, month: int) -> tuple[int, int]:
    if month == 12:
        return year + 1, 1
    return year, month + 1


def compute_nowcast_yoy_prorated(
    current_date: date,
    nowcast_mom_pct: float | None,
    official_series: list[dict],
) -> tuple[float | None, dict]:
    diagnostics = {
        "prorate_factor": None,
        "base_month": None,
        "reference_month": None,
        "projected_index": None,
        "base_index": None,
        "reference_index": None,
        "reason": None,
    }
    if nowcast_mom_pct is None:
        diagnostics["reason"] = "missing_nowcast_mom"
        return None, diagnostics

    by_month = {
        row.get("ref_date"): row
        for row in official_series
        if isinstance(row, dict) and isinstance(row.get("ref_date"), str)
    }
    ordered_months = sorted(k for k in by_month.keys() if isinstance(k, str))
    if not ordered_months:
        diagnostics["reason"] = "missing_required_official_index"
        return None, diagnostics

    base_y, base_m = prev_month(current_date.year, current_date.month)
    base_key = month_key(base_y, base_m)
    base = by_month.get(base_key)
    if base is None:
        base_key = ordered_months[-1]
        base = by_month.get(base_key)

    if base is None:
        diagnostics["reason"] = "missing_required_official_index"
        return None, diagnostics

    base_year = int(base_key[:4])
    base_month = int(base_key[5:7])
    projected_year, projected_month = next_month(base_year, base_month)
    ref_key = month_key(projected_year - 1, projected_month)
    reference = by_month.get(ref_key)

    diagnostics["base_month"] = base_key
    diagnostics["reference_month"] = ref_key
    if not base or not reference:
        diagnostics["reason"] = "missing_required_official_index"
        return None, diagnostics

    base_index = base.get("index_value")
    reference_index = reference.get("index_value")
    if base_index in (None, 0) or reference_index in (None, 0):
        diagnostics["reason"] = "invalid_required_official_index"
        return None, diagnostics

    if projected_year == current_date.year and projected_month == current_date.month:
        month_days = calendar.monthrange(current_date.year, current_date.month)[1]
        days_elapsed = current_date.day
        prorate_factor = days_elapsed / month_days
    else:
        prorate_factor = 1.0
    projected_index = float(base_index) * (1 + (float(nowcast_mom_pct) / 100.0) * prorate_factor)
    yoy = ((projected_index / float(reference_index)) - 1) * 100
    diagnostics["prorate_factor"] = round_or_none(prorate_factor, 4)
    diagnostics["projected_index"] = round_or_none(projected_index, 4)
    diagnostics["base_index"] = base_index
    diagnostics["reference_index"] = reference_index
    return round_or_none(yoy, 3), diagnostics


def apply_consensus_guardrails(consensus_payload: dict | None) -> tuple[float | None, dict]:
    diagnostics = {
        "accepted": False,
        "reason": None,
        "candidate_count": 0,
        "usable_count": 0,
        "spread": None,
    }
    if not isinstance(consensus_payload, dict):
        diagnostics["reason"] = "missing_payload"
        return None, diagnostics

    sources = consensus_payload.get("sources")
    if not isinstance(sources, list):
        diagnostics["reason"] = "missing_sources"
        return None, diagnostics

    candidates: list[float] = []
    for row in sources:
        if not isinstance(row, dict):
            continue
        candidate = row.get("headline_yoy_candidate")
        field_confidence = row.get("field_confidence")
        if field_confidence not in {"medium", "high"}:
            continue
        if not isinstance(candidate, (int, float)):
            continue
        value = float(candidate)
        if MIN_PLAUSIBLE_CONSENSUS_YOY <= value <= MAX_PLAUSIBLE_CONSENSUS_YOY:
            candidates.append(value)

    diagnostics["usable_count"] = len(candidates)
    diagnostics["candidate_count"] = sum(
        1
        for row in sources
        if isinstance(row, dict) and isinstance(row.get("headline_yoy_candidate"), (int, float))
    )
    if len(candidates) < 2:
        diagnostics["reason"] = "insufficient_high_conf_sources"
        return None, diagnostics

    spread = max(candidates) - min(candidates)
    diagnostics["spread"] = round_or_none(spread, 3)
    if spread > MAX_CONSENSUS_SPREAD_PCT:
        diagnostics["reason"] = "candidate_spread_too_wide"
        return None, diagnostics

    diagnostics["accepted"] = True
    return round(sum(candidates) / len(candidates), 3), diagnostics


def derive_lead_signal(nowcast_mom: float | None) -> str:
    if nowcast_mom is None:
        return "insufficient_data"
    if nowcast_mom > 0.02:
        return "up"
    if nowcast_mom < -0.02:
        return "down"
    return "flat"


def compute_next_release(events_payload: dict, now: datetime) -> dict | None:
    events = events_payload.get("events", []) if isinstance(events_payload, dict) else []
    if not isinstance(events, list):
        return None
    upcoming: list[dict] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        release_utc = parse_iso_datetime(event.get("release_at_utc"))
        if release_utc is None:
            continue
        if release_utc >= now:
            entry = dict(event)
            entry["_release_dt"] = release_utc
            upcoming.append(entry)
    if not upcoming:
        return None
    next_event = sorted(upcoming, key=lambda x: x["_release_dt"])[0]
    remaining = next_event["_release_dt"] - now
    seconds = int(max(0, remaining.total_seconds()))
    next_event["countdown_seconds"] = seconds
    next_event["status"] = "upcoming"
    next_event.pop("_release_dt", None)
    return next_event


def count_live_nowcast_days(historical: dict) -> int:
    count = 0
    for payload in historical.values():
        if not isinstance(payload, dict):
            continue
        meta = payload.get("meta", {})
        if isinstance(meta, dict) and meta.get("seeded"):
            continue
        headline = payload.get("headline", {})
        if isinstance(headline, dict) and isinstance(headline.get("nowcast_yoy_pct"), (int, float)):
            count += 1
    return count


def compute_forecast(
    nowcast_yoy: float | None,
    official_series: list[dict],
    consensus_yoy: float | None,
    historical: dict,
    next_release: dict | None,
    as_of: datetime,
) -> dict:
    forecast = {
        "status": "insufficient_history",
        "next_release_date": next_release.get("event_date") if isinstance(next_release, dict) else None,
        "as_of": as_of.isoformat(),
        "model_version": "v1.3.0-forecast-baseline",
        "confidence": "low",
        "point_yoy": None,
        "lower_yoy": None,
        "upper_yoy": None,
        "backtest": {"live_days": count_live_nowcast_days(historical), "minimum_required_days": FORECAST_MIN_LIVE_DAYS},
        "inputs": {
            "nowcast_yoy": nowcast_yoy,
            "consensus_yoy": consensus_yoy,
            "official_recent_yoy": None,
            "official_yoy_momentum": None,
        },
    }
    live_days = forecast["backtest"]["live_days"]
    if live_days < FORECAST_MIN_LIVE_DAYS:
        return forecast

    official_yoy_values = [
        float(row["yoy_pct"])
        for row in official_series
        if isinstance(row, dict) and isinstance(row.get("yoy_pct"), (int, float))
    ]
    if len(official_yoy_values) < 2:
        forecast["status"] = "insufficient_official_history"
        return forecast

    latest_official_yoy = official_yoy_values[-1]
    momentum = latest_official_yoy - official_yoy_values[-2]
    forecast["inputs"]["official_recent_yoy"] = round_or_none(latest_official_yoy, 3)
    forecast["inputs"]["official_yoy_momentum"] = round_or_none(momentum, 3)

    if nowcast_yoy is None:
        if consensus_yoy is None:
            forecast["status"] = "insufficient_inputs"
            return forecast
        point = float(consensus_yoy)
    else:
        baseline = (0.6 * float(nowcast_yoy)) + (0.4 * (latest_official_yoy + momentum))
        point = baseline if consensus_yoy is None else (0.75 * baseline) + (0.25 * float(consensus_yoy))

    interval_half_width = 0.35 if live_days >= 60 else 0.5
    forecast["status"] = "published"
    forecast["confidence"] = "medium" if live_days >= 60 else "low"
    forecast["point_yoy"] = round_or_none(point, 3)
    forecast["lower_yoy"] = round_or_none(point - interval_half_width, 3)
    forecast["upper_yoy"] = round_or_none(point + interval_half_width, 3)
    return forecast


def calibrate_nowcast_yoy(
    raw_nowcast_yoy: float | None,
    official_series: list[dict],
    diversity_by_category: dict[str, int],
    categories: dict,
    anomalies: int,
) -> tuple[float | None, dict]:
    diagnostics = {
        "raw_nowcast_yoy": raw_nowcast_yoy,
        "calibrated_nowcast_yoy": raw_nowcast_yoy,
        "shrink_factor": 0.0,
        "anchor_yoy": None,
        "trigger_reason": None,
        "cap_applied": False,
    }
    if raw_nowcast_yoy is None:
        diagnostics["trigger_reason"] = "missing_raw_nowcast_yoy"
        return None, diagnostics

    official_yoy_values = [
        float(row["yoy_pct"])
        for row in official_series
        if isinstance(row, dict) and isinstance(row.get("yoy_pct"), (int, float))
    ]
    if len(official_yoy_values) < 2:
        diagnostics["trigger_reason"] = "insufficient_official_yoy_history"
        return raw_nowcast_yoy, diagnostics

    latest_official = official_yoy_values[-1]
    momentum = latest_official - official_yoy_values[-2]
    anchor = latest_official + (0.5 * momentum)
    diagnostics["anchor_yoy"] = round_or_none(anchor, 3)

    covered_categories = [
        category
        for category, payload in categories.items()
        if isinstance(payload, dict) and payload.get("status") in {"fresh", "stale"}
    ]
    weak_categories = [
        category for category in covered_categories if diversity_by_category.get(category, 0) < 2
    ]
    weak_ratio = (len(weak_categories) / len(covered_categories)) if covered_categories else 1.0
    shrink = 0.0
    reasons: list[str] = []
    if weak_ratio >= 0.4:
        shrink += 0.15
        reasons.append("low_source_diversity")
    if anomalies > 0:
        shrink += min(0.2, anomalies / 60.0)
        reasons.append("anomaly_penalty")
    shrink = min(0.35, shrink)
    diagnostics["shrink_factor"] = round_or_none(shrink, 3)
    diagnostics["trigger_reason"] = ",".join(reasons) if reasons else "none"

    calibrated = (float(raw_nowcast_yoy) * (1 - shrink)) + (anchor * shrink)
    max_gap = 2.5
    gap = calibrated - anchor
    if abs(gap) > max_gap:
        calibrated = anchor + (max_gap if gap > 0 else -max_gap)
        diagnostics["cap_applied"] = True

    calibrated = round_or_none(calibrated, 3)
    diagnostics["calibrated_nowcast_yoy"] = calibrated
    return calibrated, diagnostics


def calibration_tier(live_days: int) -> str:
    if live_days < 30:
        return "bootstrapping"
    if live_days < 60:
        return "developing"
    return "stable"


def build_calibration(
    historical: dict,
    performance_summary: dict,
    forecast: dict,
    calibration_diagnostics: dict,
) -> dict:
    live_days = count_live_nowcast_days(historical)
    tier = calibration_tier(live_days)
    forecast_status = forecast.get("status") if isinstance(forecast, dict) else "unknown"
    eligible = forecast_status == "published"
    reason = None if eligible else f"forecast_status={forecast_status}"
    return {
        "maturity_tier": tier,
        "live_days": live_days,
        "minimum_days_for_stable_eval": CALIBRATION_MIN_LIVE_DAYS,
        "forecast_eligibility": {"eligible": eligible, "reason": reason},
        "current_error_metrics": {
            "mae_yoy_pct": performance_summary.get("mae_yoy_pct"),
            "median_abs_error_yoy_pct": performance_summary.get("median_abs_error_yoy_pct"),
            "directional_accuracy_yoy_pct": performance_summary.get("directional_accuracy_yoy_pct"),
            "bias_yoy_pct": performance_summary.get("bias_yoy_pct"),
            "evaluated_live_points": performance_summary.get("evaluated_live_points"),
        },
        "notes": [
            "Calibration applies moderate shrinkage toward recent official trend when diversity is weak or anomaly pressure is elevated.",
            "Early-stage metrics should be interpreted cautiously until live history depth increases.",
        ],
        "calibration_diagnostics": calibration_diagnostics,
    }


def compute_signal_quality_score(
    coverage_ratio: float,
    anomalies: int,
    blocked_conditions: list[str],
    diversity_by_category: dict[str, int],
    categories: dict,
) -> int:
    score = int(round(coverage_ratio * 100))

    if blocked_conditions:
        score -= 35

    if anomalies > 0:
        score -= min(20, anomalies)

    if any(v.get("status") == "missing" for v in categories.values()):
        score -= 10

    # Penalize if covered categories rely on a single source.
    weak = 0
    for category, payload in categories.items():
        if payload.get("status") in {"fresh", "stale"} and diversity_by_category.get(category, 0) < 2:
            weak += 1
    score -= min(20, weak * 4)

    return max(0, min(100, score))


def compute_confidence(
    coverage_ratio: float,
    anomalies: int,
    blocked_conditions: list[str],
    diversity_by_category: dict[str, int] | None = None,
    categories: dict | None = None,
) -> str:
    if blocked_conditions:
        return "low"

    if coverage_ratio >= 0.9:
        confidence = "high"
    elif coverage_ratio >= 0.6:
        confidence = "medium"
    else:
        confidence = "low"

    if anomalies > 0 and confidence == "high":
        confidence = "medium"
    elif anomalies > 0 and confidence == "medium":
        confidence = "low"

    if diversity_by_category and categories:
        for category, payload in categories.items():
            if payload.get("status") in {"fresh", "stale"} and diversity_by_category.get(category, 0) < 2:
                if confidence == "high":
                    return "medium"

    return confidence


def compute_top_driver(contributions: dict) -> dict:
    best_category: str | None = None
    best_contribution: float | None = None
    for category, contribution in contributions.items():
        if contribution is None:
            continue
        if best_contribution is None or abs(float(contribution)) > abs(float(best_contribution)):
            best_category = category
            best_contribution = float(contribution)

    if best_category is None:
        return {"category": None, "contribution_pct": None}
    return {
        "category": best_category,
        "contribution_pct": round_or_none(best_contribution, 4),
    }


def build_notes(
    categories: dict,
    anomalies: int,
    rejected_points: int,
    blocked_conditions: list[str],
    diversity_by_category: dict[str, int],
    representativeness_ratio: float,
    collection_diagnostics: dict | None = None,
) -> list[str]:
    notes: list[str] = [
        "This is an experimental nowcast estimate and not an official CPI release.",
        f"Methodology {METHOD_VERSION}: weighted category proxies with month-to-date YoY projection.",
        "Confidence rubric: gate status + weighted coverage + anomaly rate + source diversity.",
        "Model applies minimal smoothing with calibration guardrails to reduce tail errors while preserving real-time signal responsiveness.",
        "Housing includes an asking-rent momentum overlay from Rentals.ca and is not survey-equivalent to transacted-rent CPI methods.",
        f"Representativeness (fresh-weight share): {round(representativeness_ratio * 100, 1)}%.",
        "Coverage ratio is the share of the CPI basket with usable source data in this run.",
    ]

    missing = [k for k, v in categories.items() if v["status"] == "missing"]
    stale = [k for k, v in categories.items() if v["status"] == "stale"]
    single_source = [
        category
        for category, payload in categories.items()
        if payload.get("status") in {"fresh", "stale"} and diversity_by_category.get(category, 0) < 2
    ]

    if missing:
        notes.append(f"Missing categories today: {', '.join(missing)}. Confidence is downgraded.")
    if stale:
        notes.append(f"Stale categories used: {', '.join(stale)}.")
    if single_source:
        notes.append(f"Source diversity warning: single-source categories today: {', '.join(single_source)}.")
    if rejected_points:
        notes.append(f"Dropped {rejected_points} points via range checks.")
    if anomalies:
        notes.append(f"Dropped {anomalies} points via day-over-day anomaly filter.")
    if blocked_conditions:
        notes.append("Release gate failed: " + "; ".join(blocked_conditions))
    signature = (collection_diagnostics or {}).get("failure_signature", {})
    if isinstance(signature, dict) and signature.get("global_dns_outage_likely"):
        notes.append("Network diagnostic: DNS resolver outage likely during collection; multiple sources failed host resolution.")

    return notes


def collect_all_quotes(
    historical: dict,
    now: datetime,
    source_contracts: dict[str, dict],
) -> tuple[list[Quote], list[SourceHealth], dict, list[dict]]:
    quotes: list[Quote] = []
    health: list[SourceHealth] = []
    qa_checks: list[dict] = []
    preflight = dns_preflight(ttl_seconds=60)
    diagnostics = {
        "apify_retry": {
            "attempts": 0,
            "retries_used": 0,
            "succeeded": False,
            "final_status": "missing",
            "reason": None,
        },
        "network_preflight": preflight,
        "qa_retry_offsets_minutes": QA_RETRY_OFFSETS_MINUTES,
        "source_contracts_enforced": True,
    }

    for scraper_name, scraper in SCRAPER_REGISTRY:
        scraper_quotes, scraper_health, checks = retry_with_contracts(
            scraper_name=scraper_name,
            scraper=scraper,
            historical=historical,
            now=now,
            source_contracts=source_contracts,
        )
        quotes.extend(scraper_quotes)
        health.extend(scraper_health)
        qa_checks.extend(checks)

    apify_idx = next((idx for idx, row in enumerate(health) if row.source == "apify_loblaws"), None)
    retry_cfg = GATE_POLICY.get("apify_retry", {})
    max_attempts = int(retry_cfg.get("max_attempts", 1))
    backoff_seconds = int(retry_cfg.get("backoff_seconds", 0))
    diagnostics["apify_retry"]["attempts"] = max_attempts
    if apify_idx is None:
        diagnostics["apify_retry"]["reason"] = "apify_source_not_registered"
        return quotes, health, diagnostics, qa_checks

    for attempt in range(2, max_attempts + 1):
        apify_health = health[apify_idx]
        if apify_health.status == "fresh":
            diagnostics["apify_retry"]["succeeded"] = True
            diagnostics["apify_retry"]["final_status"] = "fresh"
            break
        if backoff_seconds > 0:
            time.sleep(backoff_seconds)
        retry_quotes, retry_health = scrape_grocery_apify()
        diagnostics["apify_retry"]["retries_used"] = attempt - 1
        if retry_quotes:
            quotes.extend(retry_quotes)
        if retry_health:
            health[apify_idx] = retry_health[0]
            if retry_health[0].status == "fresh":
                diagnostics["apify_retry"]["succeeded"] = True
                diagnostics["apify_retry"]["final_status"] = "fresh"
                break
    else:
        apify_health = health[apify_idx]
        diagnostics["apify_retry"]["final_status"] = apify_health.status
        diagnostics["apify_retry"]["reason"] = apify_health.detail

    # Re-evaluate APIFY contract against final post-retry state so QA reflects final source truth.
    apify_quotes = [q for q in quotes if q.source == "apify_loblaws"]
    apify_row = asdict(health[apify_idx]) if apify_idx is not None and apify_idx < len(health) else None
    if isinstance(apify_row, dict):
        final_apify_check = validate_source_contract(
            contract_name=source_contract_name("food_apify", "apify_loblaws"),
            source_name="apify_loblaws",
            quotes=apify_quotes,
            source_health=apify_row,
            historical=historical,
            now=now,
            source_contracts=source_contracts,
        )
        final_apify_check["attempts"] = int(diagnostics["apify_retry"].get("retries_used", 0)) + 1
        qa_checks = [row for row in qa_checks if row.get("source") != "apify_loblaws"]
        qa_checks.append(final_apify_check)

    dns_failures = 0
    timeout_failures = 0
    blocked_failures = 0
    for row in health:
        detail = (row.detail or "").lower()
        if "dns resolver unavailable" in detail or "nodename nor servname" in detail or "name or service not known" in detail:
            dns_failures += 1
        if "timed out" in detail or "timeout" in detail:
            timeout_failures += 1
        if "403" in detail or "forbidden" in detail or "anti-bot" in detail:
            blocked_failures += 1
    diagnostics["failure_signature"] = {
        "dns_failures": dns_failures,
        "timeout_failures": timeout_failures,
        "blocked_failures": blocked_failures,
        "total_sources": len(health),
        "global_dns_outage_likely": bool(preflight.get("ok") is False and dns_failures > 0),
    }

    return quotes, health, diagnostics, qa_checks


def extract_hero_indicators(quotes: list[Quote]) -> dict[str, float | None]:
    """Extract specific high-value 'scrappy' indicators for sidebar display."""
    indicators = {
        "average_asking_rent": None,
        "gasoline_canada_avg": None,
    }
    # We look for the most recent value for specific item_ids
    # Sort by date descending
    sorted_quotes = sorted(quotes, key=lambda q: q.observed_at, reverse=True)
    
    for q in sorted_quotes:
        if q.item_id == "average_asking_rent_canada" and indicators["average_asking_rent"] is None:
            indicators["average_asking_rent"] = q.value
        if q.item_id == "gasoline_regular_canada_avg" and indicators["gasoline_canada_avg"] is None:
            indicators["gasoline_canada_avg"] = q.value
            
    return indicators


def count_food_sources(snapshot: dict, fresh_only: bool = False) -> int:
    sources = {
        row.get("source")
        for row in snapshot.get("source_health", [])
        if isinstance(row, dict)
        and row.get("category") == "food"
        and row.get("status") in ({"fresh"} if fresh_only else {"fresh", "stale"})
    }
    return len(sources)


def build_gate_diagnostics(snapshot: dict) -> dict:
    diagnostics: dict[str, dict] = {}
    source_by_name = {s["source"]: s for s in snapshot.get("source_health", []) if isinstance(s, dict)}
    policy = GATE_POLICY

    apify = source_by_name.get("apify_loblaws")
    apify_age = apify.get("age_days") if isinstance(apify, dict) else None
    apify_ok = bool(isinstance(apify_age, int) and apify_age <= int(policy["apify_max_age_days"]))
    diagnostics["apify_recency"] = {
        "passed": apify_ok,
        "value": apify_age,
        "threshold": policy["apify_max_age_days"],
        "reason": None if apify_ok else "apify_stale_or_missing",
    }

    required = []
    for source in policy["required_sources"]:
        state = source_by_name.get(source, {}).get("status")
        ok = state in {"fresh", "stale"}
        required.append({"source": source, "status": state, "passed": ok})
    diagnostics["required_sources"] = {
        "passed": all(item["passed"] for item in required),
        "value": required,
        "threshold": "all_required",
        "reason": None if all(item["passed"] for item in required) else "missing_required_source",
    }

    energy_states = [
        source_by_name.get(source, {}).get("status")
        for source in policy["energy_required_any_of"]
    ]
    energy_ok = any(state in {"fresh", "stale"} for state in energy_states)
    diagnostics["energy_source"] = {
        "passed": energy_ok,
        "value": energy_states,
        "threshold": "any_usable",
        "reason": None if energy_ok else "no_usable_energy_source",
    }

    point_checks: list[dict] = []
    for category, min_points in policy["category_min_points"].items():
        points = snapshot.get("categories", {}).get(category, {}).get("points", 0)
        point_checks.append(
            {
                "category": category,
                "points": points,
                "threshold": min_points,
                "passed": points >= min_points,
            }
        )
    diagnostics["category_points"] = {
        "passed": all(row["passed"] for row in point_checks),
        "value": point_checks,
        "threshold": "all_min_points",
        "reason": None if all(row["passed"] for row in point_checks) else "category_min_points_failed",
    }

    official_month = snapshot.get("official_cpi", {}).get("latest_release_month")
    diagnostics["official_metadata"] = {
        "passed": bool(official_month),
        "value": official_month,
        "threshold": "required",
        "reason": None if official_month else "missing_latest_release_month",
    }

    representativeness = snapshot.get("meta", {}).get("representativeness_ratio")
    rep_threshold = float(policy["representativeness_min_fresh_ratio"])
    rep_passed = isinstance(representativeness, (int, float)) and float(representativeness) >= rep_threshold
    diagnostics["representativeness"] = {
        "passed": rep_passed,
        "value": representativeness,
        "threshold": rep_threshold,
        "reason": None if rep_passed else "fresh_weight_ratio_below_threshold",
    }

    food_policy = policy["food_gate"]
    fresh_food_sources = count_food_sources(snapshot, fresh_only=True)
    usable_food_sources = count_food_sources(snapshot, fresh_only=False)
    food_passed = (
        fresh_food_sources >= int(food_policy["min_fresh_sources"])
        and usable_food_sources >= int(food_policy["min_usable_sources"])
    )
    diagnostics["food_resilience"] = {
        "passed": food_passed,
        "value": {
            "fresh_sources": fresh_food_sources,
            "usable_sources": usable_food_sources,
            "preferred_source_status": source_by_name.get(food_policy["preferred_source"], {}).get("status"),
        },
        "threshold": {
            "min_fresh_sources": food_policy["min_fresh_sources"],
            "min_usable_sources": food_policy["min_usable_sources"],
        },
        "reason": None if food_passed else "food_source_resilience_failed",
    }
    qa_summary = snapshot.get("meta", {}).get("qa_summary", {})
    source_pass_rate = qa_summary.get("source_contract_pass_rate")
    min_pass_rate = float(policy.get("min_source_pass_rate_30d", 0.95))
    source_passed = isinstance(source_pass_rate, (int, float)) and float(source_pass_rate) >= min_pass_rate
    diagnostics["source_reliability"] = {
        "passed": source_passed,
        "value": source_pass_rate,
        "threshold": min_pass_rate,
        "reason": None if source_passed else "source_contract_pass_rate_below_threshold",
    }

    imputed_weight_ratio = qa_summary.get("imputed_weight_ratio")
    max_imputed = float(policy.get("max_imputed_weight_ratio", 0.15))
    imputation_passed = isinstance(imputed_weight_ratio, (int, float)) and float(imputed_weight_ratio) <= max_imputed
    diagnostics["imputation_weight"] = {
        "passed": imputation_passed,
        "value": imputed_weight_ratio,
        "threshold": max_imputed,
        "reason": None if imputation_passed else "imputed_weight_ratio_above_threshold",
    }

    disagreement_score = qa_summary.get("cross_source_disagreement_score")
    disagreement_by_category = qa_summary.get("cross_source_disagreement_by_category", {})
    max_disagreement_policy = policy.get("max_cross_source_disagreement_score", {})
    disagreement_passed = True
    if isinstance(disagreement_by_category, dict):
        for category, score in disagreement_by_category.items():
            threshold = float(max_disagreement_policy.get(category, 0.35))
            if isinstance(score, (int, float)) and float(score) > threshold:
                disagreement_passed = False
                break
    diagnostics["cross_source_disagreement"] = {
        "passed": disagreement_passed,
        "value": disagreement_score,
        "threshold": max_disagreement_policy,
        "reason": None if disagreement_passed else "cross_source_disagreement_above_threshold",
    }
    return diagnostics


def evaluate_gate(snapshot: dict) -> list[str]:
    blocked: list[str] = []
    diagnostics = build_gate_diagnostics(snapshot)
    if not diagnostics["required_sources"]["passed"]:
        blocked.append("Gate B failed: one or more required sources missing.")
    if not diagnostics["energy_source"]["passed"]:
        blocked.append("Gate B failed: no usable energy source.")
    if not diagnostics["category_points"]["passed"]:
        failed = [
            row for row in diagnostics["category_points"]["value"] if not row["passed"] and row["category"] in CORE_GATE_CATEGORIES
        ]
        for row in failed:
            blocked.append(f"Gate D failed: category {row['category']} has fewer than {row['threshold']} points.")
    if not diagnostics["official_metadata"]["passed"]:
        blocked.append("Gate E failed: official CPI metadata missing latest release month.")
    if not diagnostics["representativeness"]["passed"]:
        threshold = diagnostics["representativeness"]["threshold"]
        blocked.append(f"Gate F failed: representativeness ratio below {int(threshold * 100)}% fresh basket coverage.")
    if not diagnostics["food_resilience"]["passed"]:
        blocked.append("Gate A failed: food source resilience below minimum fresh/diversity threshold.")
    if not diagnostics["source_reliability"]["passed"]:
        blocked.append("Gate G failed: trailing source contract pass rate below minimum.")
    if not diagnostics["imputation_weight"]["passed"]:
        blocked.append("Gate H failed: imputed basket weight above allowed threshold.")
    if not diagnostics["cross_source_disagreement"]["passed"]:
        blocked.append("Gate I failed: cross-source disagreement above allowed threshold.")

    return blocked


def validate_snapshot(snapshot: dict) -> list[str]:
    try:
        NowcastSnapshot.model_validate(snapshot)
        return []
    except Exception as err:
        return [f"Gate C failed: snapshot schema validation error: {err}"]


def ensure_release_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(RELEASE_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS release_runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                blocked_conditions TEXT NOT NULL,
                snapshot_path TEXT NOT NULL
            )
            """
        )
        conn.commit()


def record_release_run(run_id: str, created_at: str, status: str, blocked_conditions: list[str], snapshot_path: str) -> None:
    ensure_release_db()
    with sqlite3.connect(RELEASE_DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO release_runs (run_id, created_at, status, blocked_conditions, snapshot_path) VALUES (?, ?, ?, ?, ?)",
            (run_id, created_at, status, json.dumps(blocked_conditions), snapshot_path),
        )
        conn.commit()


def ensure_qa_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(QA_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS source_run_checks (
                run_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL,
                category TEXT,
                passed INTEGER NOT NULL,
                attempts INTEGER NOT NULL,
                freshness_hours REAL,
                record_count INTEGER,
                details_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_source_reliability (
                as_of_date TEXT NOT NULL,
                source TEXT NOT NULL,
                pass_rate_30d REAL NOT NULL,
                freshness_pass_rate_30d REAL NOT NULL,
                runs_30d INTEGER NOT NULL,
                PRIMARY KEY (as_of_date, source)
            )
            """
        )
        conn.commit()


def record_qa_checks(run_id: str, created_at: str, checks: list[dict], source_health: list[dict], as_of_date: str) -> None:
    ensure_qa_db()
    health_by_source = {row.get("source"): row for row in source_health if isinstance(row, dict)}
    with sqlite3.connect(QA_DB_PATH) as conn:
        for check in checks:
            source = check.get("source")
            if not isinstance(source, str):
                continue
            health = health_by_source.get(source, {})
            category = health.get("category")
            freshness_hours = health.get("run_age_hours")
            record_count = None
            for item in check.get("checks", []):
                if item.get("name") == "record_count":
                    detail = item.get("detail", "")
                    try:
                        record_count = int(str(detail).split(" ", 1)[0])
                    except Exception:
                        record_count = None
            conn.execute(
                "INSERT INTO source_run_checks (run_id, created_at, source, category, passed, attempts, freshness_hours, record_count, details_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    created_at,
                    source,
                    category,
                    1 if check.get("passed") else 0,
                    int(check.get("attempts") or 1),
                    float(freshness_hours) if isinstance(freshness_hours, (int, float)) else None,
                    record_count,
                    json.dumps(check),
                ),
            )

        window_start = (date.fromisoformat(as_of_date) - timedelta(days=30)).isoformat()
        rows = conn.execute(
            "SELECT source, COUNT(*) AS runs, AVG(passed * 1.0) AS pass_rate, "
            "AVG(CASE WHEN freshness_hours IS NOT NULL AND freshness_hours <= 48 THEN 1.0 ELSE 0.0 END) AS freshness_rate "
            "FROM source_run_checks WHERE DATE(created_at) >= DATE(?) GROUP BY source",
            (window_start,),
        ).fetchall()
        for source, runs, pass_rate, freshness_rate in rows:
            conn.execute(
                "INSERT OR REPLACE INTO daily_source_reliability (as_of_date, source, pass_rate_30d, freshness_pass_rate_30d, runs_30d) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    as_of_date,
                    source,
                    round(float(pass_rate or 0.0), 4),
                    round(float(freshness_rate or 0.0), 4),
                    int(runs or 0),
                ),
            )
        conn.commit()


def source_pass_rate_30d(as_of_date: str) -> dict[str, float]:
    if not QA_DB_PATH.exists():
        return {}
    with sqlite3.connect(QA_DB_PATH) as conn:
        rows = conn.execute(
            "SELECT source, pass_rate_30d FROM daily_source_reliability WHERE as_of_date = ?",
            (as_of_date,),
        ).fetchall()
    return {str(source): float(rate) for source, rate in rows}


def update_historical(snapshot: dict, historical: dict) -> dict:
    day = snapshot["as_of_date"]
    official = snapshot.get("official_cpi", {})
    nowcast_mom = snapshot.get("headline", {}).get("nowcast_mom_pct")
    nowcast_yoy = snapshot.get("headline", {}).get("nowcast_yoy_pct")
    official_mom = official.get("mom_pct")
    divergence = None
    if nowcast_mom is not None and official_mom is not None:
        divergence = round_or_none(float(nowcast_mom) - float(official_mom), 4)

    historical[day] = {
        "headline": {
            "nowcast_mom_pct": nowcast_mom,
            "nowcast_yoy_pct": nowcast_yoy,
            "confidence": snapshot["headline"]["confidence"],
            "coverage_ratio": snapshot["headline"]["coverage_ratio"],
            "signal_quality_score": snapshot["headline"]["signal_quality_score"],
            "lead_signal": snapshot["headline"]["lead_signal"],
            "next_release_at_utc": snapshot["headline"].get("next_release_at_utc"),
            "consensus_yoy": snapshot["headline"].get("consensus_yoy"),
            "consensus_spread_yoy": snapshot["headline"].get("consensus_spread_yoy"),
            "deviation_yoy_pct": snapshot["headline"].get("deviation_yoy_pct"),
            "divergence_mom_pct": divergence,
        },
        "official_cpi": {
            "latest_release_month": official.get("latest_release_month"),
            "mom_pct": official_mom,
            "yoy_pct": official.get("yoy_pct"),
            "yoy_display_pct": official.get("yoy_display_pct"),
        },
        "categories": {
            k: {
                "proxy_level": v["proxy_level"],
                "daily_change_pct": v["daily_change_pct"],
                "status": v["status"],
            }
            for k, v in snapshot["categories"].items()
        },
        "category_contributions": snapshot.get("meta", {}).get("category_contributions", {}),
        "source_health": [
            {
                "source": s["source"],
                "status": s["status"],
                "category": s["category"],
                "tier": s["tier"],
                "age_days": s.get("age_days"),
                "last_success_timestamp": s.get("last_success_timestamp"),
                "last_observation_period": s.get("last_observation_period"),
            }
            for s in snapshot["source_health"]
        ],
        "release": snapshot["release"],
    }
    return historical


def build_snapshot() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    historical = load_historical()
    source_contracts = load_source_contracts()

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    now = utc_now().replace(microsecond=0)
    qa_window_close_at = (now + timedelta(hours=24)).isoformat()
    release_events = fetch_release_events()
    consensus_latest = fetch_consensus_estimate()
    next_release = compute_next_release(release_events, now)

    quotes, source_health, collection_diagnostics, qa_checks = collect_all_quotes(
        historical=historical,
        now=now,
        source_contracts=source_contracts,
    )

    # Filter out "scrappy" raw price quotes from the main index calculation
    # to prevent mixing Index (100-basis) with Prices ($2000).
    # We keep them only for the 'indicators' metadata.
    scrappy_ids = {"average_asking_rent_canada", "gasoline_regular_canada_avg"}
    index_quotes = [q for q in quotes if q.item_id not in scrappy_ids]

    indicators = extract_hero_indicators(quotes)

    computed_health = recompute_source_health(source_health, now)

    # Use index_quotes for the main calculation
    deduped = dedupe_quotes(index_quotes)
    valid_quotes, rejected_points = apply_range_checks(deduped)
    filtered, anomalies = apply_outlier_filter(valid_quotes, historical)
    reconciled_quotes, reconciliation = reconcile_cross_source_quotes(filtered, computed_health, historical)
    filtered = reconciled_quotes

    categories, category_signal_inputs = summarize_categories(filtered, computed_health)
    imputation = impute_missing_categories(categories, historical)
    compute_daily_changes(categories, historical)
    housing_overlay = apply_housing_signal_overlay(categories, indicators)

    coverage_ratio = compute_coverage(categories)
    representativeness_ratio = compute_representativeness(categories)
    nowcast_mom = compute_nowcast_mom(categories, historical)
    diversity_by_category = category_source_diversity(filtered)
    category_contributions = compute_category_contributions(categories)
    for category, rows in category_signal_inputs.items():
        if not isinstance(rows, list):
            continue
        total_source_weight = sum(float(row.get("effective_weight") or 0.0) for row in rows)
        category_contribution = category_contributions.get(category)
        for row in rows:
            share = (float(row.get("effective_weight") or 0.0) / total_source_weight) if total_source_weight > 0 else 0.0
            row["source_weight_share"] = round_or_none(share, 3)
            row["category_contribution_pct"] = category_contribution
            row["source_contribution_pct"] = round_or_none(float(category_contribution) * share, 4) if category_contribution is not None else None
    official_cpi = fetch_official_cpi_summary()
    official_series = fetch_official_cpi_series()
    official_yoy = official_cpi.get("yoy_pct")
    official_mom = official_cpi.get("mom_pct")
    official_yoy_display = round_or_none(float(official_yoy), 1) if official_yoy is not None else None
    fallback_used = False
    if nowcast_mom is None and official_mom is not None:
        # Bootstrap fallback: keep headline populated when category baseline is not yet established.
        nowcast_mom = round_or_none(float(official_mom), 3)
        fallback_used = True
    lead_signal = derive_lead_signal(nowcast_mom)
    consensus_yoy, consensus_guardrails = apply_consensus_guardrails(consensus_latest if isinstance(consensus_latest, dict) else None)
    raw_nowcast_yoy, yoy_projection = compute_nowcast_yoy_prorated(now.date(), nowcast_mom, official_series)
    nowcast_yoy, calibration_diagnostics = calibrate_nowcast_yoy(
        raw_nowcast_yoy=raw_nowcast_yoy,
        official_series=official_series,
        diversity_by_category=diversity_by_category,
        categories=categories,
        anomalies=anomalies,
    )
    forecast = compute_forecast(
        nowcast_yoy=nowcast_yoy,
        official_series=official_series,
        consensus_yoy=consensus_yoy,
        historical=historical,
        next_release=next_release,
        as_of=now,
    )
    deviation_yoy = None
    if nowcast_yoy is not None and consensus_yoy is not None:
        deviation_yoy = round_or_none(float(nowcast_yoy) - float(consensus_yoy), 3)
    # Deprecated alias retained for compatibility during transition.
    consensus_spread_yoy = deviation_yoy
    this_run_contract_pass_rate = 0.0
    if qa_checks:
        this_run_contract_pass_rate = sum(1 for c in qa_checks if c.get("passed")) / len(qa_checks)

    trailing_by_source = source_pass_rate_30d(now.date().isoformat())
    source_contract_pass_rate = (
        round_or_none(sum(trailing_by_source.values()) / len(trailing_by_source), 4)
        if trailing_by_source
        else round_or_none(this_run_contract_pass_rate, 4)
    )

    qa_summary = {
        "source_contract_pass_rate": source_contract_pass_rate,
        "fresh_weight_ratio": representativeness_ratio,
        "cross_source_disagreement_score": reconciliation["cross_source_disagreement_score"],
        "cross_source_disagreement_by_category": reconciliation["cross_source_disagreement_by_category"],
        "quarantine_sources": reconciliation["quarantine_sources"],
        "quarantine_reasons": reconciliation["quarantine_reasons"],
        "imputation_used": imputation["imputation_used"],
        "imputed_categories": imputation["imputed_categories"],
        "imputed_weight_ratio": imputation["imputed_weight_ratio"],
        "source_checks": qa_checks,
        "retry_schedule_minutes": QA_RETRY_OFFSETS_MINUTES,
    }

    snapshot = {
        "as_of_date": now.date().isoformat(),
        "timestamp": now.isoformat(),
        "headline": {
            "nowcast_mom_pct": nowcast_mom,
            "nowcast_yoy_pct": nowcast_yoy,
            "confidence": "low",
            "coverage_ratio": coverage_ratio,
            "signal_quality_score": 0,
            "lead_signal": lead_signal,
            "next_release_at_utc": next_release.get("release_at_utc") if next_release else None,
            "consensus_yoy": consensus_yoy,
            "consensus_spread_yoy": consensus_spread_yoy,
            "deviation_yoy_pct": deviation_yoy,
            "method_label": METHOD_LABEL,
        },
        "categories": categories,
        "official_cpi": official_cpi,
        "bank_of_canada": fetch_boc_cpi(),
        "source_health": computed_health,
        "notes": [],
        "meta": {
            "method_version": METHOD_VERSION,
            "weights": weights_payload(),
            "total_raw_points": len(quotes),
            "total_points_after_dedupe": len(deduped),
            "total_points_after_quality_filters": len(filtered),
            "anomaly_points": anomalies,
            "rejected_points": rejected_points,
            "representativeness_ratio": representativeness_ratio,
            "source_diversity_by_category": diversity_by_category,
            "category_signal_inputs": category_signal_inputs,
            "category_contributions": category_contributions,
            "top_driver": compute_top_driver(category_contributions),
            "qa_summary": qa_summary,
            "province_overlays": [],
            "release_intelligence": next_release or {},
            "release_events": release_events,
            "fallbacks": {"nowcast_from_official_mom": fallback_used},
            "collection_diagnostics": collection_diagnostics,
            "housing_signal_overlay": housing_overlay,
            "projection": {
                "nowcast_yoy_prorated": yoy_projection,
            },
            "consensus": {
                "headline_yoy": consensus_yoy,
                "headline_mom": consensus_latest.get("headline_mom") if isinstance(consensus_latest, dict) else None,
                "source_count": consensus_latest.get("source_count") if isinstance(consensus_latest, dict) else 0,
                "confidence": consensus_latest.get("confidence") if isinstance(consensus_latest, dict) else "low",
                "as_of": consensus_latest.get("as_of") if isinstance(consensus_latest, dict) else None,
                "source_urls": [s.get("url") for s in consensus_latest.get("sources", []) if isinstance(s, dict)]
                if isinstance(consensus_latest, dict)
                else [],
                "sources": consensus_latest.get("sources", []) if isinstance(consensus_latest, dict) else [],
                "errors": consensus_latest.get("errors", []) if isinstance(consensus_latest, dict) else [],
                "guardrails": consensus_guardrails,
            },
            "indicators": indicators,
            "forecast": forecast,
            "calibration_diagnostics": calibration_diagnostics,
        },
        "performance_ref": {
            "summary_path": str(PERFORMANCE_SUMMARY_PATH),
            "model_card_path": str(MODEL_CARD_PATH),
        },
        "release": {
            "run_id": run_id,
            "status": "started",
            "qa_status": "pending",
            "qa_window_close_at": qa_window_close_at,
            "lifecycle_states": ["started", "collect"],
            "blocked_conditions": [],
            "created_at": now.isoformat(),
            "published_at": None,
        },
    }

    snapshot["release"]["lifecycle_states"].append("validate")
    snapshot["release"]["lifecycle_states"].append("reconcile")
    snapshot["release"]["status"] = "completed"
    snapshot["release"]["lifecycle_states"].append("completed")
    snapshot["release"]["lifecycle_states"].append("publish")
    gate_diagnostics = build_gate_diagnostics(snapshot)
    snapshot["meta"]["gate_diagnostics"] = gate_diagnostics
    blocked_conditions = evaluate_gate(snapshot)
    blocked_conditions.extend(validate_snapshot(snapshot))

    status = "published" if not blocked_conditions else "failed_gate"
    snapshot["release"]["status"] = status
    snapshot["release"]["lifecycle_states"].append(status)
    snapshot["release"]["blocked_conditions"] = blocked_conditions
    snapshot["release"]["qa_status"] = "passed" if status == "published" else "failed"
    if status == "published":
        snapshot["release"]["published_at"] = now.isoformat()

    snapshot["headline"]["signal_quality_score"] = compute_signal_quality_score(
        coverage_ratio=coverage_ratio,
        anomalies=anomalies,
        blocked_conditions=blocked_conditions,
        diversity_by_category=diversity_by_category,
        categories=categories,
    )
    snapshot["headline"]["confidence"] = compute_confidence(
        coverage_ratio=coverage_ratio,
        anomalies=anomalies,
        blocked_conditions=blocked_conditions,
        diversity_by_category=diversity_by_category,
        categories=categories,
    )
    snapshot["notes"] = build_notes(
        categories=categories,
        anomalies=anomalies,
        rejected_points=rejected_points,
        blocked_conditions=blocked_conditions,
        diversity_by_category=diversity_by_category,
        representativeness_ratio=representativeness_ratio,
        collection_diagnostics=collection_diagnostics,
    )
    if fallback_used:
        snapshot["notes"].append("Nowcast MoM uses official MoM fallback until sufficient category history is available.")
    snapshot["notes"].append(
        "Deprecated fields retained for compatibility: headline.nowcast_mom_pct and headline.consensus_spread_yoy."
    )
    if nowcast_yoy is None:
        reason = yoy_projection.get("reason") if isinstance(yoy_projection, dict) else "unknown"
        snapshot["notes"].append(f"Nowcast YoY unavailable: {reason}.")
    elif raw_nowcast_yoy is not None and nowcast_yoy != raw_nowcast_yoy:
        snapshot["notes"].append(
            f"Calibrated YoY nowcast from raw {raw_nowcast_yoy}% to {nowcast_yoy}% using transparent shrinkage diagnostics."
        )
    if official_yoy_display is not None:
        snapshot["official_cpi"]["yoy_display_pct"] = official_yoy_display
        snapshot["notes"].append(
            f"Official CPI YoY display uses one-decimal release-style rounding ({official_yoy_display}%)."
        )
    if consensus_yoy is None:
        reason = consensus_guardrails.get("reason") if isinstance(consensus_guardrails, dict) else "unknown"
        snapshot["notes"].append(f"Consensus YoY withheld due to quality guardrails: {reason}.")
    if snapshot.get("meta", {}).get("forecast", {}).get("status") != "published":
        snapshot["notes"].append("Forecast is withheld until sufficient live history accumulates.")
    perf_snapshot = compute_performance_summary(historical)
    snapshot["meta"]["calibration"] = build_calibration(
        historical=historical,
        performance_summary=perf_snapshot,
        forecast=forecast,
        calibration_diagnostics=calibration_diagnostics,
    )
    return snapshot


def write_outputs(snapshot: dict) -> None:
    historical = load_historical()
    run_id = snapshot["release"]["run_id"]
    run_path = RUNS_DIR / f"{run_id}.json"

    LATEST_PATH.write_text(json.dumps(snapshot, indent=2))
    run_path.write_text(json.dumps(snapshot, indent=2))
    if not SOURCE_CONTRACTS_PATH.exists():
        SOURCE_CONTRACTS_PATH.write_text(json.dumps(DEFAULT_SOURCE_CONTRACTS, indent=2))

    status = snapshot["release"]["status"]
    if status == "published":
        PUBLISHED_LATEST_PATH.write_text(json.dumps(snapshot, indent=2))
        historical = update_historical(snapshot, historical)
        HISTORICAL_PATH.write_text(json.dumps(historical, indent=2))

    # Persist release intelligence and free-source consensus artifacts each run.
    release_payload = snapshot.get("meta", {}).get("release_events", {})
    if isinstance(release_payload, dict):
        release_payload = {
            **release_payload,
            "next_release": snapshot.get("meta", {}).get("release_intelligence", {}),
            "method_version": METHOD_VERSION,
        }
    RELEASE_EVENTS_PATH.write_text(json.dumps(release_payload, indent=2))
    CONSENSUS_LATEST_PATH.write_text(json.dumps(snapshot.get("meta", {}).get("consensus", {}), indent=2))
    record_qa_checks(
        run_id=run_id,
        created_at=snapshot["release"]["created_at"],
        checks=snapshot.get("meta", {}).get("qa_summary", {}).get("source_checks", []),
        source_health=snapshot.get("source_health", []),
        as_of_date=snapshot["as_of_date"],
    )

    performance_summary = write_performance_summary(PERFORMANCE_SUMMARY_PATH, historical)
    model_card = {
        "as_of_date": snapshot["as_of_date"],
        "method_version": METHOD_VERSION,
        "north_star": "lead_time_vs_statcan",
        "performance": performance_summary,
        "notes": [
            "Experimental nowcast model card.",
            "Metrics are computed from published historical snapshots.",
        ],
    }
    MODEL_CARD_PATH.write_text(json.dumps(model_card, indent=2))

    record_release_run(
        run_id=run_id,
        created_at=snapshot["release"]["created_at"],
        status=status,
        blocked_conditions=snapshot["release"]["blocked_conditions"],
        snapshot_path=str(run_path),
    )


def main() -> int:
    if sys.version_info >= (3, 13):
        print("Runtime guard failed: Python 3.11 is required for stable source ingestion (apify-client compatibility).")
        print("Use: python3.11 process.py")
        return 2
    snap = build_snapshot()
    write_outputs(snap)
    status = snap["release"]["status"]
    blocked = snap["release"].get("blocked_conditions", [])
    print(f"Run status: {status}")
    print(
        "Summary: "
        f"confidence={snap['headline']['confidence']} coverage={snap['headline']['coverage_ratio']} "
        f"signal_quality_score={snap['headline']['signal_quality_score']} "
        f"sources_ok={sum(1 for s in snap['source_health'] if s['status'] in {'fresh', 'stale'})}/"
        f"{len(snap['source_health'])}"
    )
    if blocked:
        print("Blocked conditions:")
        for reason in blocked:
            print(f"- {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
