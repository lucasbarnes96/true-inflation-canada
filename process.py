from __future__ import annotations

import calendar
import json
import sqlite3
import statistics
import uuid
from collections import defaultdict
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

from models import NowcastSnapshot
from performance import compute_performance_summary, write_performance_summary
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
    scrape_food,
    scrape_food_statcan,
    scrape_grocery_apify,
    scrape_health_personal,
    scrape_health_public,
    scrape_housing,
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

CATEGORY_REGISTRY: dict[str, dict] = {
    "food": {
        "weight": 0.165,
        "value_bounds": (0.1, 500.0),
        "outlier_threshold_pct": 60.0,
        "min_points": 5,
    },
    "housing": {
        "weight": 0.300,
        "value_bounds": (1.0, 400.0),
        "outlier_threshold_pct": 30.0,
        "min_points": 2,
    },
    "transport": {
        "weight": 0.150,
        "value_bounds": (50.0, 300.0),
        "outlier_threshold_pct": 40.0,
        "min_points": 1,
    },
    "energy": {
        "weight": 0.080,
        "value_bounds": (0.1, 100.0),
        "outlier_threshold_pct": 50.0,
        "min_points": 1,
    },
    "communication": {
        "weight": 0.045,
        "value_bounds": (1.0, 400.0),
        "outlier_threshold_pct": 30.0,
        "min_points": 1,
    },
    "health_personal": {
        "weight": 0.050,
        "value_bounds": (1.0, 400.0),
        "outlier_threshold_pct": 25.0,
        "min_points": 1,
    },
    "recreation_education": {
        "weight": 0.095,
        "value_bounds": (1.0, 400.0),
        "outlier_threshold_pct": 30.0,
        "min_points": 1,
    },
}

CATEGORY_WEIGHTS = {name: cfg["weight"] for name, cfg in CATEGORY_REGISTRY.items()}
VALUE_BOUNDS = {name: cfg["value_bounds"] for name, cfg in CATEGORY_REGISTRY.items()}
OUTLIER_THRESHOLD_PCT = {name: cfg["outlier_threshold_pct"] for name, cfg in CATEGORY_REGISTRY.items()}
CATEGORY_MIN_POINTS = {name: cfg["min_points"] for name, cfg in CATEGORY_REGISTRY.items()}

SCRAPER_REGISTRY = [
    ("food_openfoodfacts", scrape_food),
    ("food_statcan", scrape_food_statcan),
    ("food_apify", scrape_grocery_apify),
    ("transport_statcan", scrape_transport),
    ("housing_statcan", scrape_housing),
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
    "statcan_cpi_csv": 45,
    "ised_mobile_plan_tracker": 60,
    "crtc_cmr_report": 400,
    "healthcanada_dpd": 90,
    "pmprb_reports": 400,
    "parkscanada_fees": 180,
    "statcan_education_portal": 180,
}

APIFY_MAX_AGE_DAYS = 14
METHOD_LABEL = "YoY nowcast from public category proxies with month-to-date prorating"
METHOD_VERSION = "v1.2.0"
CORE_GATE_CATEGORIES = ("food", "housing", "transport")


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


def summarize_categories(quotes: list[Quote], source_health: list[dict]) -> dict:
    by_category: dict[str, list[Quote]] = defaultdict(list)
    for quote in quotes:
        by_category[quote.category].append(quote)

    source_by_name = {s["source"]: s for s in source_health}

    summary: dict[str, dict] = {}
    for category, weight in CATEGORY_WEIGHTS.items():
        cat_quotes = by_category.get(category, [])
        level = round_or_none(statistics.mean([q.value for q in cat_quotes]), 4) if cat_quotes else None

        status = "missing"
        if cat_quotes:
            category_statuses = [h["status"] for h in source_health if h["category"] == category]
            status = "fresh" if "fresh" in category_statuses else "stale"

        if category == "food":
            apify = source_by_name.get("apify_loblaws")
            if not apify or apify["status"] != "fresh":
                status = "missing"

        summary[category] = {
            "proxy_level": level,
            "daily_change_pct": None,
            "weight": weight,
            "points": len(cat_quotes),
            "status": status,
        }

    return summary


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
    if not historical:
        return None

    latest_day = sorted(historical.keys())[-1]
    prev_categories = historical.get(latest_day, {}).get("categories", {})

    weighted_change = 0.0
    effective_weight = 0.0
    for category, payload in categories.items():
        current = payload.get("proxy_level")
        prev = prev_categories.get(category, {}).get("proxy_level")
        weight = payload["weight"]
        if current is None or prev in (None, 0):
            continue
        category_change = (float(current) / float(prev) - 1) * 100
        weighted_change += category_change * weight
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
) -> list[str]:
    notes: list[str] = [
        "This is an experimental nowcast estimate and not an official CPI release.",
        "Methodology v1.2.0: weighted category proxies with month-to-date YoY projection.",
        "Confidence rubric: gate status + weighted coverage + anomaly rate + source diversity.",
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

    return notes


def collect_all_quotes() -> tuple[list[Quote], list[SourceHealth]]:
    quotes: list[Quote] = []
    health: list[SourceHealth] = []

    for _, scraper in SCRAPER_REGISTRY:
        scraper_quotes, scraper_health = scraper()
        quotes.extend(scraper_quotes)
        health.extend(scraper_health)

    return quotes, health


def evaluate_gate(snapshot: dict) -> list[str]:
    blocked: list[str] = []
    source_by_name = {s["source"]: s for s in snapshot["source_health"]}

    apify = source_by_name.get("apify_loblaws")
    apify_age = apify.get("age_days") if apify else None
    if not apify or apify.get("status") == "missing" or apify_age is None or apify_age > APIFY_MAX_AGE_DAYS:
        blocked.append("Gate A failed: APIFY missing or older than 14 days.")

    for required in ("statcan_cpi_csv", "statcan_gas_csv"):
        if source_by_name.get(required, {}).get("status") == "missing":
            blocked.append(f"Gate B failed: required source {required} is missing.")

    energy_ok = False
    for source in ("oeb_scrape", "statcan_energy_cpi_csv"):
        state = source_by_name.get(source, {}).get("status")
        if state in {"fresh", "stale"}:
            energy_ok = True
            break
    if not energy_ok:
        blocked.append("Gate B failed: no usable energy source.")

    for category in CORE_GATE_CATEGORIES:
        payload = snapshot["categories"].get(category, {"points": 0})
        min_points = CATEGORY_MIN_POINTS[category]
        if payload["points"] < min_points:
            blocked.append(f"Gate D failed: category {category} has fewer than {min_points} points.")

    official_month = snapshot.get("official_cpi", {}).get("latest_release_month")
    if not official_month:
        blocked.append("Gate E failed: official CPI metadata missing latest release month.")

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

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    now = utc_now().replace(microsecond=0)
    release_events = fetch_release_events()
    consensus_latest = fetch_consensus_estimate()
    next_release = compute_next_release(release_events, now)

    raw_quotes, raw_source_health = collect_all_quotes()
    source_health = recompute_source_health(raw_source_health, now=now)

    deduped = dedupe_quotes(raw_quotes)
    range_valid, rejected_points = apply_range_checks(deduped)
    filtered, anomalies = apply_outlier_filter(range_valid, historical)

    categories = summarize_categories(filtered, source_health)
    compute_daily_changes(categories, historical)

    coverage_ratio = compute_coverage(categories)
    representativeness_ratio = compute_representativeness(categories)
    nowcast_mom = compute_nowcast_mom(categories, historical)
    diversity_by_category = category_source_diversity(filtered)
    category_contributions = compute_category_contributions(categories)
    official_cpi = fetch_official_cpi_summary()
    official_series = fetch_official_cpi_series()
    official_yoy = official_cpi.get("yoy_pct")
    official_mom = official_cpi.get("mom_pct")
    fallback_used = False
    if nowcast_mom is None and official_mom is not None:
        # Bootstrap fallback: keep headline populated when category baseline is not yet established.
        nowcast_mom = round_or_none(float(official_mom), 3)
        fallback_used = True
    lead_signal = derive_lead_signal(nowcast_mom)
    consensus_yoy = consensus_latest.get("headline_yoy") if isinstance(consensus_latest, dict) else None
    nowcast_yoy, yoy_projection = compute_nowcast_yoy_prorated(now.date(), nowcast_mom, official_series)
    deviation_yoy = None
    if nowcast_yoy is not None and consensus_yoy is not None:
        deviation_yoy = round_or_none(float(nowcast_yoy) - float(consensus_yoy), 3)
    # Deprecated alias retained for compatibility during transition.
    consensus_spread_yoy = deviation_yoy

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
        "source_health": source_health,
        "notes": [],
        "meta": {
            "method_version": METHOD_VERSION,
            "total_raw_points": len(raw_quotes),
            "total_points_after_dedupe": len(deduped),
            "total_points_after_quality_filters": len(filtered),
            "anomaly_points": anomalies,
            "rejected_points": rejected_points,
            "representativeness_ratio": representativeness_ratio,
            "source_diversity_by_category": diversity_by_category,
            "category_contributions": category_contributions,
            "top_driver": compute_top_driver(category_contributions),
            "province_overlays": ["ON", "QC", "BC"],
            "release_intelligence": next_release or {},
            "release_events": release_events,
            "fallbacks": {"nowcast_from_official_mom": fallback_used},
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
            },
        },
        "performance_ref": {
            "summary_path": str(PERFORMANCE_SUMMARY_PATH),
            "model_card_path": str(MODEL_CARD_PATH),
        },
        "release": {
            "run_id": run_id,
            "status": "started",
            "lifecycle_states": ["started"],
            "blocked_conditions": [],
            "created_at": now.isoformat(),
            "published_at": None,
        },
    }

    snapshot["release"]["status"] = "completed"
    snapshot["release"]["lifecycle_states"].append("completed")
    blocked_conditions = evaluate_gate(snapshot)
    blocked_conditions.extend(validate_snapshot(snapshot))

    status = "published" if not blocked_conditions else "failed_gate"
    snapshot["release"]["status"] = status
    snapshot["release"]["lifecycle_states"].append(status)
    snapshot["release"]["blocked_conditions"] = blocked_conditions
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
    )
    if fallback_used:
        snapshot["notes"].append("Nowcast MoM uses official MoM fallback until sufficient category history is available.")
    snapshot["notes"].append(
        "Deprecated fields retained for compatibility: headline.nowcast_mom_pct and headline.consensus_spread_yoy."
    )
    if nowcast_yoy is None:
        reason = yoy_projection.get("reason") if isinstance(yoy_projection, dict) else "unknown"
        snapshot["notes"].append(f"Nowcast YoY unavailable: {reason}.")
    return snapshot


def write_outputs(snapshot: dict) -> None:
    historical = load_historical()
    run_id = snapshot["release"]["run_id"]
    run_path = RUNS_DIR / f"{run_id}.json"

    LATEST_PATH.write_text(json.dumps(snapshot, indent=2))
    run_path.write_text(json.dumps(snapshot, indent=2))

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
