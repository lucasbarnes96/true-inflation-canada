from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

from models import NowcastSnapshot
from source_catalog import SOURCE_CATALOG

DATA_DIR = Path("data")
LATEST_PATH = DATA_DIR / "latest.json"
PUBLISHED_LATEST_PATH = DATA_DIR / "published_latest.json"
HISTORICAL_PATH = DATA_DIR / "historical.json"
RELEASE_DB_PATH = DATA_DIR / "releases.db"
PERFORMANCE_SUMMARY_PATH = DATA_DIR / "performance_summary.json"
RELEASE_EVENTS_PATH = DATA_DIR / "release_events.json"
CONSENSUS_LATEST_PATH = DATA_DIR / "consensus_latest.json"

app = FastAPI(title="True Inflation Canada API", version="1.6.0")


def _load_json(path: Path, default: dict | list) -> dict | list:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


@app.get("/v1/nowcast/latest")
def nowcast_latest() -> dict:
    payload = _load_json(PUBLISHED_LATEST_PATH, {})
    if not payload:
        payload = _load_json(LATEST_PATH, {})
    if not payload:
        raise HTTPException(status_code=404, detail="No snapshot available.")
    validated = NowcastSnapshot.model_validate(payload)
    return validated.model_dump(mode="json")


@app.get("/v1/nowcast/history")
def nowcast_history(
    start: date | None = Query(default=None),
    end: date | None = Query(default=None),
) -> dict:
    history = _load_json(HISTORICAL_PATH, {})
    if not isinstance(history, dict):
        return {"items": []}

    items: list[dict] = []
    for day, payload in sorted(history.items()):
        try:
            day_date = date.fromisoformat(day)
        except ValueError:
            continue
        if start and day_date < start:
            continue
        if end and day_date > end:
            continue
        row = {"date": day, **payload}
        headline = row.get("headline", {})
        official = row.get("official_cpi", {})
        nowcast_mom = headline.get("nowcast_mom_pct")
        official_mom = official.get("mom_pct")
        if headline.get("divergence_mom_pct") is None and nowcast_mom is not None and official_mom is not None:
            headline["divergence_mom_pct"] = round(float(nowcast_mom) - float(official_mom), 4)
            row["headline"] = headline
        row["category_contributions"] = row.get("category_contributions") or row.get("meta", {}).get("category_contributions")
        items.append(row)
    return {"items": items}


@app.get("/v1/sources/health")
def sources_health() -> dict:
    payload = _load_json(PUBLISHED_LATEST_PATH, {})
    if not payload:
        payload = _load_json(LATEST_PATH, {})
    sources = payload.get("source_health", []) if isinstance(payload, dict) else []
    if isinstance(sources, list):
        for row in sources:
            if not isinstance(row, dict):
                continue
            age_days = row.get("age_days")
            if row.get("run_age_hours") is None and isinstance(age_days, (int, float)):
                row["run_age_hours"] = round(float(age_days) * 24.0, 2)
    return {"items": sources}


@app.get("/v1/releases/latest")
def releases_latest() -> dict:
    if not RELEASE_DB_PATH.exists():
        raise HTTPException(status_code=404, detail="No release runs found.")

    with sqlite3.connect(RELEASE_DB_PATH) as conn:
        row = conn.execute(
            "SELECT run_id, created_at, status, blocked_conditions, snapshot_path "
            "FROM release_runs ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="No release runs found.")

    run_id, created_at, status, blocked_conditions, snapshot_path = row
    try:
        blocked = json.loads(blocked_conditions)
    except json.JSONDecodeError:
        blocked = []
    return {
        "run_id": run_id,
        "created_at": created_at,
        "status": status,
        "blocked_conditions": blocked,
        "snapshot_path": snapshot_path,
    }


@app.get("/v1/methodology")
def methodology() -> dict:
    return {
        "summary": "Weighted category nowcast using free/public daily and monthly sources.",
        "method_version": "v1.6.0",
        "confidence_formula": {
            "inputs": ["gate_status", "coverage_ratio", "anomalies", "source_diversity"],
            "high": "coverage_ratio >= 0.9, no gate failures, low anomalies, no diversity penalty",
            "medium": "coverage_ratio >= 0.6 or diversity/anomaly penalties",
            "low": "gate failure or low weighted coverage",
        },
        "gate_policy": {
            "apify_max_age_days": 14,
            "required_sources": ["apify_loblaws", "statcan_cpi_csv", "statcan_gas_csv"],
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
        },
        "limitations": [
            "Experimental nowcast, not an official CPI release.",
            "APIFY is run weekly on free-tier constraints.",
            "Monthly sources may remain fresh for up to 45 days.",
        ],
    }


@app.get("/v1/performance/summary")
def performance_summary() -> dict:
    payload = _load_json(PERFORMANCE_SUMMARY_PATH, {})
    if not payload:
        raise HTTPException(status_code=404, detail="No performance summary available.")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Invalid performance summary format.")
    return payload


@app.get("/v1/sources/catalog")
def sources_catalog() -> dict:
    return {"items": SOURCE_CATALOG}


@app.get("/v1/releases/upcoming")
def releases_upcoming() -> dict:
    payload = _load_json(RELEASE_EVENTS_PATH, {})
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Invalid release events format.")
    events = payload.get("events", [])
    if not isinstance(events, list):
        events = []
    next_release = payload.get("next_release", {})
    return {"next_release": next_release, "events": events}


@app.get("/v1/consensus/latest")
def consensus_latest() -> dict:
    payload = _load_json(CONSENSUS_LATEST_PATH, {})
    if not payload:
        raise HTTPException(status_code=404, detail="No consensus snapshot available.")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Invalid consensus format.")
    return payload
