from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from gate_policy import METHOD_VERSION, gate_policy_payload, weights_payload
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
QA_DB_PATH = DATA_DIR / "qa_runs.db"

app = FastAPI(title="True Inflation Canada API", version=METHOD_VERSION)

# Serve index.html at root
@app.get("/")
async def read_index():
    return FileResponse('index.html')

# Mount static files (if any other assets needed, though index.html is main one)
app.mount("/static", StaticFiles(directory="."), name="static")


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
        nowcast_yoy = headline.get("nowcast_yoy_pct")
        consensus_yoy = headline.get("consensus_yoy")
        if headline.get("deviation_yoy_pct") is None and nowcast_yoy is not None and consensus_yoy is not None:
            headline["deviation_yoy_pct"] = round(float(nowcast_yoy) - float(consensus_yoy), 4)
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
    weights = weights_payload()
    latest_payload = _load_json(PUBLISHED_LATEST_PATH, {})
    as_of_utc = latest_payload.get("timestamp") if isinstance(latest_payload, dict) else None
    return {
        "summary": "Weighted category nowcast using free/public daily and monthly sources with transparent calibration diagnostics.",
        "method_version": METHOD_VERSION,
        "as_of_utc": as_of_utc,
        "weights_reference": {
            "source_table": weights.get("source_table"),
            "source_url": weights.get("source_url"),
            "analysis_reference": weights.get("analysis_reference"),
            "analysis_url": weights.get("analysis_url"),
            "basket_reference_year": weights.get("basket_reference_year"),
            "effective_month": weights.get("effective_month"),
            "tracked_share_total": weights.get("tracked_share_total"),
        },
        "confidence_formula": {
            "inputs": ["gate_status", "coverage_ratio", "anomalies", "source_diversity"],
            "high": "coverage_ratio >= 0.9, no gate failures, low anomalies, no diversity penalty",
            "medium": "coverage_ratio >= 0.6 or diversity/anomaly penalties",
            "low": "gate failure or low weighted coverage",
        },
        "gate_policy": gate_policy_payload(),
        "limitations": [
            "Experimental nowcast, not an official CPI release.",
            "APIFY auto-retry is attempted when stale/missing before gate evaluation.",
            "Monthly sources may remain fresh for up to 45 days.",
            "Communication is represented as a proxy mapped within broader official CPI components.",
            "Housing asking-rent overlay is a momentum proxy and differs from survey-based transacted-rent methods.",
            "Deprecated compatibility fields remain available: headline.nowcast_mom_pct and headline.consensus_spread_yoy.",
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


@app.get("/v1/forecast/next_release")
def forecast_next_release() -> dict:
    payload = _load_json(PUBLISHED_LATEST_PATH, {})
    if not payload:
        payload = _load_json(LATEST_PATH, {})
    if not isinstance(payload, dict):
        raise HTTPException(status_code=404, detail="No snapshot available.")
    forecast = payload.get("meta", {}).get("forecast")
    if not isinstance(forecast, dict):
        raise HTTPException(status_code=404, detail="No forecast available.")
    return forecast


@app.get("/v1/calibration/status")
def calibration_status() -> dict:
    payload = _load_json(PUBLISHED_LATEST_PATH, {})
    if not payload:
        payload = _load_json(LATEST_PATH, {})
    if not isinstance(payload, dict):
        raise HTTPException(status_code=404, detail="No snapshot available.")
    calibration = payload.get("meta", {}).get("calibration")
    if not isinstance(calibration, dict):
        raise HTTPException(status_code=404, detail="No calibration data available.")
    return calibration


@app.get("/v1/qa/status")
def qa_status() -> dict:
    payload = _load_json(LATEST_PATH, {})
    if not isinstance(payload, dict):
        raise HTTPException(status_code=404, detail="No snapshot available.")
    release = payload.get("release", {})
    qa_summary = payload.get("meta", {}).get("qa_summary", {})
    reliability: list[dict] = []
    if QA_DB_PATH.exists():
        as_of_date = payload.get("as_of_date")
        if isinstance(as_of_date, str):
            with sqlite3.connect(QA_DB_PATH) as conn:
                rows = conn.execute(
                    "SELECT source, pass_rate_30d, freshness_pass_rate_30d, runs_30d "
                    "FROM daily_source_reliability WHERE as_of_date = ? ORDER BY source",
                    (as_of_date,),
                ).fetchall()
            reliability = [
                {
                    "source": source,
                    "pass_rate_30d": pass_rate,
                    "freshness_pass_rate_30d": freshness,
                    "runs_30d": runs,
                }
                for source, pass_rate, freshness, runs in rows
            ]

    return {
        "run_id": release.get("run_id"),
        "release_status": release.get("status"),
        "qa_status": release.get("qa_status", "pending"),
        "qa_window_close_at": release.get("qa_window_close_at"),
        "blocked_conditions": release.get("blocked_conditions", []),
        "qa_summary": qa_summary,
        "source_reliability_30d": reliability,
    }
