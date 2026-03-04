#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import process
from pipeline.run_state import normalize_snapshot_run_state

DATA_DIR = Path("data")
LATEST_PATH = DATA_DIR / "latest.json"
PUBLISHED_LATEST_PATH = DATA_DIR / "published_latest.json"
RUNS_DIR = DATA_DIR / "runs"
QA_DB_PATH = DATA_DIR / "qa_runs.db"


def reliability_rows_for_date(as_of_date: str | None) -> list[tuple[float, float]]:
    if not isinstance(as_of_date, str) or not QA_DB_PATH.exists():
        return []
    with sqlite3.connect(QA_DB_PATH) as conn:
        rows = conn.execute(
            "SELECT pass_rate_30d, freshness_pass_rate_30d "
            "FROM daily_source_reliability WHERE as_of_date = ?",
            (as_of_date,),
        ).fetchall()
    return [(float(pass_rate), float(freshness)) for pass_rate, freshness in rows]


def enrich_payload(payload: dict) -> bool:
    before = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    normalize_snapshot_run_state(
        payload,
        reliability_rows=reliability_rows_for_date(payload.get("as_of_date")),
        build_qa_failure_fingerprint_fn=process.build_qa_failure_fingerprint,
    )
    after = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return before != after


def backfill_path(path: Path) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        return False
    changed = enrich_payload(payload)
    if changed:
        path.write_text(json.dumps(payload, indent=2))
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill ops and QA summary fields for legacy snapshots.")
    parser.add_argument("--runs", action="store_true", help="Also backfill data/runs/run_*.json snapshots.")
    args = parser.parse_args()

    paths = [LATEST_PATH, PUBLISHED_LATEST_PATH]
    changed = 0
    visited = 0
    for path in paths:
        visited += 1
        if backfill_path(path):
            changed += 1

    if args.runs and RUNS_DIR.exists():
        for path in sorted(RUNS_DIR.glob("run_*.json")):
            visited += 1
            if backfill_path(path):
                changed += 1

    print(f"Backfill complete. Updated {changed} of {visited} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
