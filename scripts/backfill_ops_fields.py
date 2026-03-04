#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import process

DATA_DIR = Path("data")
LATEST_PATH = DATA_DIR / "latest.json"
PUBLISHED_LATEST_PATH = DATA_DIR / "published_latest.json"
RUNS_DIR = DATA_DIR / "runs"
QA_DB_PATH = DATA_DIR / "qa_runs.db"


def infer_execution_outcome(status: Any, current: Any) -> str:
    if isinstance(current, str) and current:
        return current
    if status in {"published", "failed_gate", "degraded_published"}:
        return "success"
    return "unknown"


def infer_publication_outcome(status: Any, current: Any) -> Any:
    if isinstance(current, str) and current:
        return current
    if status == "published":
        return "published"
    if status == "failed_gate":
        return "failed_gate"
    if status == "degraded_published":
        return "carry_forward"
    return status


def avg_trailing_rates(as_of_date: Any) -> tuple[float | None, float | None]:
    if not isinstance(as_of_date, str) or not QA_DB_PATH.exists():
        return None, None
    with sqlite3.connect(QA_DB_PATH) as conn:
        rows = conn.execute(
            "SELECT pass_rate_30d, freshness_pass_rate_30d "
            "FROM daily_source_reliability WHERE as_of_date = ?",
            (as_of_date,),
        ).fetchall()
    if not rows:
        return None, None
    contract = round(sum(float(row[0]) for row in rows) / len(rows), 4)
    freshness = round(sum(float(row[1]) for row in rows) / len(rows), 4)
    return contract, freshness


def enrich_payload(payload: dict) -> bool:
    changed = False

    release = payload.get("release")
    if not isinstance(release, dict):
        release = {}
        payload["release"] = release
        changed = True
    status = release.get("status")

    execution = infer_execution_outcome(status, release.get("execution_outcome"))
    if release.get("execution_outcome") != execution:
        release["execution_outcome"] = execution
        changed = True
    publication = infer_publication_outcome(status, release.get("publication_outcome"))
    if release.get("publication_outcome") != publication:
        release["publication_outcome"] = publication
        changed = True

    meta = payload.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        payload["meta"] = meta
        changed = True
    qa_summary = meta.get("qa_summary")
    if not isinstance(qa_summary, dict):
        qa_summary = {}
        meta["qa_summary"] = qa_summary
        changed = True

    checks = qa_summary.get("source_checks")
    if not isinstance(checks, list):
        checks = []

    fingerprint = qa_summary.get("failure_fingerprint")
    if not isinstance(fingerprint, dict) or not fingerprint:
        meta_fp = meta.get("qa_failure_fingerprint")
        if isinstance(meta_fp, dict) and meta_fp:
            fingerprint = meta_fp
        elif checks:
            fingerprint = process.build_qa_failure_fingerprint(checks)
        else:
            fingerprint = {}
    if fingerprint:
        if qa_summary.get("failure_fingerprint") != fingerprint:
            qa_summary["failure_fingerprint"] = fingerprint
            changed = True
        if meta.get("qa_failure_fingerprint") != fingerprint:
            meta["qa_failure_fingerprint"] = fingerprint
            changed = True

    this_contract = qa_summary.get("this_run_source_contract_pass_rate")
    if this_contract is None and qa_summary.get("source_contract_pass_rate") is not None:
        qa_summary["this_run_source_contract_pass_rate"] = qa_summary.get("source_contract_pass_rate")
        this_contract = qa_summary["this_run_source_contract_pass_rate"]
        changed = True

    this_freshness = qa_summary.get("this_run_source_freshness_pass_rate")
    if this_freshness is None and qa_summary.get("source_freshness_pass_rate") is not None:
        qa_summary["this_run_source_freshness_pass_rate"] = qa_summary.get("source_freshness_pass_rate")
        this_freshness = qa_summary["this_run_source_freshness_pass_rate"]
        changed = True

    trailing_contract, trailing_freshness = avg_trailing_rates(payload.get("as_of_date"))
    if qa_summary.get("trailing_30d_source_contract_pass_rate") is None:
        qa_summary["trailing_30d_source_contract_pass_rate"] = (
            trailing_contract if trailing_contract is not None else this_contract
        )
        changed = True
    if qa_summary.get("trailing_30d_source_freshness_pass_rate") is None:
        qa_summary["trailing_30d_source_freshness_pass_rate"] = (
            trailing_freshness if trailing_freshness is not None else this_freshness
        )
        changed = True

    return changed


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
