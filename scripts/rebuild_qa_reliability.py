from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import process as process_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild QA reliability and fingerprints.")
    parser.add_argument("--from-date", dest="from_date", help="Inclusive YYYY-MM-DD lower bound.")
    parser.add_argument("--to-date", dest="to_date", help="Inclusive YYYY-MM-DD upper bound.")
    parser.add_argument(
        "--drop-derived",
        action="store_true",
        help="Rebuild only derived tables from existing source_run_checks/source_check_events.",
    )
    return parser.parse_args()


def in_range(as_of_date: str, from_date: str | None, to_date: str | None) -> bool:
    if from_date and as_of_date < from_date:
        return False
    if to_date and as_of_date > to_date:
        return False
    return True


def load_runs(from_date: str | None, to_date: str | None) -> list[dict]:
    runs: list[dict] = []
    for path in sorted(process_module.RUNS_DIR.glob("run_*.json")):
        payload = process_module.load_json(path, {})
        if not isinstance(payload, dict):
            continue
        as_of_date = payload.get("as_of_date")
        if not isinstance(as_of_date, str):
            continue
        if not in_range(as_of_date, from_date, to_date):
            continue
        release = payload.get("release", {})
        if not isinstance(release, dict):
            continue
        run_id = release.get("run_id")
        created_at = release.get("created_at")
        if not isinstance(run_id, str) or not isinstance(created_at, str):
            continue
        checks = payload.get("meta", {}).get("qa_summary", {}).get("source_checks", [])
        source_health = payload.get("source_health", [])
        if not isinstance(checks, list) or not isinstance(source_health, list):
            continue
        runs.append(
            {
                "run_id": run_id,
                "created_at": created_at,
                "as_of_date": as_of_date,
                "checks": checks,
                "source_health": source_health,
            }
        )
    runs.sort(key=lambda row: row["created_at"])
    return runs


def recompute_daily_reliability(conn: sqlite3.Connection, as_of_dates: list[str]) -> None:
    for as_of_date in sorted(set(as_of_dates)):
        window_start = (date.fromisoformat(as_of_date) - timedelta(days=30)).isoformat()
        rows = conn.execute(
            "SELECT source, COUNT(*) AS runs, AVG(passed * 1.0) AS pass_rate, "
            "AVG(CASE "
            "WHEN freshness_passed IS NOT NULL THEN freshness_passed * 1.0 "
            "WHEN freshness_hours IS NOT NULL AND freshness_sla_hours IS NOT NULL AND freshness_hours <= freshness_sla_hours THEN 1.0 "
            "WHEN freshness_hours IS NOT NULL AND freshness_hours <= 48 THEN 1.0 "
            "ELSE 0.0 END) AS freshness_rate "
            "FROM source_run_checks WHERE DATE(created_at) >= DATE(?) AND DATE(created_at) <= DATE(?) GROUP BY source",
            (window_start, as_of_date),
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


def rebuild_fingerprints(conn: sqlite3.Connection, run_ids: list[str]) -> None:
    for run_id in sorted(set(run_ids)):
        rows = conn.execute(
            "SELECT created_at, as_of_date, source, category, attempts, details_json FROM source_run_checks WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        if not rows:
            continue
        created_at = str(rows[0][0])
        as_of_date = str(rows[0][1])
        checks: list[dict] = []
        for _, _, source, category, attempts, details_json in rows:
            try:
                payload = json.loads(details_json)
            except Exception:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            payload["source"] = source
            payload["category"] = category
            payload["attempts"] = attempts
            checks.append(payload)
        fingerprint = process_module.build_qa_failure_fingerprint(checks)
        conn.execute(
            "INSERT OR REPLACE INTO qa_failure_fingerprints (run_id, created_at, as_of_date, total_source_checks, failed_source_checks, failed_check_events, by_check_json, by_source_json, by_category_json, top_failed_check, validator_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                created_at,
                as_of_date,
                int(fingerprint.get("total_source_checks") or 0),
                int(fingerprint.get("failed_source_checks") or 0),
                int(fingerprint.get("failed_check_events") or 0),
                json.dumps(fingerprint.get("by_check", {})),
                json.dumps(fingerprint.get("by_source", {})),
                json.dumps(fingerprint.get("by_category", {})),
                fingerprint.get("top_failed_check"),
                process_module.METHOD_VERSION,
            ),
        )


def main() -> int:
    args = parse_args()
    process_module.ensure_qa_db()

    runs = load_runs(args.from_date, args.to_date)
    with sqlite3.connect(process_module.QA_DB_PATH) as conn:
        if args.drop_derived:
            if args.from_date or args.to_date:
                where = "WHERE DATE(as_of_date) >= DATE(?) AND DATE(as_of_date) <= DATE(?)"
                conn.execute(
                    f"DELETE FROM daily_source_reliability {where}",
                    (args.from_date or "0001-01-01", args.to_date or "9999-12-31"),
                )
                conn.execute(
                    f"DELETE FROM qa_failure_fingerprints {where}",
                    (args.from_date or "0001-01-01", args.to_date or "9999-12-31"),
                )
            else:
                conn.execute("DELETE FROM daily_source_reliability")
                conn.execute("DELETE FROM qa_failure_fingerprints")
            run_rows = conn.execute(
                "SELECT DISTINCT run_id FROM source_run_checks WHERE DATE(as_of_date) >= DATE(?) AND DATE(as_of_date) <= DATE(?)",
                (args.from_date or "0001-01-01", args.to_date or "9999-12-31"),
            ).fetchall()
            as_of_rows = conn.execute(
                "SELECT DISTINCT as_of_date FROM source_run_checks WHERE DATE(as_of_date) >= DATE(?) AND DATE(as_of_date) <= DATE(?)",
                (args.from_date or "0001-01-01", args.to_date or "9999-12-31"),
            ).fetchall()
            run_ids = [str(row[0]) for row in run_rows]
            as_of_dates = [str(row[0]) for row in as_of_rows]
            recompute_daily_reliability(conn, as_of_dates)
            rebuild_fingerprints(conn, run_ids)
            conn.commit()
            print(f"Rebuilt derived QA tables for {len(run_ids)} runs.")
            return 0

        # Full rebuild from raw run artifacts.
        if args.from_date or args.to_date:
            params = (args.from_date or "0001-01-01", args.to_date or "9999-12-31")
            conn.execute("DELETE FROM source_run_checks WHERE DATE(as_of_date) >= DATE(?) AND DATE(as_of_date) <= DATE(?)", params)
            conn.execute("DELETE FROM source_check_events WHERE DATE(as_of_date) >= DATE(?) AND DATE(as_of_date) <= DATE(?)", params)
            conn.execute("DELETE FROM daily_source_reliability WHERE DATE(as_of_date) >= DATE(?) AND DATE(as_of_date) <= DATE(?)", params)
            conn.execute("DELETE FROM qa_failure_fingerprints WHERE DATE(as_of_date) >= DATE(?) AND DATE(as_of_date) <= DATE(?)", params)
        else:
            conn.execute("DELETE FROM source_run_checks")
            conn.execute("DELETE FROM source_check_events")
            conn.execute("DELETE FROM daily_source_reliability")
            conn.execute("DELETE FROM qa_failure_fingerprints")
        conn.commit()

    for run in runs:
        process_module.record_qa_checks(
            run_id=run["run_id"],
            created_at=run["created_at"],
            checks=run["checks"],
            source_health=run["source_health"],
            as_of_date=run["as_of_date"],
        )
    print(f"Rebuilt QA tables from {len(runs)} run snapshots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
