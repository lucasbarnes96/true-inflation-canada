from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scrapers import fetch_official_cpi_series


DEFAULT_DAYS = 365
DEFAULT_OUTPUT = Path("data/historical.json")


def _load_history(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _series_index(series: list[dict]) -> list[tuple[date, dict]]:
    indexed: list[tuple[date, dict]] = []
    for row in series:
        ref = row.get("ref_date")
        if not isinstance(ref, str):
            continue
        try:
            month_key = date.fromisoformat(f"{ref}-01")
        except ValueError:
            continue
        indexed.append((month_key, row))
    indexed.sort(key=lambda item: item[0])
    return indexed


def _series_for_day(day: date, indexed: list[tuple[date, dict]]) -> dict | None:
    if not indexed:
        return None
    month_key = date(day.year, day.month, 1)
    chosen: dict | None = None
    for ref_month, row in indexed:
        if ref_month <= month_key:
            chosen = row
        else:
            break
    return chosen or indexed[0][1]


def _build_seeded_row(day: date, official: dict) -> dict:
    return {
        "official_cpi": {
            "latest_release_month": official.get("ref_date"),
            "mom_pct": official.get("mom_pct"),
            "yoy_pct": official.get("yoy_pct"),
        },
        "meta": {
            "seeded": True,
            "seed_type": "official_monthly_baseline",
            "seed_source": "statcan_cpi_csv",
            "seeded_for_date": day.isoformat(),
        },
    }


def seed_history(days: int, output: Path, force: bool) -> tuple[int, int, int]:
    series = fetch_official_cpi_series()
    indexed = _series_index(series)
    if not indexed:
        raise RuntimeError("Could not load official CPI series.")

    history = _load_history(output)
    today = date.today()
    start = today - timedelta(days=days - 1)

    written = 0
    skipped_existing = 0
    missing_series = 0

    for offset in range(days):
        current_day = start + timedelta(days=offset)
        key = current_day.isoformat()

        existing = history.get(key)
        existing_seeded = isinstance(existing, dict) and isinstance(existing.get("meta"), dict) and bool(existing["meta"].get("seeded"))
        if isinstance(existing, dict) and not force and not existing_seeded:
            skipped_existing += 1
            continue

        official = _series_for_day(current_day, indexed)
        if official is None:
            missing_series += 1
            continue

        history[key] = _build_seeded_row(current_day, official)
        written += 1

    ordered = {k: history[k] for k in sorted(history.keys())}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(ordered, indent=2))
    return written, skipped_existing, missing_series


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed historical official CPI baseline data (official-only; no synthetic nowcast series)."
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Number of days to seed (default: 365).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output historical JSON path.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing non-seeded entries.")
    args = parser.parse_args()

    if args.days <= 0:
        raise SystemExit("--days must be greater than 0")

    written, skipped_existing, missing_series = seed_history(days=args.days, output=args.output, force=args.force)
    print(
        f"Seed complete: wrote={written}, skipped_non_seeded={skipped_existing}, missing_series_days={missing_series}, output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
