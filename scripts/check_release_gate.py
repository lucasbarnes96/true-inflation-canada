from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gate_policy import GATE_POLICY


def _count_live_days(history: dict) -> int:
    count = 0
    for payload in history.values():
        if not isinstance(payload, dict):
            continue
        headline = payload.get("headline", {})
        meta = payload.get("meta", {})
        seeded = isinstance(meta, dict) and bool(meta.get("seeded"))
        if seeded:
            continue
        if isinstance(headline, dict) and headline.get("nowcast_yoy_pct") is not None:
            count += 1
    return count


def main() -> int:
    rep_threshold = float(GATE_POLICY["representativeness_min_fresh_ratio"])
    parser = argparse.ArgumentParser(description="Check release and launch readiness gates.")
    parser.add_argument("--min-coverage", type=float, default=0.80, help="Minimum headline coverage ratio.")
    parser.add_argument("--min-live-days", type=int, default=30, help="Minimum authentic live nowcast days.")
    parser.add_argument(
        "--enforce-live-days",
        action="store_true",
        help="Fail the check when live nowcast days are below --min-live-days.",
    )
    parser.add_argument(
        "--min-representativeness",
        type=float,
        default=rep_threshold,
        help="Minimum fresh basket representativeness ratio.",
    )
    parser.add_argument(
        "--strict-official-parity",
        action="store_true",
        help="Fail if official CPI one-decimal display does not match rounded source value.",
    )
    args = parser.parse_args()

    latest_path = Path("data/latest.json")
    historical_path = Path("data/historical.json")
    if not latest_path.exists():
        print("Gate check failed: data/latest.json not found.")
        return 1
    if not historical_path.exists():
        print("Gate check failed: data/historical.json not found.")
        return 1

    payload = json.loads(latest_path.read_text())
    historical = json.loads(historical_path.read_text())
    release = payload.get("release", {})
    headline = payload.get("headline", {})
    official = payload.get("official_cpi", {})
    consensus = payload.get("meta", {}).get("consensus", {})
    status = release.get("status")
    blocked = release.get("blocked_conditions", [])
    errors: list[str] = []

    print(f"Release status: {status}")
    if blocked:
        print("Blocked conditions:")
        for reason in blocked:
            print(f"- {reason}")

    coverage = headline.get("coverage_ratio")
    if not isinstance(coverage, (int, float)) or float(coverage) < float(args.min_coverage):
        errors.append(f"Coverage below threshold: {coverage} < {args.min_coverage}")

    representativeness = payload.get("meta", {}).get("representativeness_ratio")
    if not isinstance(representativeness, (int, float)) or float(representativeness) < float(args.min_representativeness):
        errors.append(
            "Representativeness below threshold: "
            f"{representativeness} < {args.min_representativeness}"
        )

    live_days = _count_live_days(historical if isinstance(historical, dict) else {})
    if live_days < args.min_live_days:
        msg = f"Live nowcast history short: {live_days} < {args.min_live_days} days (soft-launch informational)"
        if args.enforce_live_days:
            errors.append(msg)
        else:
            print(f"Info: {msg}")

    # Consensus is optional; if present, it must pass quality checks.
    consensus_yoy = headline.get("consensus_yoy")
    if consensus_yoy is not None:
        conf = consensus.get("confidence")
        source_count = consensus.get("source_count", 0)
        if conf not in {"medium", "high"} or not isinstance(source_count, int) or source_count < 2:
            errors.append("Consensus present but below quality threshold (confidence/source_count).")

    if args.strict_official_parity:
        yoy = official.get("yoy_pct")
        yoy_display = official.get("yoy_display_pct")
        expected = round(float(yoy), 1) if yoy is not None else None
        if expected is None or yoy_display is None or float(yoy_display) != float(expected):
            errors.append(f"Official CPI display parity failed: yoy_pct={yoy}, yoy_display_pct={yoy_display}, expected={expected}")

    if errors:
        print("Launch gate failures:")
        for err in errors:
            print(f"- {err}")

    if status != "published":
        return 1
    if errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
