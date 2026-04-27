import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "chart_data.json"

REQUIRED_ADJUSTERS = {
    "M1+",
    "M1++",
    "M3",
    "CPI",
}

REQUIRED_ASSETS = {
    "TSX",
    "Canadian REITs",
    "Bitcoin (CAD)",
    "Ethereum (CAD)",
    "Crude Oil",
    "S&P 500 (CAD)",
    "NASDAQ (CAD)",
    "Dow Jones (CAD)",
    "Gold (CAD)",
    "Silver (CAD)",
    "Canadian House Prices (NHPI)",
    "Labour Productivity",
}

YAHOO_ASSETS = {
    "TSX",
    "Canadian REITs",
    "Bitcoin (CAD)",
    "Ethereum (CAD)",
    "Crude Oil",
    "S&P 500 (CAD)",
    "NASDAQ (CAD)",
    "Dow Jones (CAD)",
    "Gold (CAD)",
    "Silver (CAD)",
}

STATCAN_ASSETS = {
    "Canadian House Prices (NHPI)",
    "Labour Productivity",
}

MAX_STALENESS_DAYS = {
    "adjusters": 120,
    "yahoo_assets": 120,
    "statcan_monthly_assets": 120,
    "statcan_quarterly_assets": 240,
}


def parse_updated_at(value: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("metadata.updated_at must be a non-empty string")
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_series_month(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Series date entries must be strings")
    return datetime.strptime(value, "%Y-%m").replace(tzinfo=UTC)


def validate_numeric_list(values, label: str):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be a non-empty list")
    for index, value in enumerate(values):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{label}[{index}] must be a finite number")


def validate_series(name: str, series: dict, expect_normalized: bool):
    if not isinstance(series, dict):
        raise ValueError(f"{name} series must be an object")

    dates = series.get("dates")
    values = series.get("values")
    if not isinstance(dates, list) or not dates:
        raise ValueError(f"{name}.dates must be a non-empty list")
    validate_numeric_list(values, f"{name}.values")
    if len(dates) != len(values):
        raise ValueError(f"{name} has mismatched dates/values lengths")

    parsed_dates = [parse_series_month(item) for item in dates]
    if parsed_dates != sorted(parsed_dates):
        raise ValueError(f"{name} dates must be sorted ascending")

    if expect_normalized:
        normalized = series.get("normalized")
        validate_numeric_list(normalized, f"{name}.normalized")
        if len(normalized) != len(dates):
            raise ValueError(f"{name} has mismatched dates/normalized lengths")

    return parsed_dates[-1]


def validate_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Payload root must be an object")

    metadata = payload.get("metadata")
    adjusters = payload.get("adjusters")
    assets = payload.get("assets")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    if not isinstance(adjusters, dict):
        raise ValueError("adjusters must be an object")
    if not isinstance(assets, dict):
        raise ValueError("assets must be an object")

    updated_at = parse_updated_at(metadata.get("updated_at"))

    missing_adjusters = sorted(REQUIRED_ADJUSTERS - set(adjusters))
    missing_assets = sorted(REQUIRED_ASSETS - set(assets))
    if missing_adjusters:
        raise ValueError(f"Missing required adjusters: {', '.join(missing_adjusters)}")
    if missing_assets:
        raise ValueError(f"Missing required assets: {', '.join(missing_assets)}")

    latest_adjuster_dt = None
    for name, series in adjusters.items():
        latest_dt = validate_series(name, series, expect_normalized=True)
        latest_adjuster_dt = max(latest_adjuster_dt, latest_dt) if latest_adjuster_dt else latest_dt

    latest_yahoo_dt = None
    latest_statcan_monthly_dt = None
    latest_statcan_quarterly_dt = None
    for name, series in assets.items():
        latest_dt = validate_series(name, series, expect_normalized=False)
        if name in YAHOO_ASSETS:
            latest_yahoo_dt = max(latest_yahoo_dt, latest_dt) if latest_yahoo_dt else latest_dt
        elif name == "Labour Productivity":
            latest_statcan_quarterly_dt = (
                max(latest_statcan_quarterly_dt, latest_dt)
                if latest_statcan_quarterly_dt
                else latest_dt
            )
        elif name in STATCAN_ASSETS:
            latest_statcan_monthly_dt = (
                max(latest_statcan_monthly_dt, latest_dt)
                if latest_statcan_monthly_dt
                else latest_dt
            )

    staleness = {}
    if latest_adjuster_dt is None or latest_yahoo_dt is None or latest_statcan_monthly_dt is None or latest_statcan_quarterly_dt is None:
        raise ValueError("Could not compute source-family latest dates")

    staleness["adjusters"] = (updated_at - latest_adjuster_dt).days
    staleness["yahoo_assets"] = (updated_at - latest_yahoo_dt).days
    staleness["statcan_monthly_assets"] = (updated_at - latest_statcan_monthly_dt).days
    staleness["statcan_quarterly_assets"] = (updated_at - latest_statcan_quarterly_dt).days

    for family, max_days in MAX_STALENESS_DAYS.items():
        if staleness[family] > max_days:
            raise ValueError(
                f"{family} is stale by {staleness[family]} days (limit {max_days})"
            )

    return {
        "publish_allowed": True,
        "required_adjusters_expected": len(REQUIRED_ADJUSTERS),
        "required_adjusters_present": len(REQUIRED_ADJUSTERS),
        "required_assets_expected": len(REQUIRED_ASSETS),
        "required_assets_present": len(REQUIRED_ASSETS),
        "latest_dates": {
            "adjusters": latest_adjuster_dt.strftime("%Y-%m"),
            "yahoo_assets": latest_yahoo_dt.strftime("%Y-%m"),
            "statcan_assets": max(latest_statcan_monthly_dt, latest_statcan_quarterly_dt).strftime("%Y-%m"),
        },
        "staleness_days": staleness,
    }


def load_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description="Validate chart_data.json for publish safety.")
    parser.add_argument(
        "--path",
        default=str(DATA_PATH),
        help="Path to the payload JSON file.",
    )
    args = parser.parse_args()

    payload_path = Path(args.path)
    payload = load_payload(payload_path)
    summary = validate_payload(payload)
    print(
        "Validation summary: "
        f"required_adjusters={summary['required_adjusters_present']}/{summary['required_adjusters_expected']}, "
        f"required_assets={summary['required_assets_present']}/{summary['required_assets_expected']}, "
        f"latest_adjuster={summary['latest_dates']['adjusters']}, "
        f"latest_yahoo={summary['latest_dates']['yahoo_assets']}, "
        f"latest_statcan={summary['latest_dates']['statcan_assets']}, "
        f"publish_allowed={summary['publish_allowed']}"
    )


if __name__ == "__main__":
    main()
