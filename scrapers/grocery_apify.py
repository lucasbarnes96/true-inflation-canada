"""Grocery scraper using Apify actors with redundancy and schema checks."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import date, datetime, timezone
from typing import Any

try:
    from apify_client import ApifyClient
except ImportError:
    ApifyClient = None

from .common import utc_now_iso
from .types import Quote, SourceHealth

DEFAULT_ACTOR_IDS = [
    "sunny_eternity/loblaws-grocery-scraper",
    "ko_red/loblaws-grocery-scraper",
]
DEFAULT_CATEGORY_URL = "https://www.realcanadiansuperstore.ca/food/dairy-eggs/c/28003"
DEFAULT_BANNER = "superstore"


def _load_env_value(key: str) -> str | None:
    value = os.getenv(key)
    if value:
        return value
    if os.path.exists(".env"):
        try:
            with open(".env", "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line.startswith(f"{key}="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            return None
    return None


def _load_token() -> str | None:
    return _load_env_value("APIFY_TOKEN")


def _env_flag(name: str, default: bool = True) -> bool:
    value = (_load_env_value(name) or "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _load_previous_apify_timestamp() -> str | None:
    for path in ("data/published_latest.json", "data/latest.json"):
        if not os.path.exists(path):
            continue
        try:
            payload = json.loads(open(path, "r", encoding="utf-8").read())
        except Exception:
            continue
        for row in payload.get("source_health", []):
            if row.get("source") == "apify_loblaws":
                ts = row.get("last_success_timestamp")
                if ts:
                    return ts
    return None


def _actor_ids() -> list[str]:
    raw = (_load_env_value("APIFY_ACTOR_IDS") or "").strip()
    if not raw:
        return DEFAULT_ACTOR_IDS
    values = [x.strip() for x in raw.split(",")]
    return [x for x in values if x]


def _category_url_candidates(category_url: str) -> list[str]:
    values: list[str] = []
    normalized = category_url.strip()
    if normalized:
        values.append(normalized)

    if "://www.realcanadiansuperstore.ca/food/" in normalized:
        values.append(normalized.replace("://www.realcanadiansuperstore.ca/food/", "://www.realcanadiansuperstore.ca/en/food/"))
    if "://www.realcanadiansuperstore.ca/en/en/" in normalized:
        values.append(normalized.replace("/en/en/", "/en/"))

    deduped: list[str] = []
    for value in values:
        if value and value not in deduped:
            deduped.append(value)
    return deduped


def _parse_price(item: dict[str, Any]) -> float | None:
    price = item.get("price")
    if isinstance(price, dict):
        for key in ("value", "amount", "current"):
            value = price.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    elif isinstance(price, str):
        match = re.search(r"(\d+(?:\.\d+)?)", price)
        if match:
            try:
                return float(match.group(1))
            except (TypeError, ValueError):
                pass
    elif price is not None:
        try:
            return float(price)
        except (TypeError, ValueError):
            pass

    for key in ("currentPrice", "salePrice", "priceValue"):
        value = item.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    for key in ("comparable_unit_price",):
        value = item.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    parsed_unit_price = item.get("parsed_unit_price")
    if isinstance(parsed_unit_price, dict):
        value = parsed_unit_price.get("value")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def _parse_name(item: dict[str, Any]) -> str | None:
    for key in ("name", "title", "productName", "displayName"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _parse_unit(item: dict[str, Any]) -> str:
    for key in ("package_size", "packageSize", "size", "unitText", "unit"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            cleaned = re.sub(r"\s+", " ", value.strip())
            return cleaned.lower()
    normalized = item.get("normalized_package_size")
    if isinstance(normalized, dict):
        size = normalized.get("size")
        unit = normalized.get("unit")
        if size is not None and unit:
            return f"{size}{unit}".lower()
    return "unknown_unit"


def normalize_apify_item(item: dict[str, Any], observed: date, source_run_id: str) -> Quote | None:
    name = _parse_name(item)
    value = _parse_price(item)
    if not name or value is None or value <= 0:
        return None

    unit = _parse_unit(item)
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    canonical = f"{base}|{unit}"
    suffix = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:10]
    item_id = f"{base[:50]}-{suffix}"

    return Quote(
        category="food",
        item_id=item_id,
        value=float(value),
        observed_at=observed,
        source="apify_loblaws",
        source_run_id=source_run_id,
    )


def scrape_grocery_apify() -> tuple[list[Quote], list[SourceHealth]]:
    """Run Apify actor(s) with deterministic fallback."""
    if not _env_flag("APIFY_ENABLED", default=True):
        previous_ts = _load_previous_apify_timestamp()
        status = "stale" if previous_ts else "missing"
        detail = "APIFY run skipped by schedule (APIFY_ENABLED=false)."
        if previous_ts:
            detail += " Reusing last successful APIFY timestamp for freshness gating."
        return [], [
            SourceHealth(
                source="apify_loblaws",
                category="food",
                tier=1,
                status=status,
                last_success_timestamp=previous_ts,
                detail=detail,
                source_run_id=None,
            )
        ]

    token = _load_token()

    if not token:
        return [], [
            SourceHealth(
                source="apify_loblaws",
                category="food",
                tier=1,
                status="missing",
                last_success_timestamp=None,
                detail="APIFY_TOKEN not found in environment.",
                source_run_id=None,
            )
        ]

    if ApifyClient is None:
        return [], [
            SourceHealth(
                source="apify_loblaws",
                category="food",
                tier=1,
                status="missing",
                last_success_timestamp=None,
                detail="apify-client library not installed.",
                source_run_id=None,
            )
        ]

    if sys.version_info >= (3, 13):
        previous_ts = _load_previous_apify_timestamp()
        status = "stale" if previous_ts else "missing"
        detail = "Python 3.13 is not supported for apify-client in this project. Use Python 3.11."
        if previous_ts:
            detail += " Reusing last successful APIFY timestamp for freshness gating."
        return [], [
            SourceHealth(
                source="apify_loblaws",
                category="food",
                tier=1,
                status=status,
                last_success_timestamp=previous_ts,
                detail=detail,
                source_run_id=None,
            )
        ]

    quotes: list[Quote] = []
    errors: list[str] = []
    observed = datetime.now(timezone.utc).date()
    category_url = _load_env_value("APIFY_CATEGORY_URL") or DEFAULT_CATEGORY_URL
    category_url_candidates = _category_url_candidates(category_url)
    banner = (_load_env_value("APIFY_BANNER") or DEFAULT_BANNER).strip().lower()
    location_id = (_load_env_value("APIFY_LOCATION_ID") or "").strip()
    max_items_raw = _load_env_value("APIFY_MAX_ITEMS") or "50"
    try:
        max_items = int(max_items_raw)
    except ValueError:
        max_items = 50
    actor_ids = _actor_ids()
    source_run_id: str | None = None

    try:
        # NOTE: apify-client currently crashes on Python 3.13 in some envs.
        # Pinning runtime to 3.11 in CI/prod avoids this failure mode.
        client = ApifyClient(token)
    except BaseException as err:  # pragma: no cover - runtime dependent
        previous_ts = _load_previous_apify_timestamp()
        status = "stale" if previous_ts else "missing"
        detail = f"Apify client init failed: {err}"
        if previous_ts:
            detail += " Reusing last successful APIFY timestamp for freshness gating."
        return [], [
            SourceHealth(
                source="apify_loblaws",
                category="food",
                tier=1,
                status=status,
                last_success_timestamp=previous_ts,
                detail=detail,
                source_run_id=None,
            )
        ]

    for actor_id in actor_ids:
        for category_url_try in category_url_candidates:
            run_input = {
                "banner": banner,
                "categoryUrl": category_url_try,
                "maxItems": max_items,
                "proxyConfig": {"useApifyProxy": True},
            }
            if location_id:
                run_input["locationId"] = location_id

            try:
                run = client.actor(actor_id).call(run_input=run_input)
                source_run_id = str(run.get("id") or run.get("defaultDatasetId") or "")
                dataset_id = run.get("defaultDatasetId")
                if not dataset_id:
                    errors.append(f"{actor_id}: missing defaultDatasetId")
                    continue
                dataset_items = client.dataset(dataset_id).list_items(limit=max_items).items

                for item in dataset_items:
                    if not isinstance(item, dict):
                        continue
                    quote = normalize_apify_item(item, observed=observed, source_run_id=source_run_id or "")
                    if quote is not None:
                        quotes.append(quote)

                if quotes:
                    return quotes, [
                        SourceHealth(
                            source="apify_loblaws",
                            category="food",
                            tier=1,
                            status="fresh",
                            last_success_timestamp=utc_now_iso(),
                            detail=f"Collected {len(quotes)} prices from actor {actor_id} (banner={banner}, category={category_url_try}, location={location_id or 'default'}).",
                            source_run_id=source_run_id,
                        )
                    ]
                errors.append(f"{actor_id}: no valid records for category={category_url_try} location={location_id or 'default'}")
            except BaseException as err:  # pragma: no cover - network/runtime dependent
                errors.append(f"{actor_id}: {err}")

    previous_ts = _load_previous_apify_timestamp()
    detail = "Apify run failed for all actors."
    if errors:
        detail = detail + " " + " | ".join(errors[:3])
    status = "stale" if previous_ts else "missing"
    if previous_ts:
        detail += " Reusing last successful APIFY timestamp for freshness gating."
    return [], [
        SourceHealth(
            source="apify_loblaws",
            category="food",
            tier=1,
            status=status,
            last_success_timestamp=previous_ts,
            detail=detail,
            source_run_id=source_run_id,
        )
    ]
