from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process import SCRAPER_REGISTRY, SOURCE_SLA_DAYS, recompute_source_health, source_age_days
from scrapers.common import dns_preflight


def classify_detail(detail: str) -> str:
    text = (detail or "").lower()
    if "dns resolver unavailable" in text or "nodename nor servname" in text or "name or service not known" in text:
        return "dns"
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if "403" in text or "forbidden" in text or "anti-bot" in text:
        return "blocked"
    if "python 3.13 is not supported" in text:
        return "runtime"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose source scraper health and network preflight.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    preflight = dns_preflight(ttl_seconds=0)
    rows: list[dict] = []
    raw_health = []
    raw_by_source: dict[str, dict] = {}
    for name, scraper in SCRAPER_REGISTRY:
        started = time.time()
        try:
            _, health = scraper()
            elapsed_ms = int((time.time() - started) * 1000)
            for item in health:
                raw_health.append(item)
                raw_by_source[item.source] = {
                    "scraper": name,
                    "detail": item.detail,
                    "elapsed_ms": elapsed_ms,
                }
        except Exception as err:  # pragma: no cover - runtime dependent
            elapsed_ms = int((time.time() - started) * 1000)
            rows.append(
                {
                    "scraper": name,
                    "source": "<scraper_error>",
                    "category": None,
                    "status": "missing",
                    "age_days": None,
                    "sla_days": None,
                    "valid_within_sla": False,
                    "error_class": classify_detail(str(err)),
                    "detail": str(err),
                    "elapsed_ms": elapsed_ms,
                }
            )

    recomputed = recompute_source_health(raw_health, now)
    for item in recomputed:
        source = item.get("source")
        detail = str(item.get("detail") or "")
        meta = raw_by_source.get(source, {})
        age = source_age_days(item.get("last_success_timestamp"), now=now)
        sla = SOURCE_SLA_DAYS.get(source)
        valid = (item.get("status") == "fresh") or (age is not None and sla is not None and age <= sla)
        rows.append(
            {
                "scraper": meta.get("scraper"),
                "source": source,
                "category": item.get("category"),
                "status": item.get("status"),
                "age_days": age,
                "sla_days": sla,
                "valid_within_sla": valid,
                "error_class": classify_detail(detail),
                "detail": detail,
                "elapsed_ms": meta.get("elapsed_ms"),
            }
        )

    summary = {
        "timestamp": now.isoformat(),
        "python_version": platform.python_version(),
        "dns_preflight": preflight,
        "total_sources": len(rows),
        "valid_sources": sum(1 for r in rows if r["valid_within_sla"]),
        "errors_by_class": {
            "dns": sum(1 for r in rows if r["error_class"] == "dns"),
            "timeout": sum(1 for r in rows if r["error_class"] == "timeout"),
            "blocked": sum(1 for r in rows if r["error_class"] == "blocked"),
            "runtime": sum(1 for r in rows if r["error_class"] == "runtime"),
            "other": sum(1 for r in rows if r["error_class"] == "other"),
        },
        "rows": rows,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Python: {summary['python_version']}")
        print(f"DNS preflight ok: {preflight.get('ok')} failures={len(preflight.get('failures', []))}")
        print(f"Valid sources (within SLA): {summary['valid_sources']}/{summary['total_sources']}")
        print("Error classes:", summary["errors_by_class"])
        for row in rows:
            print(
                f"{row['source']}: status={row['status']} age={row['age_days']} sla={row['sla_days']} "
                f"valid={row['valid_within_sla']} class={row['error_class']} detail={row['detail']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
