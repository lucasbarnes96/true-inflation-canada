from __future__ import annotations

import json
import re
import ssl
import socket
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse
from urllib.error import URLError

DEFAULT_TIMEOUT_SECONDS = 20
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
DNS_PREFLIGHT_HOSTS = (
    "www150.statcan.gc.ca",
    "www2.nrcan.gc.ca",
    "rentals.ca",
)

_DNS_PREFLIGHT_CACHE: dict[str, Any] = {
    "checked_at_epoch": None,
    "ok": None,
    "failures": [],
}


class FetchError(Exception):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _is_dns_error(err: Exception) -> bool:
    text = str(err).lower()
    if "nodename nor servname provided" in text or "name or service not known" in text:
        return True
    if isinstance(err, URLError):
        reason = getattr(err, "reason", None)
        if isinstance(reason, socket.gaierror):
            return True
        if reason and isinstance(reason, Exception):
            rtext = str(reason).lower()
            if "nodename nor servname provided" in rtext or "name or service not known" in rtext:
                return True
    return False


def dns_preflight(ttl_seconds: int = 300, hosts: tuple[str, ...] = DNS_PREFLIGHT_HOSTS) -> dict:
    now_epoch = int(time.time())
    checked = _DNS_PREFLIGHT_CACHE.get("checked_at_epoch")
    cached_ok = _DNS_PREFLIGHT_CACHE.get("ok")
    if isinstance(checked, int) and (now_epoch - checked) < ttl_seconds and cached_ok is not None:
        return {
            "checked_at_epoch": checked,
            "ok": bool(cached_ok),
            "failures": list(_DNS_PREFLIGHT_CACHE.get("failures") or []),
            "cached": True,
        }

    failures: list[dict] = []
    for host in hosts:
        try:
            socket.gethostbyname(host)
        except Exception as err:  # pragma: no cover - network dependent
            failures.append({"host": host, "error": str(err)})

    ok = len(failures) == 0
    _DNS_PREFLIGHT_CACHE["checked_at_epoch"] = now_epoch
    _DNS_PREFLIGHT_CACHE["ok"] = ok
    _DNS_PREFLIGHT_CACHE["failures"] = failures
    return {
        "checked_at_epoch": now_epoch,
        "ok": ok,
        "failures": failures,
        "cached": False,
    }


def fetch_url(
    url: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = 2,
    verify: bool = True,
    allowed_insecure_hosts: set[str] | None = None,
) -> str:
    if not verify:
        host = (urlparse(url).hostname or "").lower()
        if not host:
            raise FetchError(f"Cannot disable TLS verification for invalid URL host: {url}")
        allowed = {h.lower() for h in (allowed_insecure_hosts or set())}
        if host not in allowed:
            raise FetchError(f"Insecure TLS mode not permitted for host: {host}")

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            headers = {
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.google.com/",
            }
            req = urllib.request.Request(url, headers=headers)
            context = None
            if not verify:
                context = ssl._create_unverified_context()
            with urllib.request.urlopen(req, timeout=timeout, context=context) as response:
                return response.read().decode("utf-8", errors="ignore")
        except Exception as err:  # pragma: no cover - network dependent
            last_err = err
            if _is_dns_error(err):
                preflight = dns_preflight(ttl_seconds=60)
                if not preflight["ok"]:
                    failed = ", ".join(item["host"] for item in preflight.get("failures", [])[:3])
                    raise FetchError(
                        f"DNS resolver unavailable. Preflight failed for: {failed}. Original error: {err}"
                    )
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise FetchError(f"Failed to fetch URL: {url}: {last_err}")


def fetch_json(url: str, timeout: int = DEFAULT_TIMEOUT_SECONDS, retries: int = 2) -> Any:
    text = fetch_url(url, timeout=timeout, retries=retries)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise FetchError(f"Invalid JSON from {url}: {exc}") from exc


def parse_floats_from_text(text: str) -> list[float]:
    candidates = re.findall(r"(?<!\d)(\d{1,4}(?:\.\d{1,4})?)(?!\d)", text)
    values: list[float] = []
    for value in candidates:
        try:
            values.append(float(value))
        except ValueError:
            continue
    return values
