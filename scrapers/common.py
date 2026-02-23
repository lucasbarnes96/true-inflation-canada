from __future__ import annotations

import json
import re
import socket
import ssl
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse
from urllib.error import URLError

try:  # pragma: no cover - dependency availability is environment-specific
    import certifi
except Exception:  # pragma: no cover
    certifi = None  # type: ignore[assignment]

DEFAULT_TIMEOUT_SECONDS = 20
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
DNS_PREFLIGHT_HOSTS = (
    "www150.statcan.gc.ca",
    "www2.nrcan.gc.ca",
    "rentals.ca",
)
MAX_FETCH_RETRIES = 3
MAX_FETCH_BACKOFF_SECONDS = 6.0
HOST_RETRY_CAPS = {
    "crtc.gc.ca": 1,
    "www2.nrcan.gc.ca": 1,
    "www150.statcan.gc.ca": 2,
}
HOST_BACKOFF_SECONDS = {
    "crtc.gc.ca": 1.0,
    "www2.nrcan.gc.ca": 1.0,
    "www150.statcan.gc.ca": 1.5,
}

_DNS_PREFLIGHT_CACHE: dict[str, Any] = {
    "checked_at_epoch": None,
    "ok": None,
    "failures": [],
}


class FetchError(Exception):
    pass


def tls_context() -> ssl.SSLContext:
    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


TLS_CONTEXT = tls_context()


def tls_trust_store_info() -> dict[str, str]:
    if certifi is not None:
        return {"mode": "certifi", "cafile": certifi.where()}
    return {"mode": "system_default", "cafile": ""}


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
    # Keep compatibility args but do not allow insecure TLS transport.
    _ = allowed_insecure_hosts
    if not verify:
        host = (urlparse(url).hostname or "").lower()
        raise FetchError(f"Insecure TLS mode is disabled for host: {host or 'unknown'}")

    host = (urlparse(url).hostname or "").lower()
    retries = max(0, min(int(retries), MAX_FETCH_RETRIES))
    for suffix, retry_cap in HOST_RETRY_CAPS.items():
        if host.endswith(suffix):
            retries = min(retries, int(retry_cap))
            break
    backoff_base = 1.5
    for suffix, host_backoff in HOST_BACKOFF_SECONDS.items():
        if host.endswith(suffix):
            backoff_base = float(host_backoff)
            break
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
            with urllib.request.urlopen(req, timeout=timeout, context=TLS_CONTEXT) as response:
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
                sleep_seconds = min(MAX_FETCH_BACKOFF_SECONDS, backoff_base * (attempt + 1))
                time.sleep(sleep_seconds)
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
