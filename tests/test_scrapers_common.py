from __future__ import annotations

import socket
import ssl
import unittest
from unittest.mock import patch
from urllib.error import URLError

from scrapers.common import FetchError, fetch_url


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read(self):
        return b"ok"


class ScrapersCommonTests(unittest.TestCase):
    def test_fetch_url_defaults_to_verified_tls(self) -> None:
        with patch("urllib.request.urlopen", return_value=_FakeResponse()) as urlopen:
            text = fetch_url("https://example.com", retries=0)
        self.assertEqual("ok", text)
        self.assertIn("context", urlopen.call_args.kwargs)
        self.assertIsInstance(urlopen.call_args.kwargs["context"], ssl.SSLContext)

    def test_fetch_url_rejects_insecure_mode(self) -> None:
        with self.assertRaises(FetchError) as ctx:
            fetch_url(
                "https://crtc.gc.ca/eng/publications/reports/policymonitoring/2024/index.htm",
                retries=0,
                verify=False,
            )
        self.assertIn("Insecure TLS mode is disabled", str(ctx.exception))

    def test_fetch_url_applies_host_retry_cap(self) -> None:
        side_effect = [URLError("temporary"), _FakeResponse()]
        with patch("urllib.request.urlopen", side_effect=side_effect) as urlopen:
            with patch("time.sleep", return_value=None):
                text = fetch_url("https://crtc.gc.ca/eng/publications/reports/policymonitoring/2024/index.htm", retries=3)
        self.assertEqual("ok", text)
        self.assertEqual(2, urlopen.call_count)

    def test_fetch_url_dns_outage_fast_fails_without_retries(self) -> None:
        dns_err = URLError(socket.gaierror(8, "nodename nor servname provided, or not known"))
        with patch("urllib.request.urlopen", side_effect=dns_err) as urlopen:
            with patch(
                "scrapers.common.dns_preflight",
                return_value={"ok": False, "failures": [{"host": "www150.statcan.gc.ca", "error": "dns"}]},
            ):
                with self.assertRaises(FetchError) as ctx:
                    fetch_url("https://example.com", retries=2)
        self.assertIn("DNS resolver unavailable", str(ctx.exception))
        self.assertEqual(1, urlopen.call_count)


if __name__ == "__main__":
    unittest.main()
