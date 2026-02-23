from __future__ import annotations

import unittest
from unittest.mock import patch

from scrapers.health_public import PMPRB_REPORTS_URLS, scrape_health_public


class HealthPublicTests(unittest.TestCase):
    def test_pmprb_uses_fallback_url_sequence(self) -> None:
        calls: list[str] = []

        def fake_fetch(url: str, timeout: int = 20, retries: int = 1):
            calls.append(url)
            if "dpd-bdpp" in url:
                return "price 22 33"
            if url == PMPRB_REPORTS_URLS[0]:
                raise RuntimeError("404")
            if url == PMPRB_REPORTS_URLS[1]:
                return "index 11 12"
            raise RuntimeError("should not hit third URL")

        with patch("scrapers.health_public.fetch_url", side_effect=fake_fetch):
            quotes, health = scrape_health_public()

        self.assertGreater(len(quotes), 0)
        pmprb_health = next(item for item in health if item.source == "pmprb_reports")
        self.assertIn(PMPRB_REPORTS_URLS[1], pmprb_health.detail)
        self.assertEqual("fresh", pmprb_health.status)
        self.assertIn(PMPRB_REPORTS_URLS[0], calls)
        self.assertIn(PMPRB_REPORTS_URLS[1], calls)

    def test_pmprb_missing_only_after_all_urls_fail(self) -> None:
        def always_fail_pmprb(url: str, timeout: int = 20, retries: int = 1):
            if "dpd-bdpp" in url:
                return "price 22"
            raise RuntimeError("down")

        with patch("scrapers.health_public.fetch_url", side_effect=always_fail_pmprb):
            _, health = scrape_health_public()

        pmprb_health = next(item for item in health if item.source == "pmprb_reports")
        self.assertEqual("missing", pmprb_health.status)
        self.assertIn("All PMPRB URLs failed", pmprb_health.detail)


if __name__ == "__main__":
    unittest.main()
