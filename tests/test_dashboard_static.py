from __future__ import annotations

import unittest
from pathlib import Path


class DashboardStaticTests(unittest.TestCase):
    def test_drivers_placeholder_copy_present(self) -> None:
        html = Path("index.html").read_text()
        self.assertIn('id="category-placeholder"', html)
        self.assertIn("Insufficient day-over-day history for category contribution ranking.", html)

    def test_yoy_terminology_present(self) -> None:
        html = Path("index.html").read_text()
        self.assertIn("Nowcast vs Official CPI (Year-over-Year)", html)
        self.assertIn('id="chart-maturity"', html)
        self.assertIn("Deviation from Expectations", html)
        self.assertIn("Calculation ID", html)
        self.assertIn("Experimental open-source nowcast using public data. Not official StatCan CPI.", html)
        self.assertIn("Calibration & Maturity", html)

    def test_about_and_readiness_sections_present(self) -> None:
        html = Path("index.html").read_text()
        self.assertIn("About This Project", html)
        self.assertIn('id="about-grid"', html)
        self.assertIn('id="category-readiness"', html)
        self.assertIn('id="live-history-progress"', html)
        self.assertIn("Category contribution = weight × daily_change_pct.", html)
        self.assertIn('href="/about"', html)
        self.assertIn('href="#source-diagnostics"', html)
        self.assertIn('id="sticky-disclaimer"', html)

    def test_footer_link_is_noopener(self) -> None:
        html = Path("index.html").read_text()
        self.assertIn('target="_blank" rel="noopener noreferrer"', html)

    def test_chart_js_is_pinned(self) -> None:
        html = Path("index.html").read_text()
        self.assertIn("cdn.jsdelivr.net/npm/chart.js@4.4.3", html)


if __name__ == "__main__":
    unittest.main()
