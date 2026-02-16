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
        self.assertIn("Deviation from Expectations", html)
        self.assertIn("Calculation ID", html)
        self.assertIn("Experimental open-source nowcast using public data. Not official StatCan CPI.", html)


if __name__ == "__main__":
    unittest.main()
