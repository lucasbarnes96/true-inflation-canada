from __future__ import annotations

import unittest
from pathlib import Path


class AboutStaticTests(unittest.TestCase):
    def test_about_page_exists_with_core_sections(self) -> None:
        html = Path("about.html").read_text()
        self.assertIn("Transparent Inflation Nowcast Built on Public Data", html)
        self.assertIn("How the Math Works", html)
        self.assertIn("Data Source Transparency (All Tracked Sources)", html)
        self.assertIn("Why Some Signals Use Fallback or Reused Data", html)
        self.assertIn("Current Status", html)
        self.assertIn("Ingestion State", html)
        self.assertIn("Known Shortcomings", html)
        self.assertIn("Long-Term Vision", html)

    def test_about_page_links_are_safe(self) -> None:
        html = Path("about.html").read_text()
        self.assertIn('target="_blank" rel="noopener noreferrer"', html)
        self.assertIn('href="/"', html)


if __name__ == "__main__":
    unittest.main()
