from __future__ import annotations

import unittest

from scrapers.consensus_free import fetch_consensus_estimate
from scrapers.release_calendar_statcan import fetch_release_events


class ReleaseConsensusTests(unittest.TestCase):
    def test_fetch_release_events_shape(self) -> None:
        payload = fetch_release_events()
        self.assertIn("events", payload)
        self.assertIsInstance(payload["events"], list)
        self.assertGreaterEqual(len(payload["events"]), 1)

    def test_fetch_consensus_shape(self) -> None:
        payload = fetch_consensus_estimate()
        self.assertIn("headline_yoy", payload)
        self.assertIn("source_count", payload)
        self.assertIn("confidence", payload)


if __name__ == "__main__":
    unittest.main()

