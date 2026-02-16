from __future__ import annotations

import json
import runpy
import tempfile
import unittest
from datetime import date
from pathlib import Path


SCRIPT_PATH = Path("scripts/seed_history.py")


class SeedHistoryTests(unittest.TestCase):
    def _load_module(self) -> dict:
        return runpy.run_path(str(SCRIPT_PATH))

    def test_seed_history_writes_rows_with_provenance(self) -> None:
        module = self._load_module()
        seed_history = module["seed_history"]
        seed_history.__globals__["fetch_official_cpi_series"] = lambda: [
            {"ref_date": "2025-12", "index_value": 160.0, "mom_pct": -0.1, "yoy_pct": 2.0},
            {"ref_date": "2026-01", "index_value": 161.0, "mom_pct": 0.2, "yoy_pct": 2.2},
            {"ref_date": "2026-02", "index_value": 162.0, "mom_pct": 0.3, "yoy_pct": 2.4},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "historical.json"
            written, skipped, missing = seed_history(days=365, output=out, force=False)
            self.assertGreaterEqual(written, 300)
            self.assertEqual(0, skipped)
            self.assertEqual(0, missing)

            payload = json.loads(out.read_text())
            sample = payload[sorted(payload.keys())[-1]]
            self.assertTrue(sample["meta"]["seeded"])
            self.assertEqual("official_monthly_baseline", sample["meta"]["seed_type"])
            self.assertEqual("statcan_cpi_csv", sample["meta"]["seed_source"])
            self.assertIn("official_cpi", sample)
            self.assertNotIn("headline", sample)

    def test_seed_history_skips_non_seeded_rows_without_force(self) -> None:
        module = self._load_module()
        seed_history = module["seed_history"]
        seed_history.__globals__["fetch_official_cpi_series"] = lambda: [
            {"ref_date": "2026-02", "index_value": 162.0, "mom_pct": 0.3, "yoy_pct": 2.4},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "historical.json"
            today = date.today().isoformat()
            existing = {
                today: {
                    "headline": {"nowcast_mom_pct": 0.9},
                    "meta": {"seeded": False},
                }
            }
            out.write_text(json.dumps(existing))
            written, skipped, _ = seed_history(days=1, output=out, force=False)
            self.assertEqual(0, written)
            self.assertEqual(1, skipped)
            payload = json.loads(out.read_text())
            self.assertEqual(0.9, payload[today]["headline"]["nowcast_mom_pct"])


if __name__ == "__main__":
    unittest.main()
