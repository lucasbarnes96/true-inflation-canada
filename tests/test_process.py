from __future__ import annotations

import unittest
from datetime import date, datetime, timezone

from process import (
    CATEGORY_WEIGHTS,
    build_gate_diagnostics,
    compute_nowcast_yoy_prorated,
    compute_category_contributions,
    compute_confidence,
    compute_coverage,
    compute_next_release,
    compute_signal_quality_score,
    dedupe_quotes,
    evaluate_gate,
)
from scrapers.types import Quote


class ProcessTests(unittest.TestCase):
    def test_dedupe_quotes(self) -> None:
        q1 = Quote("food", "milk", 4.0, date(2026, 2, 15), "src")
        q2 = Quote("food", "milk", 5.0, date(2026, 2, 15), "src")
        deduped = dedupe_quotes([q1, q2])
        self.assertEqual(1, len(deduped))
        self.assertEqual(5.0, deduped[0].value)

    def test_compute_coverage(self) -> None:
        categories = {
            "food": {"status": "fresh", "proxy_level": 1.0, "weight": CATEGORY_WEIGHTS["food"]},
            "housing": {"status": "missing", "proxy_level": None, "weight": CATEGORY_WEIGHTS["housing"]},
            "transport": {"status": "fresh", "proxy_level": 1.0, "weight": CATEGORY_WEIGHTS["transport"]},
            "energy": {"status": "stale", "proxy_level": 1.0, "weight": CATEGORY_WEIGHTS["energy"]},
        }
        coverage = compute_coverage(categories)
        self.assertGreater(coverage, 0)
        self.assertLess(coverage, 1)

    def test_compute_confidence(self) -> None:
        self.assertEqual("high", compute_confidence(0.95, 0, []))
        self.assertEqual("medium", compute_confidence(0.7, 0, []))
        self.assertEqual("low", compute_confidence(0.4, 0, []))
        self.assertEqual("medium", compute_confidence(0.95, 3, []))
        self.assertEqual("low", compute_confidence(0.7, 3, []))
        self.assertEqual("low", compute_confidence(0.95, 0, ["gate failed"]))

    def test_compute_signal_quality_score(self) -> None:
        categories = {
            "food": {"status": "fresh", "proxy_level": 1.0},
            "housing": {"status": "fresh", "proxy_level": 1.0},
        }
        diversity = {"food": 2, "housing": 2}
        score = compute_signal_quality_score(0.9, 0, [], diversity, categories)
        self.assertGreaterEqual(score, 85)

        penalized = compute_signal_quality_score(
            0.9,
            4,
            ["gate failed"],
            {"food": 1, "housing": 1},
            categories,
        )
        self.assertLess(penalized, score)

    def test_compute_category_contributions(self) -> None:
        categories = {
            "food": {"daily_change_pct": 1.0, "weight": 0.2},
            "housing": {"daily_change_pct": -0.5, "weight": 0.3},
            "energy": {"daily_change_pct": None, "weight": 0.1},
        }
        out = compute_category_contributions(categories)
        self.assertEqual(0.2, out["food"])
        self.assertEqual(-0.15, out["housing"])
        self.assertIsNone(out["energy"])

    def test_compute_next_release(self) -> None:
        payload = {
            "events": [
                {"release_at_utc": "2026-02-15T13:30:00+00:00"},
                {"release_at_utc": "2026-02-17T13:30:00+00:00", "event_date": "2026-02-17"},
            ]
        }
        out = compute_next_release(
            payload,
            datetime(2026, 2, 16, 12, 0, 0, tzinfo=timezone.utc),
        )
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual("2026-02-17", out["event_date"])

    def test_compute_nowcast_yoy_prorated_mid_month(self) -> None:
        series = [
            {"ref_date": "2025-01", "index_value": 150.0},
            {"ref_date": "2025-02", "index_value": 152.0},
            {"ref_date": "2026-01", "index_value": 160.0},
        ]
        out, meta = compute_nowcast_yoy_prorated(date(2026, 2, 16), 1.2, series)
        self.assertIsNotNone(out)
        self.assertAlmostEqual(16 / 28, meta["prorate_factor"], places=4)

    def test_compute_nowcast_yoy_prorated_day_one(self) -> None:
        series = [
            {"ref_date": "2025-02", "index_value": 152.0},
            {"ref_date": "2026-01", "index_value": 160.0},
        ]
        out, meta = compute_nowcast_yoy_prorated(date(2026, 2, 1), 1.0, series)
        self.assertIsNotNone(out)
        self.assertAlmostEqual(1 / 28, meta["prorate_factor"], places=4)

    def test_compute_nowcast_yoy_prorated_month_end(self) -> None:
        series = [
            {"ref_date": "2025-02", "index_value": 152.0},
            {"ref_date": "2026-01", "index_value": 160.0},
        ]
        out, meta = compute_nowcast_yoy_prorated(date(2026, 2, 28), 1.0, series)
        self.assertIsNotNone(out)
        self.assertAlmostEqual(1.0, meta["prorate_factor"], places=4)

    def test_compute_nowcast_yoy_prorated_missing_inputs(self) -> None:
        out, meta = compute_nowcast_yoy_prorated(date(2026, 2, 16), None, [])
        self.assertIsNone(out)
        self.assertEqual("missing_nowcast_mom", meta["reason"])

    def test_evaluate_gate_pass(self) -> None:
        snapshot = {
            "source_health": [
                {"source": "apify_loblaws", "category": "food", "status": "fresh", "age_days": 1},
                {"source": "statcan_food_prices", "category": "food", "status": "fresh", "age_days": 3},
                {"source": "statcan_cpi_csv", "status": "fresh", "age_days": 2},
                {"source": "statcan_gas_csv", "status": "fresh", "age_days": 2},
                {"source": "oeb_scrape", "status": "fresh", "age_days": 0},
            ],
            "categories": {
                "food": {"points": 10},
                "housing": {"points": 3},
                "transport": {"points": 1},
                "energy": {"points": 1},
            },
            "official_cpi": {"latest_release_month": "2025-12"},
            "meta": {"representativeness_ratio": 0.95},
        }
        self.assertEqual([], evaluate_gate(snapshot))

    def test_evaluate_gate_fail_when_apify_stale(self) -> None:
        snapshot = {
            "source_health": [
                {"source": "apify_loblaws", "category": "food", "status": "stale", "age_days": 20},
                {"source": "statcan_cpi_csv", "status": "fresh", "age_days": 2},
                {"source": "statcan_gas_csv", "status": "fresh", "age_days": 2},
                {"source": "oeb_scrape", "status": "fresh", "age_days": 0},
            ],
            "categories": {
                "food": {"points": 10},
                "housing": {"points": 3},
                "transport": {"points": 1},
                "energy": {"points": 1},
            },
            "official_cpi": {"latest_release_month": "2025-12"},
            "meta": {"representativeness_ratio": 0.95},
        }
        blocked = evaluate_gate(snapshot)
        self.assertTrue(any("Gate A failed" in item for item in blocked))

    def test_gate_diagnostics_include_representativeness(self) -> None:
        snapshot = {
            "source_health": [
                {"source": "apify_loblaws", "category": "food", "status": "fresh", "age_days": 1},
                {"source": "statcan_food_prices", "category": "food", "status": "fresh", "age_days": 2},
                {"source": "statcan_cpi_csv", "status": "fresh", "age_days": 2},
                {"source": "statcan_gas_csv", "status": "fresh", "age_days": 2},
                {"source": "oeb_scrape", "status": "fresh", "age_days": 0},
            ],
            "categories": {
                "food": {"points": 10},
                "housing": {"points": 3},
                "transport": {"points": 1},
                "energy": {"points": 1},
                "communication": {"points": 1},
                "health_personal": {"points": 1},
                "recreation_education": {"points": 1},
            },
            "official_cpi": {"latest_release_month": "2025-12"},
            "meta": {"representativeness_ratio": 0.9},
        }
        diagnostics = build_gate_diagnostics(snapshot)
        self.assertTrue(diagnostics["representativeness"]["passed"])


if __name__ == "__main__":
    unittest.main()
