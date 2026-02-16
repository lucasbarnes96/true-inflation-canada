from __future__ import annotations

import unittest

from performance import compute_performance_summary


class PerformanceTests(unittest.TestCase):
    def test_compute_performance_summary_empty(self) -> None:
        out = compute_performance_summary({})
        self.assertEqual("v1.2.0", out["method_version"])
        self.assertEqual(0, out["evaluated_points"])
        self.assertIsNone(out["mae_mom_pct"])

    def test_compute_performance_summary_happy_path(self) -> None:
        historical = {
            "2026-01-01": {
                "headline": {"nowcast_mom_pct": 0.2, "divergence_mom_pct": 0.1, "lead_signal": "up"},
                "official_cpi": {"mom_pct": 0.1},
            },
            "2026-01-02": {
                "headline": {"nowcast_mom_pct": -0.1, "divergence_mom_pct": -0.05, "lead_signal": "down"},
                "official_cpi": {"mom_pct": -0.05},
            },
        }
        out = compute_performance_summary(historical, window_days=10)
        self.assertEqual(2, out["evaluated_points"])
        self.assertIsNotNone(out["mae_mom_pct"])
        self.assertIsNotNone(out["directional_accuracy_pct"])
        self.assertIsNotNone(out["lead_time_score_pct"])


if __name__ == "__main__":
    unittest.main()
