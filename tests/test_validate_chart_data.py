import unittest
from copy import deepcopy

from scripts.validate_chart_data import validate_payload


def build_payload():
    return {
        "metadata": {
            "description": "test payload",
            "updated_at": "2026-04-27T05:28:44+00:00",
        },
        "adjusters": {
            "M1+": {"dates": ["2026-01", "2026-02"], "values": [1.0, 2.0], "normalized": [1.0, 2.0]},
            "M1++": {"dates": ["2026-01", "2026-02"], "values": [1.0, 2.0], "normalized": [1.0, 2.0]},
            "M3": {"dates": ["2026-01", "2026-02"], "values": [1.0, 2.0], "normalized": [1.0, 2.0]},
            "CPI": {"dates": ["2026-01", "2026-02"], "values": [1.0, 2.0], "normalized": [1.0, 2.0]},
        },
        "assets": {
            "TSX": {"dates": ["2026-02", "2026-03"], "values": [1.0, 2.0]},
            "Canadian REITs": {"dates": ["2026-02", "2026-03"], "values": [1.0, 2.0]},
            "Bitcoin (CAD)": {"dates": ["2026-02", "2026-03"], "values": [1.0, 2.0]},
            "Ethereum (CAD)": {"dates": ["2026-02", "2026-03"], "values": [1.0, 2.0]},
            "Crude Oil": {"dates": ["2026-02", "2026-03"], "values": [1.0, 2.0]},
            "S&P 500 (CAD)": {"dates": ["2026-02", "2026-03"], "values": [1.0, 2.0]},
            "NASDAQ (CAD)": {"dates": ["2026-02", "2026-03"], "values": [1.0, 2.0]},
            "Dow Jones (CAD)": {"dates": ["2026-02", "2026-03"], "values": [1.0, 2.0]},
            "Gold (CAD)": {"dates": ["2026-02", "2026-03"], "values": [1.0, 2.0]},
            "Silver (CAD)": {"dates": ["2026-02", "2026-03"], "values": [1.0, 2.0]},
            "Canadian House Prices (NHPI)": {"dates": ["2026-01", "2026-02"], "values": [1.0, 2.0]},
            "Labour Productivity": {"dates": ["2025-07", "2025-10"], "values": [1.0, 2.0]},
        },
    }


class ValidateChartDataTests(unittest.TestCase):
    def test_valid_payload_passes(self):
        summary = validate_payload(build_payload())
        self.assertTrue(summary["publish_allowed"])
        self.assertEqual(summary["required_adjusters_present"], 4)
        self.assertEqual(summary["required_assets_present"], 12)

    def test_missing_required_adjuster_fails(self):
        payload = build_payload()
        payload["adjusters"].pop("M3")
        with self.assertRaisesRegex(ValueError, "Missing required adjusters"):
            validate_payload(payload)

    def test_missing_required_asset_fails(self):
        payload = build_payload()
        payload["assets"].pop("TSX")
        with self.assertRaisesRegex(ValueError, "Missing required assets"):
            validate_payload(payload)

    def test_mismatched_lengths_fail(self):
        payload = build_payload()
        payload["assets"]["TSX"] = {"dates": ["2026-02"], "values": [1.0, 2.0]}
        with self.assertRaisesRegex(ValueError, "mismatched dates/values lengths"):
            validate_payload(payload)

    def test_invalid_numeric_value_fails(self):
        payload = build_payload()
        payload["adjusters"]["CPI"]["values"][1] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite number"):
            validate_payload(payload)

    def test_malformed_timestamp_fails(self):
        payload = build_payload()
        payload["metadata"]["updated_at"] = "not-a-timestamp"
        with self.assertRaises(ValueError):
            validate_payload(payload)

    def test_stale_quarterly_data_fails(self):
        payload = build_payload()
        payload["assets"]["Labour Productivity"]["dates"] = ["2024-01", "2024-02"]
        with self.assertRaisesRegex(ValueError, "statcan_quarterly_assets is stale"):
            validate_payload(payload)


if __name__ == "__main__":
    unittest.main()
