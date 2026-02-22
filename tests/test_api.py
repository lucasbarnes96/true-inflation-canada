from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
    from api.main import app

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    TestClient = None  # type: ignore[assignment]
    app = None  # type: ignore[assignment]


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed")
class ApiContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data_dir = Path("data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.published_latest = self.data_dir / "published_latest.json"
        self.historical = self.data_dir / "historical.json"
        self.releases_db = self.data_dir / "releases.db"
        self.qa_db = self.data_dir / "qa_runs.db"

        snapshot = {
            "as_of_date": "2026-02-15",
            "timestamp": "2026-02-15T00:00:00+00:00",
            "headline": {
                "nowcast_mom_pct": 0.1,
                "nowcast_yoy_pct": 2.4,
                "confidence": "medium",
                "coverage_ratio": 0.75,
                "signal_quality_score": 74,
                "lead_signal": "up",
                "next_release_at_utc": "2026-02-17T13:30:00+00:00",
                "consensus_yoy": 2.6,
                "consensus_spread_yoy": -0.1,
                "deviation_yoy_pct": -0.2,
                "method_label": "test",
            },
            "categories": {
                "food": {"proxy_level": 1.0, "daily_change_pct": 0.1, "weight": 0.165, "points": 8, "status": "fresh"},
                "housing": {"proxy_level": 1.0, "daily_change_pct": 0.1, "weight": 0.3, "points": 3, "status": "fresh"},
                "transport": {"proxy_level": 1.0, "daily_change_pct": 0.1, "weight": 0.15, "points": 2, "status": "fresh"},
                "energy": {"proxy_level": 1.0, "daily_change_pct": 0.1, "weight": 0.08, "points": 2, "status": "fresh"},
            },
            "official_cpi": {"latest_release_month": "2025-12", "mom_pct": 0.2, "yoy_pct": 2.5},
            "bank_of_canada": {},
            "source_health": [
                {
                    "source": "apify_loblaws",
                    "category": "food",
                    "tier": 1,
                    "status": "fresh",
                    "last_success_timestamp": "2026-02-14T00:00:00+00:00",
                    "detail": "ok",
                    "source_run_id": "abc",
                    "age_days": 1,
                    "updated_days_ago": "updated 1 day ago",
                }
            ],
            "notes": [],
            "meta": {
                "method_version": "v1.5.0",
                "weights": {"tracked_share_total": 0.9},
                "forecast": {
                    "status": "published",
                    "point_yoy": 2.7,
                    "lower_yoy": 2.3,
                    "upper_yoy": 3.1,
                    "confidence": "medium",
                    "next_release_date": "2026-02-17",
                },
                "calibration": {
                    "maturity_tier": "bootstrapping",
                    "live_days": 5,
                    "minimum_days_for_stable_eval": 30,
                    "forecast_eligibility": {"eligible": False, "reason": "forecast_status=insufficient_history"},
                    "current_error_metrics": {"mae_yoy_pct": 0.4},
                },
                "qa_summary": {
                    "source_contract_pass_rate": 0.98,
                    "fresh_weight_ratio": 0.9,
                    "cross_source_disagreement_score": 0.12,
                    "cross_source_disagreement_by_category": {"food": 0.12},
                    "quarantine_sources": [],
                    "imputation_used": False,
                    "imputed_categories": [],
                    "imputed_weight_ratio": 0.0,
                },
            },
            "performance_ref": {
                "summary_path": "data/performance_summary.json",
                "model_card_path": "data/model_card_latest.json",
            },
            "release": {
                "run_id": "run_123",
                "status": "published",
                "qa_status": "passed",
                "qa_window_close_at": "2026-02-16T00:00:00+00:00",
                "lifecycle_states": ["started", "completed", "published"],
                "blocked_conditions": [],
                "created_at": "2026-02-15T00:00:00+00:00",
                "published_at": "2026-02-15T00:00:00+00:00",
            },
        }
        self.published_latest.write_text(json.dumps(snapshot))
        self.historical.write_text(
            json.dumps(
                {
                    "2026-02-15": {
                        "headline": {"nowcast_mom_pct": 0.1, "nowcast_yoy_pct": 2.4, "lead_signal": "up"},
                        "official_cpi": {"mom_pct": 0.2, "yoy_pct": 2.5, "latest_release_month": "2025-12"},
                        "category_contributions": {"food": 0.02},
                        "meta": {"seeded": True, "seed_type": "official_monthly_baseline", "seed_source": "statcan_cpi_csv"},
                    }
                }
            )
        )
        (self.data_dir / "performance_summary.json").write_text(
            json.dumps(
                {
                    "method_version": "v1.5.0",
                    "window_days": 120,
                    "evaluated_points": 10,
                    "evaluated_live_points": 5,
                    "mae_yoy_pct": 0.22,
                    "median_abs_error_yoy_pct": 0.19,
                    "directional_accuracy_yoy_pct": 60.0,
                    "bias_yoy_pct": -0.04,
                    "mae_mom_pct": 0.11,
                    "directional_accuracy_pct": 70.0,
                    "lead_time_score_pct": 70.0,
                }
            )
        )
        (self.data_dir / "release_events.json").write_text(
            json.dumps(
                {
                    "events": [
                        {
                            "event_date": "2026-02-17",
                            "release_at_et": "2026-02-17 08:30 ET",
                            "release_at_utc": "2026-02-17T13:30:00+00:00",
                            "series": "Canada CPI",
                            "status": "scheduled",
                        }
                    ],
                    "next_release": {
                        "event_date": "2026-02-17",
                        "release_at_et": "2026-02-17 08:30 ET",
                        "release_at_utc": "2026-02-17T13:30:00+00:00",
                        "series": "Canada CPI",
                        "status": "upcoming",
                        "countdown_seconds": 3600,
                    },
                }
            )
        )
        (self.data_dir / "consensus_latest.json").write_text(
            json.dumps(
                {
                    "as_of": "2026-02-16T12:00:00+00:00",
                    "headline_yoy": 2.6,
                    "headline_mom": 0.3,
                    "source_count": 2,
                    "confidence": "medium",
                }
            )
        )

        with sqlite3.connect(self.releases_db) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS release_runs (run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL, blocked_conditions TEXT NOT NULL, snapshot_path TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT OR REPLACE INTO release_runs (run_id, created_at, status, blocked_conditions, snapshot_path) VALUES (?, ?, ?, ?, ?)",
                ("run_123", "2026-02-15T00:00:00+00:00", "published", "[]", "data/runs/run_123.json"),
            )
            conn.commit()
        with sqlite3.connect(self.qa_db) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS daily_source_reliability (as_of_date TEXT NOT NULL, source TEXT NOT NULL, pass_rate_30d REAL NOT NULL, freshness_pass_rate_30d REAL NOT NULL, runs_30d INTEGER NOT NULL, PRIMARY KEY (as_of_date, source))"
            )
            conn.execute(
                "INSERT OR REPLACE INTO daily_source_reliability (as_of_date, source, pass_rate_30d, freshness_pass_rate_30d, runs_30d) VALUES (?, ?, ?, ?, ?)",
                ("2026-02-15", "apify_loblaws", 0.97, 0.92, 12),
            )
            conn.commit()

        self.client = TestClient(app)

    def test_latest_endpoint(self) -> None:
        resp = self.client.get("/v1/nowcast/latest")
        self.assertEqual(200, resp.status_code)
        body = resp.json()
        self.assertEqual("run_123", body["release"]["run_id"])
        self.assertIn("signal_quality_score", body["headline"])
        self.assertIn("lead_signal", body["headline"])
        self.assertIn("nowcast_yoy_pct", body["headline"])
        self.assertEqual("v1.5.0", body["meta"]["method_version"])

    def test_methodology_endpoint(self) -> None:
        resp = self.client.get("/v1/methodology")
        self.assertEqual(200, resp.status_code)
        body = resp.json()
        self.assertIn("gate_policy", body)
        self.assertIn("weights_reference", body)

    def test_performance_summary_endpoint(self) -> None:
        resp = self.client.get("/v1/performance/summary")
        self.assertEqual(200, resp.status_code)
        body = resp.json()
        self.assertEqual("v1.5.0", body["method_version"])

    def test_sources_catalog_endpoint(self) -> None:
        resp = self.client.get("/v1/sources/catalog")
        self.assertEqual(200, resp.status_code)
        body = resp.json()
        self.assertIn("items", body)
        self.assertGreater(len(body["items"]), 0)

    def test_releases_upcoming_endpoint(self) -> None:
        resp = self.client.get("/v1/releases/upcoming")
        self.assertEqual(200, resp.status_code)
        body = resp.json()
        self.assertIn("next_release", body)
        self.assertEqual("2026-02-17", body["next_release"]["event_date"])

    def test_consensus_latest_endpoint(self) -> None:
        resp = self.client.get("/v1/consensus/latest")
        self.assertEqual(200, resp.status_code)
        body = resp.json()
        self.assertEqual(2.6, body["headline_yoy"])

    def test_forecast_endpoint(self) -> None:
        resp = self.client.get("/v1/forecast/next_release")
        self.assertEqual(200, resp.status_code)
        body = resp.json()
        self.assertEqual("published", body["status"])
        self.assertIn("point_yoy", body)

    def test_calibration_endpoint(self) -> None:
        resp = self.client.get("/v1/calibration/status")
        self.assertEqual(200, resp.status_code)
        body = resp.json()
        self.assertIn("maturity_tier", body)

    def test_qa_status_endpoint(self) -> None:
        resp = self.client.get("/v1/qa/status")
        self.assertEqual(200, resp.status_code)
        body = resp.json()
        self.assertEqual("passed", body["qa_status"])
        self.assertIn("qa_summary", body)

    def test_history_preserves_seeded_meta(self) -> None:
        resp = self.client.get("/v1/nowcast/history")
        self.assertEqual(200, resp.status_code)
        items = resp.json()["items"]
        self.assertEqual(1, len(items))
        self.assertTrue(items[0]["meta"]["seeded"])
        self.assertEqual(2.4, items[0]["headline"]["nowcast_yoy_pct"])


if __name__ == "__main__":
    unittest.main()
