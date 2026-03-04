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
        self.latest = self.data_dir / "latest.json"
        self.published_latest = self.data_dir / "published_latest.json"
        self.historical = self.data_dir / "historical.json"
        self.releases_db = self.data_dir / "releases.db"
        self.qa_db = self.data_dir / "qa_runs.db"
        self.methodology_json = self.data_dir / "methodology.json"
        self.source_catalog_json = self.data_dir / "source_catalog.json"
        self.performance_summary = self.data_dir / "performance_summary.json"
        self.release_events = self.data_dir / "release_events.json"
        self.consensus_latest = self.data_dir / "consensus_latest.json"
        self._backup_paths = [
            self.latest,
            self.published_latest,
            self.historical,
            self.releases_db,
            self.qa_db,
            self.performance_summary,
            self.release_events,
            self.consensus_latest,
            self.methodology_json,
            self.source_catalog_json,
        ]
        self._backups: dict[Path, bytes | None] = {}
        for path in self._backup_paths:
            self._backups[path] = path.read_bytes() if path.exists() else None

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
                    "this_run_source_contract_pass_rate": 1.0,
                    "trailing_30d_source_contract_pass_rate": 0.98,
                    "fresh_weight_ratio": 0.9,
                    "cross_source_disagreement_score": 0.12,
                    "cross_source_disagreement_by_category": {"food": 0.12},
                    "quarantine_sources": [],
                    "imputation_used": False,
                    "imputed_categories": [],
                    "imputed_weight_ratio": 0.0,
                    "failure_fingerprint": {
                        "top_failed_check": None,
                        "failed_source_checks": 0,
                    },
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
                "execution_outcome": "success",
                "publication_outcome": "published",
                "qa_window_close_at": "2026-02-16T00:00:00+00:00",
                "lifecycle_states": ["started", "completed", "published"],
                "blocked_conditions": [],
                "created_at": "2026-02-15T00:00:00+00:00",
                "published_at": "2026-02-15T00:00:00+00:00",
            },
        }
        self.latest.write_text(json.dumps(snapshot))
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
        self.performance_summary.write_text(
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
        self.release_events.write_text(
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
        self.consensus_latest.write_text(
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
        self.methodology_json.write_text(
            json.dumps(
                {
                    "method_version": "v1.5.0-test-static",
                    "gate_policy": {"representativeness_min_fresh_ratio": 0.8},
                    "weights_reference": {"tracked_share_total": 0.9},
                }
            )
        )
        self.source_catalog_json.write_text(json.dumps({"items": [{"source": "static_source"}]}))

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

    def tearDown(self) -> None:
        for path, content in self._backups.items():
            if content is None:
                if path.exists():
                    path.unlink()
                continue
            path.write_bytes(content)

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
        self.assertEqual("v1.5.0-test-static", body["method_version"])
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
        self.assertEqual("static_source", body["items"][0]["source"])

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
        self.assertEqual("success", body["execution_outcome"])
        self.assertEqual("published", body["publication_outcome"])
        self.assertIn("this_run_source_contract_pass_rate", body)
        self.assertIn("trailing_30d_source_contract_pass_rate", body)
        self.assertIn("failure_fingerprint", body)
        self.assertIn("qa_summary", body)

    def test_ops_run_health_endpoint(self) -> None:
        resp = self.client.get("/v1/ops/run-health")
        self.assertEqual(200, resp.status_code)
        body = resp.json()
        self.assertEqual("run_123", body["run_id"])
        self.assertEqual("success", body["execution_outcome"])
        self.assertEqual("published", body["publication_outcome"])

    def test_qa_status_legacy_fallbacks(self) -> None:
        payload = json.loads(self.latest.read_text())
        payload["release"].pop("execution_outcome", None)
        payload["release"].pop("publication_outcome", None)
        payload["meta"]["qa_failure_fingerprint"] = {"top_failed_check": "median_jump", "failed_check_events": 3}
        payload["meta"]["qa_summary"].pop("failure_fingerprint", None)
        payload["meta"]["qa_summary"].pop("this_run_source_contract_pass_rate", None)
        payload["meta"]["qa_summary"].pop("trailing_30d_source_contract_pass_rate", None)
        payload["meta"]["qa_summary"]["source_contract_pass_rate"] = 0.88
        self.latest.write_text(json.dumps(payload))

        resp = self.client.get("/v1/qa/status")
        self.assertEqual(200, resp.status_code)
        body = resp.json()
        self.assertEqual("success", body["execution_outcome"])
        self.assertEqual("published", body["publication_outcome"])
        self.assertEqual(0.88, body["this_run_source_contract_pass_rate"])
        self.assertEqual("median_jump", body["top_failed_check"])

    def test_history_preserves_seeded_meta(self) -> None:
        resp = self.client.get("/v1/nowcast/history")
        self.assertEqual(200, resp.status_code)
        items = resp.json()["items"]
        self.assertEqual(1, len(items))
        self.assertTrue(items[0]["meta"]["seeded"])
        self.assertEqual(2.4, items[0]["headline"]["nowcast_yoy_pct"])

    def test_history_includes_failed_gate_daily_rows(self) -> None:
        self.historical.write_text(
            json.dumps(
                {
                    "2026-02-24": {
                        "headline": {"nowcast_mom_pct": 0.1, "nowcast_yoy_pct": 1.9, "lead_signal": "up"},
                        "official_cpi": {"mom_pct": 0.0, "yoy_pct": 2.3, "latest_release_month": "2026-01"},
                        "meta": {"seeded": False, "freshness_composition": {"fresh_0_1d_weight_ratio": 0.855}},
                        "release": {"status": "failed_gate", "blocked_conditions": ["Gate X failed"]},
                    }
                }
            )
        )
        resp = self.client.get("/v1/nowcast/history")
        self.assertEqual(200, resp.status_code)
        items = resp.json()["items"]
        self.assertEqual(1, len(items))
        self.assertEqual("failed_gate", items[0]["release"]["status"])
        self.assertEqual(1.9, items[0]["headline"]["nowcast_yoy_pct"])

    def test_data_asset_route_blocks_path_traversal(self) -> None:
        resp = self.client.get("/data/../../api/main.py")
        self.assertIn(resp.status_code, {403, 404})

    def test_about_routes(self) -> None:
        resp = self.client.get("/about")
        self.assertEqual(200, resp.status_code)
        self.assertIn("text/html", resp.headers.get("content-type", ""))

        resp_html = self.client.get("/about.html")
        self.assertEqual(200, resp_html.status_code)
        self.assertIn("text/html", resp_html.headers.get("content-type", ""))

    def test_changelog_routes(self) -> None:
        resp = self.client.get("/changelog")
        self.assertEqual(200, resp.status_code)
        self.assertIn("text/html", resp.headers.get("content-type", ""))

        resp_html = self.client.get("/changelog.html")
        self.assertEqual(200, resp_html.status_code)
        self.assertIn("text/html", resp_html.headers.get("content-type", ""))


if __name__ == "__main__":
    unittest.main()
