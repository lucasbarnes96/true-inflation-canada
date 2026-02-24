from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

import process as process_module
from process import (
    CATEGORY_WEIGHTS,
    build_gate_diagnostics,
    build_carry_forward_snapshot,
    classify_source_error,
    compute_freshness_composition,
    compute_nowcast_yoy_prorated,
    compute_category_contributions,
    compute_confidence,
    compute_coverage,
    compute_next_release,
    compute_signal_quality_score,
    dedupe_quotes,
    evaluate_gate,
    historical_row_from_snapshot,
    recompute_source_health,
    reconcile_historical_from_runs,
    validate_source_contract,
    write_outputs,
)
from scrapers.types import Quote, SourceHealth


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

    def test_validate_source_contract_optional_source_missing_without_recent_success_fails(self) -> None:
        check = validate_source_contract(
            contract_name="test_optional_source",
            source_name="test_optional_source",
            quotes=[],
            source_health={"source": "test_optional_source", "category": "food", "last_success_timestamp": None},
            historical={},
            now=datetime(2026, 2, 22, 12, 0, 0, tzinfo=timezone.utc),
            source_contracts={
                "test_optional_source": {
                    "min_records": 0,
                    "max_records": 100,
                    "allowed_value_range": [0.1, 500.0],
                    "max_stale_hours": 48.0,
                }
            },
        )
        self.assertFalse(check["passed"])

    def test_classify_source_error(self) -> None:
        self.assertEqual("tls", classify_source_error("SSL certificate verify failed"))
        self.assertEqual("dns", classify_source_error("nodename nor servname provided"))
        self.assertEqual("timeout", classify_source_error("request timed out"))
        self.assertEqual("blocked", classify_source_error("403 forbidden from endpoint"))
        self.assertEqual("parse", classify_source_error("Invalid JSON payload parse error"))

    def test_recompute_source_health_adds_reason_fields(self) -> None:
        raw = [
            SourceHealth(
                source="crtc_cmr_report",
                category="communication",
                tier=2,
                status="missing",
                last_success_timestamp=None,
                detail="Fetch failed: SSL certificate verify failed",
            )
        ]
        out = recompute_source_health(raw, datetime(2026, 2, 23, 15, 0, 0, tzinfo=timezone.utc))
        self.assertEqual(1, len(out))
        row = out[0]
        self.assertIn("error_class", row)
        self.assertIn("status_reason", row)
        self.assertIn("pending_reason", row)
        self.assertEqual("tls", row["error_class"])
        self.assertTrue(
            str(row["status_reason"]).startswith("source_missing")
            or str(row["status_reason"]).startswith("degraded_but_usable")
        )

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
            "meta": {
                "representativeness_ratio": 0.95,
                "qa_summary": {
                    "source_contract_pass_rate": 1.0,
                    "source_freshness_pass_rate": 1.0,
                    "imputed_weight_ratio": 0.0,
                    "cross_source_disagreement_score": 0.1,
                    "cross_source_disagreement_by_category": {"food": 0.1},
                    "source_inventory_ratio": 1.0,
                    "missing_sources": [],
                },
            },
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
            "meta": {
                "representativeness_ratio": 0.95,
                "qa_summary": {
                    "source_contract_pass_rate": 1.0,
                    "source_freshness_pass_rate": 1.0,
                    "imputed_weight_ratio": 0.0,
                    "cross_source_disagreement_score": 0.1,
                    "cross_source_disagreement_by_category": {"food": 0.1},
                    "source_inventory_ratio": 1.0,
                    "missing_sources": [],
                },
            },
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
            "meta": {
                "representativeness_ratio": 0.9,
                "qa_summary": {
                    "source_contract_pass_rate": 1.0,
                    "source_freshness_pass_rate": 1.0,
                    "imputed_weight_ratio": 0.0,
                    "cross_source_disagreement_score": 0.1,
                    "cross_source_disagreement_by_category": {"food": 0.1},
                    "source_inventory_ratio": 1.0,
                    "missing_sources": [],
                },
            },
        }
        diagnostics = build_gate_diagnostics(snapshot)
        self.assertTrue(diagnostics["representativeness"]["passed"])

    def test_evaluate_gate_fail_when_source_reliability_low(self) -> None:
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
            "meta": {
                "representativeness_ratio": 0.9,
                "qa_summary": {
                    "source_contract_pass_rate": 0.5,
                    "source_freshness_pass_rate": 1.0,
                    "imputed_weight_ratio": 0.0,
                    "cross_source_disagreement_score": 0.1,
                    "cross_source_disagreement_by_category": {"food": 0.1},
                    "source_inventory_ratio": 1.0,
                    "missing_sources": [],
                },
            },
        }
        blocked = evaluate_gate(snapshot)
        self.assertTrue(any("Gate G failed" in item for item in blocked))

    def test_evaluate_gate_fail_when_source_freshness_low(self) -> None:
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
            "meta": {
                "representativeness_ratio": 0.9,
                "qa_summary": {
                    "source_contract_pass_rate": 1.0,
                    "source_freshness_pass_rate": 0.4,
                    "imputed_weight_ratio": 0.0,
                    "cross_source_disagreement_score": 0.1,
                    "cross_source_disagreement_by_category": {"food": 0.1},
                    "source_inventory_ratio": 1.0,
                    "missing_sources": [],
                },
            },
        }
        blocked = evaluate_gate(snapshot)
        self.assertTrue(any("Gate J failed" in item for item in blocked))

    def test_compute_freshness_composition_bucket_boundaries(self) -> None:
        age_by_category = {
            "food": 0,
            "housing": 1,
            "transport": 2,
            "energy": 7,
            "communication": 8,
            "health_personal": 30,
            "recreation_education": 31,
        }
        category_signal_inputs = {
            category: [{"source": f"src_{category}", "effective_weight": 1.0}]
            for category in CATEGORY_WEIGHTS.keys()
        }
        source_health = [
            {"source": f"src_{category}", "age_days": age}
            for category, age in age_by_category.items()
        ]

        out = compute_freshness_composition(category_signal_inputs, source_health)
        total_weight = sum(CATEGORY_WEIGHTS.values())
        expected_0_1 = (CATEGORY_WEIGHTS["food"] + CATEGORY_WEIGHTS["housing"]) / total_weight
        expected_2_7 = (CATEGORY_WEIGHTS["transport"] + CATEGORY_WEIGHTS["energy"]) / total_weight
        expected_8_30 = (CATEGORY_WEIGHTS["communication"] + CATEGORY_WEIGHTS["health_personal"]) / total_weight
        expected_overflow = CATEGORY_WEIGHTS["recreation_education"] / total_weight

        self.assertAlmostEqual(expected_0_1, out["fresh_0_1d_weight_ratio"], places=4)
        self.assertAlmostEqual(expected_2_7, out["fresh_2_7d_weight_ratio"], places=4)
        self.assertAlmostEqual(expected_8_30, out["fresh_8_30d_weight_ratio"], places=4)
        self.assertAlmostEqual(expected_overflow, out["stale_gt_30d_or_missing_weight_ratio"], places=4)
        self.assertAlmostEqual(
            1.0,
            out["fresh_0_1d_weight_ratio"]
            + out["fresh_2_7d_weight_ratio"]
            + out["fresh_8_30d_weight_ratio"]
            + out["stale_gt_30d_or_missing_weight_ratio"],
            places=3,
        )

    def test_compute_freshness_composition_missing_age_goes_overflow(self) -> None:
        category_signal_inputs = {
            category: [{"source": f"src_{category}", "effective_weight": 1.0}]
            for category in CATEGORY_WEIGHTS.keys()
        }
        source_health = [{"source": f"src_{category}", "age_days": 0} for category in CATEGORY_WEIGHTS.keys()]
        source_health[0]["age_days"] = None
        out = compute_freshness_composition(category_signal_inputs, source_health)
        self.assertGreater(out["stale_gt_30d_or_missing_weight_ratio"], 0.0)

    def test_historical_row_from_snapshot_recomputes_missing_freshness(self) -> None:
        snapshot = {
            "as_of_date": "2026-02-24",
            "headline": {
                "nowcast_mom_pct": 0.1,
                "nowcast_yoy_pct": 1.9,
                "confidence": "medium",
                "coverage_ratio": 1.0,
                "signal_quality_score": 80,
                "lead_signal": "up",
            },
            "official_cpi": {"mom_pct": 0.0, "yoy_pct": 2.3, "latest_release_month": "2026-01"},
            "categories": {"food": {"proxy_level": 1.0, "daily_change_pct": 0.1, "status": "fresh"}},
            "meta": {
                "freshness_composition": {},
                "category_signal_inputs": {
                    category: [{"source": f"src_{category}", "effective_weight": 1.0}]
                    for category in CATEGORY_WEIGHTS.keys()
                },
            },
            "source_health": [
                {"source": f"src_{category}", "status": "fresh", "category": category, "tier": 1, "age_days": 0}
                for category in CATEGORY_WEIGHTS.keys()
            ],
            "release": {"status": "failed_gate", "carry_forward": False, "quality_tier": "blocked"},
        }
        out = historical_row_from_snapshot(snapshot)
        self.assertIn("fresh_0_1d_weight_ratio", out["meta"]["freshness_composition"])
        self.assertEqual(1.0, out["meta"]["freshness_composition"]["fresh_0_1d_weight_ratio"])

    def test_reconcile_historical_from_runs_includes_failed_gate_latest_created_at(self) -> None:
        run_early = {
            "as_of_date": "2026-02-24",
            "headline": {
                "nowcast_mom_pct": 0.0,
                "nowcast_yoy_pct": 1.2,
                "confidence": "medium",
                "coverage_ratio": 1.0,
                "signal_quality_score": 80,
                "lead_signal": "flat",
            },
            "official_cpi": {"mom_pct": 0.0, "yoy_pct": 2.3, "latest_release_month": "2026-01"},
            "categories": {"food": {"proxy_level": 1.0, "daily_change_pct": 0.0, "status": "fresh"}},
            "meta": {
                "freshness_composition": {},
                "category_signal_inputs": {category: [{"source": "src_food", "effective_weight": 1.0}] for category in CATEGORY_WEIGHTS},
            },
            "source_health": [{"source": "src_food", "status": "fresh", "category": "food", "tier": 1, "age_days": 0}],
            "release": {"status": "published", "created_at": "2026-02-24T12:00:00+00:00"},
        }
        run_late = {
            **run_early,
            "headline": {**run_early["headline"], "nowcast_yoy_pct": 1.9},
            "release": {"status": "failed_gate", "created_at": "2026-02-24T17:00:00+00:00"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            (runs_dir / "run_a.json").write_text(json.dumps(run_early))
            (runs_dir / "run_b.json").write_text(json.dumps(run_late))
            with patch.object(process_module, "RUNS_DIR", runs_dir):
                merged = reconcile_historical_from_runs({})
        self.assertIn("2026-02-24", merged)
        self.assertEqual(1.9, merged["2026-02-24"]["headline"]["nowcast_yoy_pct"])

    def test_write_outputs_updates_historical_for_failed_gate_runs(self) -> None:
        snapshot = {
            "as_of_date": "2026-02-24",
            "timestamp": "2026-02-24T17:00:00+00:00",
            "headline": {
                "nowcast_mom_pct": 0.1,
                "nowcast_yoy_pct": 1.9,
                "confidence": "low",
                "coverage_ratio": 1.0,
                "signal_quality_score": 30,
                "lead_signal": "up",
            },
            "categories": {"food": {"proxy_level": 1.0, "daily_change_pct": 0.1, "status": "fresh"}},
            "official_cpi": {"mom_pct": 0.0, "yoy_pct": 2.3, "latest_release_month": "2026-01"},
            "source_health": [{"source": "src_food", "status": "fresh", "category": "food", "tier": 1, "age_days": 0}],
            "meta": {"release_events": {}, "consensus": {}},
            "release": {
                "run_id": "run_test",
                "status": "failed_gate",
                "blocked_conditions": ["Gate X failed"],
                "created_at": "2026-02-24T17:00:00+00:00",
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            runs_dir = data_dir / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            latest_path = data_dir / "latest.json"
            historical_path = data_dir / "historical.json"
            published_path = data_dir / "published_latest.json"
            source_contracts_path = data_dir / "source_contracts.json"
            release_events_path = data_dir / "release_events.json"
            consensus_latest_path = data_dir / "consensus_latest.json"
            methodology_path = data_dir / "methodology.json"
            source_catalog_path = data_dir / "source_catalog.json"
            performance_path = data_dir / "performance_summary.json"
            model_card_path = data_dir / "model_card.json"
            with (
                patch.object(process_module, "RUNS_DIR", runs_dir),
                patch.object(process_module, "LATEST_PATH", latest_path),
                patch.object(process_module, "HISTORICAL_PATH", historical_path),
                patch.object(process_module, "PUBLISHED_LATEST_PATH", published_path),
                patch.object(process_module, "SOURCE_CONTRACTS_PATH", source_contracts_path),
                patch.object(process_module, "RELEASE_EVENTS_PATH", release_events_path),
                patch.object(process_module, "CONSENSUS_LATEST_PATH", consensus_latest_path),
                patch.object(process_module, "METHODOLOGY_PATH", methodology_path),
                patch.object(process_module, "SOURCE_CATALOG_PATH", source_catalog_path),
                patch.object(process_module, "PERFORMANCE_SUMMARY_PATH", performance_path),
                patch.object(process_module, "MODEL_CARD_PATH", model_card_path),
                patch.object(process_module, "reconcile_historical_from_runs", return_value={}),
                patch.object(process_module, "update_historical", return_value={"2026-02-24": {"headline": {"nowcast_yoy_pct": 1.9}}}) as mock_update,
                patch.object(process_module, "write_performance_summary", return_value={}),
                patch.object(process_module, "build_methodology_payload", return_value={}),
                patch.object(process_module, "record_qa_checks"),
                patch.object(process_module, "record_release_run"),
            ):
                write_outputs(snapshot)
        mock_update.assert_called_once()

    def test_build_carry_forward_snapshot_marks_degraded_published(self) -> None:
        sample = {
            "as_of_date": "2026-02-23",
            "timestamp": "2026-02-23T00:00:00+00:00",
            "headline": {
                "nowcast_mom_pct": 0.0,
                "nowcast_yoy_pct": 1.382,
                "confidence": "low",
                "coverage_ratio": 0.8,
                "signal_quality_score": 25,
                "lead_signal": "flat",
                "method_label": "test",
            },
            "categories": {},
            "official_cpi": {},
            "bank_of_canada": {},
            "source_health": [{"source": "src_food", "category": "food", "tier": 1, "status": "fresh", "detail": "", "age_days": 0}],
            "notes": [],
            "meta": {"category_signal_inputs": {"food": [{"source": "src_food", "effective_weight": 1.0}]}},
            "release": {"status": "published", "run_id": "run_x", "qa_status": "passed", "lifecycle_states": [], "blocked_conditions": [], "created_at": "2026-02-23T00:00:00+00:00"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            published_path = Path(tmp) / "published_latest.json"
            latest_path = Path(tmp) / "latest.json"
            published_path.write_text(json.dumps(sample))
            latest_path.write_text(json.dumps(sample))
            with patch.object(process_module, "PUBLISHED_LATEST_PATH", published_path), patch.object(process_module, "LATEST_PATH", latest_path):
                out = build_carry_forward_snapshot(
                    now=datetime(2026, 2, 24, 12, 0, 0, tzinfo=timezone.utc),
                    reason="pipeline_exception:RuntimeError",
                )
        assert out is not None
        self.assertEqual("published", out["release"]["status"])
        self.assertEqual("degraded", out["release"]["quality_tier"])
        self.assertTrue(out["release"]["carry_forward"])
        self.assertIn("fresh_0_1d_weight_ratio", out["meta"]["freshness_composition"])


if __name__ == "__main__":
    unittest.main()
