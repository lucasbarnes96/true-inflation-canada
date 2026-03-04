from __future__ import annotations

import unittest

from pipeline.run_state import infer_execution_outcome, infer_publication_outcome, normalize_snapshot_run_state


class RunStateTests(unittest.TestCase):
    def test_outcome_inference_for_failed_gate(self) -> None:
        self.assertEqual("success", infer_execution_outcome("failed_gate", None))
        self.assertEqual("failed_gate", infer_publication_outcome("failed_gate", None))

    def test_outcome_inference_for_crashed(self) -> None:
        self.assertEqual("crash", infer_execution_outcome("crashed", None))
        self.assertEqual("failed_gate", infer_publication_outcome("crashed", None))

    def test_normalize_snapshot_run_state_uses_legacy_and_reliability(self) -> None:
        payload = {
            "as_of_date": "2026-02-24",
            "meta": {
                "qa_summary": {
                    "source_contract_pass_rate": 0.72,
                    "source_freshness_pass_rate": 0.98,
                    "source_checks": [
                        {"source": "s1", "category": "food", "check_name": "median_jump", "passed": False},
                    ],
                }
            },
            "release": {"status": "failed_gate"},
        }

        out = normalize_snapshot_run_state(
            payload,
            reliability_rows=[(0.66, 0.99), (0.68, 0.97)],
            build_qa_failure_fingerprint_fn=lambda checks: {"top_failed_check": "median_jump", "failed_check_events": 1},
        )

        self.assertEqual("success", out["release"]["execution_outcome"])
        self.assertEqual("failed_gate", out["release"]["publication_outcome"])
        self.assertEqual(0.72, out["meta"]["qa_summary"]["this_run_source_contract_pass_rate"])
        self.assertEqual(0.98, out["meta"]["qa_summary"]["this_run_source_freshness_pass_rate"])
        self.assertEqual(0.67, out["meta"]["qa_summary"]["trailing_30d_source_contract_pass_rate"])
        self.assertEqual(0.98, out["meta"]["qa_summary"]["trailing_30d_source_freshness_pass_rate"])
        self.assertEqual("median_jump", out["meta"]["qa_summary"]["failure_fingerprint"]["top_failed_check"])
        self.assertEqual("median_jump", out["meta"]["qa_failure_fingerprint"]["top_failed_check"])


if __name__ == "__main__":
    unittest.main()
