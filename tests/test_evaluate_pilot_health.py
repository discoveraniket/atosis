#!/usr/bin/env python3
"""
Unit tests for Pilot Health Check & Analysis Reporter (S1_prereg_pilot.md §8).
"""

import unittest
from scripts.pilot.evaluate_pilot_health import (
    compute_cell_metrics,
    evaluate_health_gates,
    sample_extraction_hand_check,
    HAND_CHECK_SEED,
)


class TestEvaluatePilotHealth(unittest.TestCase):
    def test_compute_cell_metrics(self):
        """Verify structural bounds, parse failures, and failure rates per §6 and §8."""
        records = [
            # 10 records total
            # 2 Cat 1: 1 truncated, 1 parse error (E_SYNTAX)
            {"category": 1, "parse_error": "truncated", "pass_at_1": False},
            {"category": 1, "parse_error": "E_SYNTAX", "pass_at_1": False},
            # 3 Cat 2: 2 E_PLACEMENT, 1 E_OTHER
            {"category": 2, "skeleton_error": "E_PLACEMENT", "pass_at_1": False},
            {"category": 2, "skeleton_error": "E_PLACEMENT", "pass_at_1": False},
            {"category": 2, "skeleton_error": "E_OTHER", "pass_at_1": False},
            # 5 Cat 3: 4 pass unit tests, 1 fails
            {"category": 3, "pass_at_1": True, "content_errors": []},
            {"category": 3, "pass_at_1": True, "content_errors": []},
            {"category": 3, "pass_at_1": True, "content_errors": []},
            {"category": 3, "pass_at_1": True, "content_errors": []},
            {"category": 3, "pass_at_1": False, "content_errors": ["err1"]},
        ]

        metrics = compute_cell_metrics(records)

        self.assertEqual(metrics["total_generations"], 10)
        self.assertEqual(metrics["category_1_count"], 2)
        self.assertEqual(metrics["category_2_count"], 3)
        self.assertEqual(metrics["category_3_count"], 5)
        self.assertEqual(metrics["truncated_count"], 1)
        self.assertEqual(metrics["parse_failure_count"], 1)  # Cat 1 - truncated

        # Headline rate: Cat 1 + Cat 2 = 5/10 = 0.50
        self.assertEqual(metrics["upper_bound_structural_failure_rate"], 0.50)
        # Lower bound: Cat 1 + E_PLACEMENT = 2 + 2 = 4/10 = 0.40
        self.assertEqual(metrics["lower_bound_structural_failure_rate"], 0.40)
        # E_OTHER rate = 1/10 = 0.10
        self.assertEqual(metrics["e_other_rate"], 0.10)
        # Pass@1 rate = 4/10 = 0.40
        self.assertEqual(metrics["pass_at_1_rate"], 0.40)
        # Truncation rate = 1/10 = 0.10
        self.assertEqual(metrics["truncation_rate"], 0.10)

    def test_evaluate_health_gates_passing(self):
        """Verify passing gates evaluation when all §8 conditions are met."""
        mock_cells = {
            "d1_A": {"pass_at_1_rate": 0.50, "total_generations": 30, "pass_at_1_count": 15, "parse_failure_rate": 0.03, "truncation_rate": 0.0},
            "d1_B": {"pass_at_1_rate": 0.60, "total_generations": 30, "pass_at_1_count": 18, "parse_failure_rate": 0.03, "truncation_rate": 0.0},
            "d1_C": {"pass_at_1_rate": 0.55, "total_generations": 30, "pass_at_1_count": 16, "parse_failure_rate": 0.03, "truncation_rate": 0.0},
            "d3_A": {"pass_at_1_rate": 0.30, "total_generations": 30, "pass_at_1_count": 9, "parse_failure_rate": 0.0, "truncation_rate": 0.0},
            "d3_B": {"pass_at_1_rate": 0.40, "total_generations": 30, "pass_at_1_count": 12, "parse_failure_rate": 0.0, "truncation_rate": 0.0},
            "d3_C": {"pass_at_1_rate": 0.35, "total_generations": 30, "pass_at_1_count": 10, "parse_failure_rate": 0.0, "truncation_rate": 0.0},
            "d5_A": {"pass_at_1_rate": 0.10, "total_generations": 30, "pass_at_1_count": 3, "parse_failure_rate": 0.0, "truncation_rate": 0.0},
            "d5_B": {"pass_at_1_rate": 0.15, "total_generations": 30, "pass_at_1_count": 4, "parse_failure_rate": 0.0, "truncation_rate": 0.0},
            "d5_C": {"pass_at_1_rate": 0.12, "total_generations": 30, "pass_at_1_count": 3, "parse_failure_rate": 0.0, "truncation_rate": 0.0},
        }

        eval_gates = evaluate_health_gates(mock_cells)
        self.assertTrue(eval_gates["all_gates_passed"])
        self.assertTrue(eval_gates["gates"]["d1_pass_rate_floor"]["passed"])
        self.assertTrue(eval_gates["gates"]["d1_pass_rate_ceiling"]["passed"])
        self.assertTrue(eval_gates["gates"]["d5_pass_rate_floor"]["passed"])
        self.assertTrue(eval_gates["gates"]["d1_parse_failures"]["passed"])
        self.assertTrue(eval_gates["gates"]["truncation_rate"]["passed"])

    def test_evaluate_health_gates_failure(self):
        """Verify gate failures are caught explicitly."""
        # Case: d1 best rate is 0.30 (< 40% floor target)
        mock_cells = {
            "d1_A": {"pass_at_1_rate": 0.20, "total_generations": 30, "pass_at_1_count": 6, "parse_failure_rate": 0.0, "truncation_rate": 0.0},
            "d1_B": {"pass_at_1_rate": 0.30, "total_generations": 30, "pass_at_1_count": 9, "parse_failure_rate": 0.0, "truncation_rate": 0.0},
            "d1_C": {"pass_at_1_rate": 0.25, "total_generations": 30, "pass_at_1_count": 7, "parse_failure_rate": 0.0, "truncation_rate": 0.0},
            "d5_A": {"pass_at_1_rate": 0.02, "total_generations": 30, "pass_at_1_count": 1, "parse_failure_rate": 0.0, "truncation_rate": 0.0},
            "d5_B": {"pass_at_1_rate": 0.03, "total_generations": 30, "pass_at_1_count": 1, "parse_failure_rate": 0.0, "truncation_rate": 0.0},
            "d5_C": {"pass_at_1_rate": 0.01, "total_generations": 30, "pass_at_1_count": 0, "parse_failure_rate": 0.0, "truncation_rate": 0.0},
        }
        eval_gates = evaluate_health_gates(mock_cells)
        self.assertFalse(eval_gates["all_gates_passed"])
        self.assertFalse(eval_gates["gates"]["d1_pass_rate_floor"]["passed"])
        self.assertFalse(eval_gates["gates"]["d5_pass_rate_floor"]["passed"])

    def test_stratified_hand_check_sampling(self):
        """Verify sampling extracts exactly 20 samples per condition (6 d=1, 7 d=3, 7 d=5)."""
        # Create synthetic records: 30 per depth per condition
        records = []
        for cond in ("A", "B", "C"):
            for depth in (1, 3, 5):
                for idx in range(30):
                    records.append({
                        "task_id": f"task_{cond}_d{depth}_{idx:02d}",
                        "condition": cond,
                        "depth": depth,
                        "extraction": {
                            "raw_text": f"raw_output_{cond}_{depth}_{idx}",
                            "extracted_body": f"body_{cond}_{depth}_{idx}",
                            "cut_reason": "test_cut",
                        },
                    })

        samples = sample_extraction_hand_check(records, seed=HAND_CHECK_SEED)

        for cond in ("A", "B", "C"):
            cond_samples = samples[cond]
            self.assertEqual(len(cond_samples), 20)
            d1_count = sum(1 for s in cond_samples if s["depth"] == 1)
            d3_count = sum(1 for s in cond_samples if s["depth"] == 3)
            d5_count = sum(1 for s in cond_samples if s["depth"] == 5)
            self.assertEqual(d1_count, 6)
            self.assertEqual(d3_count, 7)
            self.assertEqual(d5_count, 7)


if __name__ == "__main__":
    unittest.main()
