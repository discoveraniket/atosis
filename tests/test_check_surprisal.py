#!/usr/bin/env python3
"""
Unit tests for Boundary-Span Surprisal Feasibility Checker (S1_prereg_pilot.md §7).
"""

import math
import unittest
from scripts.pilot.check_surprisal import (
    locate_boundary_span_tokens,
    compute_span_surprisal,
    SPACE_RUNS,
)


class TestCheckSurprisal(unittest.TestCase):
    def test_space_runs_specification(self):
        """Verify audited space counts match §7: 4, 8, 12, 16, 20 spaces."""
        self.assertEqual(SPACE_RUNS, [4, 8, 12, 16, 20])

    def test_locate_boundary_span_tokens(self):
        """Verify boundary token location using character offset mapping."""
        # Mock token records
        text = "total *= 34\n        total += 33"
        # Character offsets:
        # 'total *= 34' is chars 0-11
        # '\n' is chars 11-12
        # '        ' is chars 12-20
        # 'total' is chars 20-25
        # ' += 33' is chars 25-31
        token_records = [
            {"index": 0, "token_str": "total", "char_span": [0, 5], "surprisal": 1.0},
            {"index": 1, "token_str": " *=", "char_span": [5, 8], "surprisal": 2.0},
            {"index": 2, "token_str": " 34", "char_span": [8, 11], "surprisal": 3.0},
            {"index": 3, "token_str": "\n", "char_span": [11, 12], "surprisal": 0.5},
            {"index": 4, "token_str": "        ", "char_span": [12, 20], "surprisal": 4.0},
            {"index": 5, "token_str": "total", "char_span": [20, 25], "surprisal": 1.5},
            {"index": 6, "token_str": " +=", "char_span": [25, 28], "surprisal": 2.0},
            {"index": 7, "token_str": " 33", "char_span": [28, 31], "surprisal": 3.0},
        ]

        tok1, tok2 = locate_boundary_span_tokens(
            token_records=token_records,
            full_text=text,
            inner_last_stmt="total *= 34",
            next_enclosing_stmt="total += 33",
        )

        # tok1 should be token 2 (' 34')
        self.assertEqual(tok1, 2)
        # tok2 should be token 5 ('total')
        self.assertEqual(tok2, 5)

    def test_compute_span_surprisal(self):
        """Verify primary and sensitivity span surprisal calculation and units."""
        token_records = [
            {"index": 0, "token_str": "a", "token_id": 10, "surprisal": 1.0},
            {"index": 1, "token_str": "b", "token_id": 20, "surprisal": 2.0},
            {"index": 2, "token_str": "c", "token_id": 30, "surprisal": 3.0},
            {"index": 3, "token_str": "d", "token_id": 40, "surprisal": 4.0},
        ]

        # Span from token 1 to token 3
        span_metrics = compute_span_surprisal(token_records, start_idx=1, end_idx=3)

        # Primary span: tokens 1, 2, 3 -> surprisals: 2.0 + 3.0 + 4.0 = 9.0
        primary = span_metrics["primary_span"]
        self.assertEqual(primary["token_count"], 3)
        self.assertEqual(primary["tokens"], ["b", "c", "d"])
        self.assertEqual(primary["total_surprisal_nats"], 9.0)
        self.assertEqual(primary["mean_surprisal_nats"], 3.0)
        self.assertAlmostEqual(primary["total_surprisal_bits"], 9.0 / math.log(2), places=4)

        # Sensitivity span: tokens 1, 2 -> surprisals: 2.0 + 3.0 = 5.0
        sensitivity = span_metrics["sensitivity_span"]
        self.assertEqual(sensitivity["token_count"], 2)
        self.assertEqual(sensitivity["tokens"], ["b", "c"])
        self.assertEqual(sensitivity["total_surprisal_nats"], 5.0)
        self.assertEqual(sensitivity["mean_surprisal_nats"], 2.5)
        self.assertAlmostEqual(sensitivity["total_surprisal_bits"], 5.0 / math.log(2), places=4)


if __name__ == "__main__":
    unittest.main()
