#!/usr/bin/env python3
"""
Unit tests for AST Skeleton Scorer (S1_prereg_pilot.md §6, §8).

Verifies:
1. Scorer self-test on demonstration tasks across Conditions A, B, C (all Category 3).
2. All 9 fixed negative controls score with expected category and error sub-code.
3. Strict category precedence (1: Unparsable, 2: Skeleton error, 3: Structurally correct).
4. Skeleton extraction, AST reduction, and E_PLACEMENT vs. E_OTHER classification.
"""

import json
import unittest
from pathlib import Path

from src.spatial_tax.scorer import (
    score_generation,
    extract_skeleton,
    skeletons_equal,
    classify_skeleton_error,
    run_scorer_self_test,
    Category,
    ParseErrorCode,
    SkeletonErrorCode,
    NEGATIVE_CONTROLS,
)


class TestScorer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        demo_path = Path("tasks/pilot/demo_tasks.json")
        if not demo_path.exists():
            demo_path = Path("tasks/demo_tasks.json")
        with open(demo_path, "r", encoding="utf-8") as f:
            cls.demo_tasks = json.load(f)

    def test_scorer_self_test_all_passed(self):
        """Verify the integrated run_scorer_self_test returns all_passed=True (§8)."""
        results = run_scorer_self_test(self.demo_tasks)
        self.assertTrue(results["all_passed"])
        self.assertEqual(len(results["canonical_tests"]), len(self.demo_tasks) * 3)
        self.assertEqual(len(results["negative_controls"]), len(NEGATIVE_CONTROLS))

    def test_canonical_demonstrations_score_category_3(self):
        """Canonical solutions converted to A, B, and C must score as Category 3 (§8)."""
        results = run_scorer_self_test(self.demo_tasks)
        for t in results["canonical_tests"]:
            self.assertTrue(
                t["passed"],
                f"Canonical task {t['task_id']} in condition {t['condition']} failed: {t['category']}",
            )
            self.assertEqual(t["category"], Category.STRUCTURALLY_CORRECT.name)

    def test_all_negative_controls(self):
        """All 9 fixed negative controls must produce exact expected category and sub-code (§8)."""
        results = run_scorer_self_test(self.demo_tasks)
        for c in results["negative_controls"]:
            self.assertTrue(
                c["passed"],
                f"Negative control '{c['name']}' failed: expected {c['expected']}, got {c['got']}",
            )

    def test_category_precedence_order(self):
        """
        Precedence (§6): Checked strictly in order:
        1. Unparsable (truncation, unclosed scope, syntax error)
        2. Skeleton error
        3. Structurally correct
        A truncated output with a missing statement must be Category 1 (TRUNCATED), not Category 2.
        """
        ref_code = (
            "def process_data(data):\n"
            "    total = 0\n"
            "    if len(data) > 0:\n"
            "        total += 1\n"
            "    return total"
        )
        # Incomplete / truncated text
        gen_truncated = "def process_data(data):\n    total = 0\n    if len(data) > 0:"
        res = score_generation(
            generated_text=gen_truncated,
            reference_python=ref_code,
            condition="A",
            is_truncated=True,
        )
        self.assertEqual(res.category, Category.UNPARSABLE)
        self.assertEqual(res.parse_error, ParseErrorCode.TRUNCATED)

    def test_skeleton_extraction_and_equality(self):
        """Verify AST skeleton ignores operators and constants, keeping only structure."""
        code1 = (
            "def process_data(data):\n"
            "    total = 0\n"
            "    for v1 in data:\n"
            "        total += 10\n"
            "    return total"
        )
        code2 = (
            "def process_data(data):\n"
            "    total = 0\n"
            "    for v1 in data:\n"
            "        total *= 99\n"
            "    return total"
        )
        skel1 = extract_skeleton(code1)
        skel2 = extract_skeleton(code2)
        self.assertIsNotNone(skel1)
        self.assertIsNotNone(skel2)
        self.assertTrue(skeletons_equal(skel1, skel2))

    def test_skeleton_error_classification(self):
        """Verify distinction between E_PLACEMENT and E_OTHER."""
        ref = (
            "def process_data(data):\n"
            "    total = 0\n"
            "    if len(data) > 0:\n"
            "        total += 1\n"
            "        total -= 2\n"
            "    return total"
        )
        # Same statements, moved outside block -> E_PLACEMENT
        placement_gen = (
            "def process_data(data):\n"
            "    total = 0\n"
            "    if len(data) > 0:\n"
            "        total += 1\n"
            "    total -= 2\n"
            "    return total"
        )
        res_placement = score_generation(placement_gen, ref, condition="A")
        self.assertEqual(res_placement.category, Category.SKELETON_ERROR)
        self.assertEqual(res_placement.skeleton_error, SkeletonErrorCode.E_PLACEMENT)

        # Missing statement -> E_OTHER
        missing_gen = (
            "def process_data(data):\n"
            "    total = 0\n"
            "    if len(data) > 0:\n"
            "        total += 1\n"
            "    return total"
        )
        res_other = score_generation(missing_gen, ref, condition="A")
        self.assertEqual(res_other.category, Category.SKELETON_ERROR)
        self.assertEqual(res_other.skeleton_error, SkeletonErrorCode.E_OTHER)


if __name__ == "__main__":
    unittest.main()
