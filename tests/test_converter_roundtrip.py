#!/usr/bin/env python3
"""
Converter Round-Trip Verification Test (S1_prereg_pilot.md §8).

Automated test verifying:
1. Canonical Python source is converted to Condition B and to Condition C.
2. Both are converted back to Python using the §6 converter rules.
3. Both converted-back results equal the canonical source exactly (exact string match).
4. Condition C equals Condition B with leading whitespace removed from every line.
5. Unit tests yield identical outputs on canonical vs. converted-back versions (both B and C).
"""

import json
import random
import unittest
from pathlib import Path
from typing import Dict, Any, List

from src.spatial_tax.converter import (
    python_to_condition_b,
    condition_b_to_condition_c,
    python_to_condition_c,
    convert_brace_to_python,
    ConversionError,
)
from src.spatial_tax.mutants import execute_function
from src.spatial_tax.generator import generate_task


def verify_task_roundtrip(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify all 5 round-trip conditions for a given task specification dictionary.
    Returns a dictionary of boolean passes and diagnostic details.
    """
    canonical = task["canonical_python"]
    inputs = task.get("unit_test_inputs", [[], [1, 2, 3], [-1, 5, 0, 2]])

    # 1. Convert to Condition B and Condition C
    code_b = python_to_condition_b(canonical)
    code_c = python_to_condition_c(canonical)

    # 4. Condition C must equal Condition B with leading whitespace removed from every line
    expected_c = condition_b_to_condition_c(code_b)
    cond_c_matches_stripped_b = (code_c == expected_c)

    # 2. Both converted back to Python with Section 6 rules
    conv_b = convert_brace_to_python(code_b)
    conv_c = convert_brace_to_python(code_c)

    b_no_errors = (len(conv_b.errors) == 0 and conv_b.depth_at_end == 0)
    c_no_errors = (len(conv_c.errors) == 0 and conv_c.depth_at_end == 0)

    # 3. Both results equal canonical source exactly (exact string match)
    b_exact_match = (conv_b.python_code == canonical)
    c_exact_match = (conv_c.python_code == canonical)

    # 5. Unit tests give identical outputs on canonical vs. converted-back versions
    ref_outputs = execute_function(canonical, inputs)
    out_b = execute_function(conv_b.python_code, inputs)
    out_c = execute_function(conv_c.python_code, inputs)

    unit_tests_b_match = (out_b == ref_outputs)
    unit_tests_c_match = (out_c == ref_outputs)

    all_passed = (
        cond_c_matches_stripped_b
        and b_no_errors
        and c_no_errors
        and b_exact_match
        and c_exact_match
        and unit_tests_b_match
        and unit_tests_c_match
    )

    return {
        "task_id": task.get("task_id", "unknown"),
        "depth": task.get("depth", 0),
        "all_passed": all_passed,
        "cond_c_matches_stripped_b": cond_c_matches_stripped_b,
        "b_no_errors": b_no_errors,
        "c_no_errors": c_no_errors,
        "b_exact_match": b_exact_match,
        "c_exact_match": c_exact_match,
        "unit_tests_b_match": unit_tests_b_match,
        "unit_tests_c_match": unit_tests_c_match,
        "ref_outputs": ref_outputs,
        "out_b": out_b,
        "out_c": out_c,
    }


class TestConverterRoundTrip(unittest.TestCase):
    """Test suite for converter round-trip verification per §8."""

    def test_demonstrations_roundtrip(self):
        """Verify 100% roundtrip agreement on the 3 pre-registered demonstration tasks."""
        demo_path = Path("tasks/pilot/demo_tasks.json")
        if not demo_path.exists():
            demo_path = Path("tasks/demo_tasks.json")

        self.assertTrue(demo_path.exists(), f"Demonstrations file not found at {demo_path}")

        with open(demo_path) as f:
            demo_tasks = json.load(f)

        self.assertEqual(len(demo_tasks), 3, "Expected exactly 3 demonstration tasks (d=1, 2, 3)")

        for task in demo_tasks:
            result = verify_task_roundtrip(task)
            task_id = result["task_id"]
            self.assertTrue(
                result["cond_c_matches_stripped_b"],
                f"{task_id}: Condition C did not match Condition B with whitespace stripped",
            )
            self.assertTrue(result["b_no_errors"], f"{task_id}: Conversion errors in B: {result}")
            self.assertTrue(result["c_no_errors"], f"{task_id}: Conversion errors in C: {result}")
            self.assertTrue(
                result["b_exact_match"],
                f"{task_id}: Converted B code does not match canonical source exactly",
            )
            self.assertTrue(
                result["c_exact_match"],
                f"{task_id}: Converted C code does not match canonical source exactly",
            )
            self.assertTrue(
                result["unit_tests_b_match"],
                f"{task_id}: Converted B unit tests differ: {result['out_b']} != {result['ref_outputs']}",
            )
            self.assertTrue(
                result["unit_tests_c_match"],
                f"{task_id}: Converted C unit tests differ: {result['out_c']} != {result['ref_outputs']}",
            )
            self.assertTrue(result["all_passed"], f"{task_id}: Failed overall roundtrip verification")

    def test_synthetic_dummy_tasks_roundtrip(self):
        """
        Verify round-trip on newly generated synthetic tasks across all depths d in {1, 3, 5}.
        Guarantees evaluation-set isolation (synthetic dummy seed 9999).
        """
        dummy_rng = random.Random(9999)
        test_depths = [1, 2, 3, 5]

        for depth in test_depths:
            for i in range(5):  # 5 sample tasks per depth
                task_id = f"dummy_d{depth}_{i:03d}"
                task_spec = generate_task(task_id, depth, dummy_rng)
                task_dict = {
                    "task_id": task_spec.task_id,
                    "depth": task_spec.depth,
                    "canonical_python": task_spec.canonical_python,
                    "unit_test_inputs": task_spec.unit_test_inputs,
                }
                result = verify_task_roundtrip(task_dict)
                self.assertTrue(
                    result["all_passed"],
                    f"Synthetic task {task_id} failed roundtrip verification: {result}",
                )

    def test_post_freeze_tasks_if_present(self):
        """
        If official test_tasks.json exists (post-freeze), verify all 90 tasks pass 100%.
        Skipped gracefully pre-freeze to preserve evaluation-set isolation.
        """
        test_path = Path("tasks/pilot/test_tasks.json")
        if not test_path.exists():
            test_path = Path("tasks/test_tasks.json")

        if not test_path.exists():
            # Expected pre-freeze: official tasks are not generated until Phase 5
            return

        with open(test_path) as f:
            test_tasks = json.load(f)

        failures = []
        for task in test_tasks:
            result = verify_task_roundtrip(task)
            if not result["all_passed"]:
                failures.append(result)

        self.assertEqual(
            len(failures),
            0,
            f"Post-freeze roundtrip failed on {len(failures)} tasks: {[f['task_id'] for f in failures]}",
        )


if __name__ == "__main__":
    unittest.main()
