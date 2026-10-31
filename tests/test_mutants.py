#!/usr/bin/env python3
"""
Unit tests for single-move scope mutant generator and filter (S1_prereg_pilot.md §3, §8).
"""

import unittest
from src.spatial_tax.mutants import (
    generate_scope_mutants,
    check_task_mutants,
    execute_function,
    is_update_stmt,
)
import ast


class TestMutants(unittest.TestCase):
    def setUp(self):
        self.unit_test_inputs = [[], [1, 2, 3], [-1, 5, 0, 2]]

    def test_update_stmt_detection(self):
        """Verify update statements are identified and total = 0 is protected."""
        assign_zero = ast.parse("total = 0").body[0]
        assign_val = ast.parse("total = 42").body[0]
        aug_assign = ast.parse("total += 5").body[0]
        other_assign = ast.parse("x = 10").body[0]
        ret_stmt = ast.parse("return total").body[0]

        self.assertFalse(is_update_stmt(assign_zero))
        self.assertTrue(is_update_stmt(assign_val))
        self.assertTrue(is_update_stmt(aug_assign))
        self.assertFalse(is_update_stmt(other_assign))
        self.assertFalse(is_update_stmt(ret_stmt))

    def test_d1_mutants(self):
        """Depth 1 has 1 block with 3 statements -> 1 outward mutant."""
        code_d1 = (
            "def process_data(data):\n"
            "    total = 0\n"
            "    if len(data) > 0:\n"
            "        total += 10\n"
            "        total *= 2\n"
            "        total += 5\n"
            "    return total\n"
        )
        mutants = generate_scope_mutants(code_d1)
        self.assertEqual(len(mutants), 1)
        self.assertEqual(mutants[0].move_type, "outward")
        self.assertEqual(mutants[0].target_level, 0)

        # Confirm the outward move placed 'total += 5' outside the if block
        expected_mutant_code = (
            "def process_data(data):\n"
            "    total = 0\n"
            "    if len(data) > 0:\n"
            "        total += 10\n"
            "        total *= 2\n"
            "    total += 5\n"
            "    return total"
        )
        self.assertEqual(mutants[0].mutated_python.strip(), expected_mutant_code.strip())

    def test_empty_block_prevention(self):
        """A move that would leave a block empty is not generated (§3)."""
        code_single_stmt = (
            "def process_data(data):\n"
            "    total = 0\n"
            "    if len(data) > 0:\n"
            "        total += 10\n"
            "    return total\n"
        )
        mutants = generate_scope_mutants(code_single_stmt)
        # Cannot move outward because if block would be left with 0 statements
        # Cannot move inward because next statement is return total
        self.assertEqual(len(mutants), 0)

    def test_d3_mutants_and_inward_move(self):
        """Verify inward and outward moves on a nested d=3 task."""
        code_d3 = (
            "def process_data(data):\n"
            "    total = 0\n"
            "    if len(data) > 0:\n"
            "        total += 1\n"
            "        total -= 2\n"
            "        for v1 in data:\n"
            "            total *= 3\n"
            "            total += 4\n"
            "            if v1 > 0:\n"
            "                total -= 5\n"
            "                total += 6\n"
            "            total += 7\n"
            "        total *= 8\n"
            "    return total\n"
        )
        mutants = generate_scope_mutants(code_d3)
        # Expected:
        # Level 0 (outer if): outward (1)
        # Level 1 (for): outward (1), inward (1)
        # Level 2 (inner if): outward (1), inward (1)
        self.assertEqual(len(mutants), 5)
        move_types = [(m.move_type, m.target_level) for m in mutants]
        self.assertIn(("outward", 0), move_types)
        self.assertIn(("outward", 1), move_types)
        self.assertIn(("inward", 1), move_types)
        self.assertIn(("outward", 2), move_types)
        self.assertIn(("inward", 2), move_types)

        # Verify inward move to level 2 (inner if) took 'total += 7' inside it
        inward_lvl2 = [m for m in mutants if m.move_type == "inward" and m.target_level == 2][0]
        self.assertIn("total -= 5\n                total += 6\n                total += 7", inward_lvl2.mutated_python)

    def test_check_task_mutants_evaluation(self):
        """Verify catch rate calculation against unit tests."""
        code_d3 = (
            "def process_data(data):\n"
            "    total = 0\n"
            "    if len(data) > 0:\n"
            "        total += 1\n"
            "        total -= 2\n"
            "        for v1 in data:\n"
            "            total *= 3\n"
            "            total += 4\n"
            "            if v1 > 0:\n"
            "                total -= 5\n"
            "                total += 6\n"
            "            total += 7\n"
            "        total *= 8\n"
            "    return total\n"
        )
        result = check_task_mutants(code_d3, self.unit_test_inputs, task_id="demo_test")
        self.assertEqual(result.total_mutants, 5)
        self.assertGreaterEqual(result.catch_rate, 0.80)
        self.assertTrue(result.passed)


class TestGeneratorConstraints(unittest.TestCase):
    """Verify Task Generator structural rules (§3)."""

    def test_generator_header_constraints(self):
        """
        Verify:
        1. for v2 in data: appears at most once per task.
        2. <v> resolves to the innermost active loop variable (v2 if v2 is active, v1 otherwise).
        3. No consecutive repeats of block header templates.
        """
        import random
        from src.spatial_tax.generator import generate_candidate_task

        for seed in range(50):
            rng = random.Random(seed)
            task = generate_candidate_task(f"test_{seed}", depth=5, rng=rng)
            lines = [l.strip() for l in task.canonical_python.splitlines()]
            headers = [l for l in lines if l.endswith(":")]

            # 1. for v2 in data: at most once per task
            v2_headers = [h for h in headers if "for v2 in data:" in h]
            self.assertLessEqual(len(v2_headers), 1, f"Seed {seed} had multiple v2 loops: {headers}")

            # 2. Innermost active loop variable resolution
            seen_v2 = False
            for h in headers:
                if "for v2 in data:" in h:
                    seen_v2 = True
                if "v1 > 0" in h or "v1 % 2 == 0" in h:
                    self.assertFalse(seen_v2, f"Seed {seed} used v1 after v2 was opened: {headers}")
                if "v2 > 0" in h or "v2 % 2 == 0" in h:
                    self.assertTrue(seen_v2, f"Seed {seed} used v2 before v2 was opened: {headers}")

            # 3. No consecutive repeats of deeper header templates
            for idx in range(len(headers) - 1):
                h1, h2 = headers[idx], headers[idx + 1]
                self.assertNotEqual(h1, h2, f"Consecutive repeat of header in seed {seed}: {h1}")


if __name__ == "__main__":
    unittest.main()
