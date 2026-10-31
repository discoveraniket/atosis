#!/usr/bin/env python3
"""
Unit tests for Prompt Builder and Demonstration Format Validator (S1_prereg_pilot.md §5).
"""

import json
import unittest
from pathlib import Path

from scripts.pilot.build_prompts import (
    validate_demonstration_format,
    build_demonstrations_block,
    build_prompt,
    get_condition_signature,
    FormatValidationError,
    SIGNATURE_A,
    SIGNATURE_BC,
)


class TestBuildPrompts(unittest.TestCase):
    def setUp(self):
        demo_path = Path("tasks/pilot/demo_tasks.json")
        if not demo_path.exists():
            demo_path = Path("tasks/demo_tasks.json")
        with open(demo_path) as f:
            self.demo_tasks = json.load(f)

    def test_signatures(self):
        """Condition A uses colon, Conditions B & C use brace."""
        self.assertEqual(get_condition_signature("A"), SIGNATURE_A)
        self.assertEqual(get_condition_signature("B"), SIGNATURE_BC)
        self.assertEqual(get_condition_signature("C"), SIGNATURE_BC)
        with self.assertRaises(ValueError):
            get_condition_signature("D")

    def test_demonstration_format_condition_a(self):
        """Condition A rejects braces and requires colons on block headers."""
        valid_a = (
            "def process_data(data):\n"
            "    total = 0\n"
            "    if len(data) > 0:\n"
            "        total += 1\n"
            "    return total"
        )
        validate_demonstration_format(valid_a, "A")

        # Braces in Condition A
        invalid_brace = valid_a.replace("if len(data) > 0:", "if len(data) > 0 {")
        with self.assertRaises(FormatValidationError):
            validate_demonstration_format(invalid_brace, "A")

        # Missing colon in Condition A
        invalid_no_colon = valid_a.replace("if len(data) > 0:", "if len(data) > 0")
        with self.assertRaises(FormatValidationError):
            validate_demonstration_format(invalid_no_colon, "A")

    def test_demonstration_format_conditions_b_and_c(self):
        """Conditions B and C reject colons before braces and require opening braces."""
        valid_b = (
            "def process_data(data) {\n"
            "    total = 0\n"
            "    if len(data) > 0 {\n"
            "        total += 1\n"
            "    }\n"
            "    return total\n"
            "}"
        )
        validate_demonstration_format(valid_b, "B")

        valid_c = (
            "def process_data(data) {\n"
            "total = 0\n"
            "if len(data) > 0 {\n"
            "total += 1\n"
            "}\n"
            "return total\n"
            "}"
        )
        validate_demonstration_format(valid_c, "C")

        # Colon before brace ': {' must be rejected
        colon_brace = valid_b.replace("if len(data) > 0 {", "if len(data) > 0: {")
        with self.assertRaises(FormatValidationError):
            validate_demonstration_format(colon_brace, "B")

        # Missing brace on block header
        no_brace = valid_b.replace("if len(data) > 0 {", "if len(data) > 0:")
        with self.assertRaises(FormatValidationError):
            validate_demonstration_format(no_brace, "B")

        # Unbalanced braces
        unbalanced = valid_b[:-2]  # drop final '}'
        with self.assertRaises(FormatValidationError):
            validate_demonstration_format(unbalanced, "B")

        # Indentation in Condition C must be rejected
        with self.assertRaises(FormatValidationError):
            validate_demonstration_format(valid_b, "C")

    def test_prompt_layout_and_termination(self):
        """Prompt follows §5 layout and ends with exactly one newline after signature."""
        target_task = self.demo_tasks[0]

        for cond in ("A", "B", "C"):
            prompt = build_prompt(self.demo_tasks, target_task, cond)

            # Ends with exactly one newline
            self.assertTrue(prompt.endswith("\n"))
            self.assertFalse(prompt.endswith("\n\n"))

            # Signature matches condition
            expected_sig = SIGNATURE_A if cond == "A" else SIGNATURE_BC
            self.assertTrue(prompt.rstrip().endswith(expected_sig))

            # Prompt contains 3 demonstrations separated by \n\n
            parts = prompt.split("\n\n")
            # 3 demonstrations + 1 target block = 4 main sections
            self.assertEqual(len(parts), 4, f"Condition {cond} expected 4 sections, got {len(parts)}")


if __name__ == "__main__":
    unittest.main()
