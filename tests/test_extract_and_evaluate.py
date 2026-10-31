#!/usr/bin/env python3
"""
Unit tests for Output Extraction & Test Evaluation (S1_prereg_pilot.md §6).
"""

import unittest
from scripts.pilot.extract_and_evaluate import (
    extract_generation,
    find_content_errors,
    evaluate_single_generation,
    ExtractionResult,
)


class TestExtractAndEvaluate(unittest.TestCase):
    def setUp(self):
        self.ref_python = (
            "def process_data(data):\n"
            "    total = 0\n"
            "    if len(data) > 0:\n"
            "        total += 10\n"
            "        total *= 2\n"
            "    return total"
        )
        self.unit_tests = [[], [1, 2, 3], [-1, 5, 0, 2]]

    def test_special_token_and_overrun_cut(self):
        """Verify extraction cuts at first special token and logs overrun text."""
        raw_output = "    total = 0\n    return total<|endoftext|>extraneous text"
        ext = extract_generation(raw_output, condition="A", token_count=10)

        self.assertIn("<|endoftext|>", ext.special_tokens_seen)
        self.assertTrue(ext.has_overrun)
        self.assertIn("extraneous text", ext.overrun_text)
        self.assertNotIn("extraneous text", ext.extracted_body)

    def test_condition_a_zero_indent_cut(self):
        """Condition A cuts at first non-empty line with zero indentation."""
        raw_output = (
            "    total = 0\n"
            "    if len(data) > 0:\n"
            "        total += 10\n"
            "    return total\n"
            "print('Outside function')\n"
        )
        ext = extract_generation(raw_output, condition="A", token_count=20)
        self.assertTrue(ext.has_overrun)
        self.assertIn("print('Outside function')", ext.overrun_text)
        self.assertNotIn("Outside function", ext.extracted_body)
        self.assertIn("return total", ext.extracted_body)

    def test_condition_b_depth_tracking(self):
        """Condition B tracks depth starting at 1 and cuts after closing brace."""
        raw_output = (
            "    total = 0\n"
            "    if len(data) > 0 {\n"
            "        total += 10\n"
            "    }\n"
            "    return total\n"
            "}\n"
            "garbage after closing brace"
        )
        ext = extract_generation(raw_output, condition="B", token_count=25)
        self.assertEqual(ext.depth_at_cut, 0)
        self.assertFalse(ext.is_truncated)
        self.assertTrue(ext.has_overrun)
        self.assertIn("garbage after closing brace", ext.overrun_text)
        self.assertTrue(ext.full_function_code.endswith("}"))

    def test_truncation_detection(self):
        """token_count >= 512 without scope closure flags is_truncated."""
        unclosed_raw = "    total = 0\n    if len(data) > 0 {\n        total += 10\n"
        ext = extract_generation(unclosed_raw, condition="B", token_count=512)
        self.assertTrue(ext.is_truncated)

        # But if closed within 512 tokens, is_truncated is False
        closed_raw = unclosed_raw + "    }\n    return total\n}"
        ext_closed = extract_generation(closed_raw, condition="B", token_count=512)
        self.assertFalse(ext_closed.is_truncated)

    def test_evaluate_perfect_generation(self):
        """A correct generation yields Category 3, Pass@1 True, zero content errors."""
        raw_b = (
            "    total = 0\n"
            "    if len(data) > 0 {\n"
            "        total += 10\n"
            "        total *= 2\n"
            "    }\n"
            "    return total\n"
            "}<|endoftext|>"
        )
        eval_record = evaluate_single_generation(
            raw_output=raw_b,
            reference_python=self.ref_python,
            condition="B",
            token_count=30,
            unit_test_inputs=self.unit_tests,
            task_id="demo_test",
        )
        self.assertEqual(eval_record.category, 3)
        self.assertEqual(eval_record.category_name, "STRUCTURALLY_CORRECT")
        self.assertTrue(eval_record.pass_at_1)
        self.assertEqual(len(eval_record.content_errors), 0)

    def test_evaluate_content_error(self):
        """Category 3 with modified constant flags content error and fails Pass@1."""
        raw_b_wrong_const = (
            "    total = 0\n"
            "    if len(data) > 0 {\n"
            "        total += 99  # Changed constant\n"
            "        total *= 2\n"
            "    }\n"
            "    return total\n"
            "}<|endoftext|>"
        )
        eval_record = evaluate_single_generation(
            raw_output=raw_b_wrong_const,
            reference_python=self.ref_python,
            condition="B",
            token_count=30,
            unit_test_inputs=self.unit_tests,
            task_id="demo_test",
        )
        self.assertEqual(eval_record.category, 3)
        self.assertFalse(eval_record.pass_at_1)
        self.assertGreater(len(eval_record.content_errors), 0)


if __name__ == "__main__":
    unittest.main()
