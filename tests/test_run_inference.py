#!/usr/bin/env python3
"""
Unit tests for Model Inference & Determinism Runner (S1_prereg_pilot.md §4, §5 & Guideline §5.1).
"""

import unittest
from unittest.mock import MagicMock
from pathlib import Path

from scripts.pilot.run_inference import (
    PINNED_MODEL_NAME,
    PINNED_REVISION,
    PINNED_CPU_THREADS,
    PINNED_SEED,
    PINNED_MAX_NEW_TOKENS,
    PINNED_CONDITIONS,
    sha256_text,
    get_git_commit_hash,
    run_determinism_check,
)


class TestRunInference(unittest.TestCase):
    def test_pinned_constants(self):
        """Verify inference constants match S1_prereg_pilot.md §4 and §5 exactly."""
        self.assertEqual(PINNED_MODEL_NAME, "Qwen/Qwen2.5-Coder-1.5B-Instruct")
        self.assertEqual(PINNED_REVISION, "2e1fd397ee46e1388853d2af2c993145b0f1098a")
        self.assertEqual(PINNED_CPU_THREADS, 16)
        self.assertEqual(PINNED_SEED, 42)
        self.assertEqual(PINNED_MAX_NEW_TOKENS, 512)
        self.assertEqual(PINNED_CONDITIONS, ["A", "B", "C"])

    def test_sha256_hashing(self):
        """Verify SHA-256 computation conforms to standard cryptographic hex length."""
        sample_text = "def process_data(data):\n    return total"
        h = sha256_text(sample_text)
        self.assertEqual(len(h), 64)
        self.assertRegex(h, r"^[0-9a-f]{64}$")

    def test_git_commit_retrieval(self):
        """Verify git commit retrieval succeeds on master repository."""
        commit = get_git_commit_hash()
        self.assertTrue(len(commit) == 40 or commit == "unknown")

    def test_determinism_check_pass(self):
        """Verify determinism check passes when model outputs are bit-for-bit identical."""
        mock_runner = MagicMock()
        mock_runner.threads = 16
        mock_runner.generate.return_value = {
            "prompt_token_count": 100,
            "prompt_sha256": "abc123",
            "output_raw": "    total = 0\n    return total",
            "output_tokens": [10, 20, 30],
            "output_token_count": 3,
            "output_sha256": "hash_identical",
            "generation_time_sec": 0.1,
            "tokens_per_sec": 30.0,
        }

        demo_path = Path("tasks/pilot/demo_tasks.json")
        summary = run_determinism_check(
            runner=mock_runner,
            demo_tasks_path=demo_path,
            repetitions=3,
            output_path=None,
        )

        self.assertEqual(summary["status"], "PASS")
        self.assertTrue(summary["all_identical"])
        self.assertEqual(summary["total_generations"], 27)  # 3 tasks × 3 conditions × 3 runs
        self.assertEqual(len(summary["discrepancies"]), 0)

    def test_determinism_check_failure(self):
        """Verify determinism check halts and raises RuntimeError if runs diverge."""
        mock_runner = MagicMock()
        mock_runner.threads = 16
        # Simulate non-deterministic generation on third call
        call_count = 0
        def fake_generate(prompt):
            nonlocal call_count
            call_count += 1
            h = "hash_divergent" if call_count == 3 else "hash_normal"
            return {
                "prompt_token_count": 100,
                "prompt_sha256": "abc123",
                "output_raw": f"text_{h}",
                "output_tokens": [10, call_count],
                "output_token_count": 2,
                "output_sha256": h,
                "generation_time_sec": 0.1,
                "tokens_per_sec": 20.0,
            }

        mock_runner.generate.side_effect = fake_generate
        demo_path = Path("tasks/pilot/demo_tasks.json")

        with self.assertRaises(RuntimeError):
            run_determinism_check(
                runner=mock_runner,
                demo_tasks_path=demo_path,
                repetitions=3,
                output_path=None,
            )


if __name__ == "__main__":
    unittest.main()
