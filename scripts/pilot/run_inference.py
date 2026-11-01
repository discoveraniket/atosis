#!/usr/bin/env python3
"""
Model Inference & Determinism Runner (S1_prereg_pilot.md §4, §5 & Guideline §5.1).

Executes model generation under pinned experimental conditions:
- Model: Qwen/Qwen2.5-Coder-1.5B-Instruct
- Revision: 2e1fd397ee46e1388853d2af2c993145b0f1098a
- Hardware: CPU only, pinned to 16 threads (torch.set_num_threads(16))
- Precision: native bfloat16 (dtype=torch.bfloat16)
- Decoding: greedy (do_sample=False, max_new_tokens=512)
- Random seed: torch.manual_seed(42)
- Special tokens: preserved in raw output (skip_special_tokens=False)

Pre-Run Determinism Check (§5):
Runs 3 demonstration tasks across 3 conditions (9 prompts) 3 times.
All outputs and token sequences must be 100% bit-for-bit identical across runs.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Support direct script and module execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pilot.build_prompts import build_prompt


# §4 & §5 Pinned Constants
PINNED_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
PINNED_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
PINNED_CPU_THREADS = 16
PINNED_SEED = 42
PINNED_MAX_NEW_TOKENS = 512
PINNED_CONDITIONS = ["A", "B", "C"]

DEFAULT_SNAPSHOT_DIR = (
    PROJECT_ROOT / "models" / "hf_cache" / "hub" /
    "models--Qwen--Qwen2.5-Coder-1.5B-Instruct" / "snapshots" / PINNED_REVISION
)


def get_git_commit_hash() -> str:
    """Retrieve current Git commit hash (git rev-parse HEAD) per Guideline §6.1."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            text=True
        ).strip()
        if not commit or len(commit) < 7:
            raise ValueError(f"Invalid Git commit hash: '{commit}'")
        return commit
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve Git commit hash per Guideline §6.1: {e}") from e


def sha256_text(text: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ModelRunner:
    """Handles deterministic model loading and greedy inference on CPU."""

    def __init__(self, snapshot_dir: Optional[Path] = None):
        self.snapshot_dir = Path(snapshot_dir or DEFAULT_SNAPSHOT_DIR)
        if not self.snapshot_dir.exists():
            raise FileNotFoundError(
                f"Model snapshot directory not found: {self.snapshot_dir}\n"
                f"Expected pinned revision {PINNED_REVISION}."
            )

        # Enforce CPU thread pinning (§4)
        torch.set_num_threads(PINNED_CPU_THREADS)
        self.threads = torch.get_num_threads()

        print(f"Loading tokenizer from: {self.snapshot_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.snapshot_dir))

        print(f"Loading model in native bfloat16 on CPU ({self.threads} threads)...")
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.snapshot_dir),
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        t1 = time.time()
        print(f"Model loaded in {t1 - t0:.2f}s.")

    def generate(self, prompt: str) -> Dict[str, Any]:
        """
        Run greedy completion on a prompt per §5.
        Returns raw output text, token IDs, SHA-256 hashes, and execution metrics.
        """
        # Tokenize prompt without special template formatting
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        prompt_token_count = input_ids.shape[1]
        prompt_hash = sha256_text(prompt)

        # Pin seed prior to generation (§5)
        torch.manual_seed(PINNED_SEED)

        t0 = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=PINNED_MAX_NEW_TOKENS,
                do_sample=False,  # Greedy decoding
                pad_token_id=self.tokenizer.eos_token_id,
            )
        t1 = time.time()

        # Slice off input tokens to isolate newly generated completion
        gen_tokens = output_ids[0][prompt_token_count:].tolist()
        gen_token_count = len(gen_tokens)

        # Decode keeping special tokens (skip_special_tokens=False per §6)
        raw_output_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=False)
        output_hash = sha256_text(raw_output_text)

        return {
            "prompt_token_count": prompt_token_count,
            "prompt_sha256": prompt_hash,
            "output_raw": raw_output_text,
            "output_tokens": gen_tokens,
            "output_token_count": gen_token_count,
            "output_sha256": output_hash,
            "generation_time_sec": round(t1 - t0, 3),
            "tokens_per_sec": round(gen_token_count / max(t1 - t0, 1e-6), 2),
        }


def run_determinism_check(
    runner: ModelRunner,
    demo_tasks_path: Path,
    repetitions: int = 3,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run the pre-run Determinism Check (§5):
    Run 3 demonstration tasks across all 3 conditions (9 prompts) 3 times.
    Verifies that all outputs and token sequences are 100% bit-for-bit identical across runs.
    """
    with open(demo_tasks_path) as f:
        demos = json.load(f)

    if len(demos) != 3:
        raise ValueError(f"Expected 3 demonstration tasks, found {len(demos)}")

    git_commit = get_git_commit_hash()
    timestamp_start = datetime.now(timezone.utc).isoformat()

    print("\n" + "=" * 70)
    print(f"STARTING DETERMINISM CHECK (§5): 3 tasks × 3 conditions × {repetitions} runs = {3 * 3 * repetitions} generations")
    print("=" * 70)

    results_by_prompt: Dict[str, List[Dict[str, Any]]] = {}
    total_generations = 0
    all_identical = True
    discrepancies: List[Dict[str, Any]] = []

    for task_idx, task in enumerate(demos):
        task_id = task["task_id"]
        depth = task["depth"]

        for condition in PINNED_CONDITIONS:
            prompt_key = f"{task_id}_cond_{condition}"
            prompt_text = build_prompt(demos, task, condition)
            prompt_hash = sha256_text(prompt_text)

            print(f"\nPrompt [{prompt_key}] (Depth {depth}):")
            runs = []

            for run_num in range(1, repetitions + 1):
                gen_result = runner.generate(prompt_text)
                total_generations += 1

                run_entry = {
                    "run_number": run_num,
                    "task_id": task_id,
                    "depth": depth,
                    "condition": condition,
                    "prompt_sha256": prompt_hash,
                    "output_sha256": gen_result["output_sha256"],
                    "output_token_count": gen_result["output_token_count"],
                    "output_tokens": gen_result["output_tokens"],
                    "output_raw": gen_result["output_raw"],
                    "generation_time_sec": gen_result["generation_time_sec"],
                }
                runs.append(run_entry)
                print(f"  Run {run_num}: {gen_result['output_token_count']} tokens, SHA-256={gen_result['output_sha256'][:12]}... ({gen_result['generation_time_sec']}s)")

            # Check bit-for-bit equality across runs
            first_hash = runs[0]["output_sha256"]
            first_tokens = runs[0]["output_tokens"]
            first_raw = runs[0]["output_raw"]

            prompt_identical = True
            for r in runs[1:]:
                if r["output_sha256"] != first_hash or r["output_tokens"] != first_tokens or r["output_raw"] != first_raw:
                    prompt_identical = False
                    all_identical = False
                    discrepancies.append({
                        "prompt_key": prompt_key,
                        "run_1_hash": first_hash,
                        "differing_run": r["run_number"],
                        "differing_hash": r["output_sha256"],
                    })

            status_str = "PASS (100% bit-for-bit identical)" if prompt_identical else "FAIL (NON-DETERMINISTIC)"
            print(f"  Result: {status_str}")
            results_by_prompt[prompt_key] = runs

    timestamp_end = datetime.now(timezone.utc).isoformat()

    audit_summary = {
        "status": "PASS" if all_identical else "FAIL",
        "all_identical": all_identical,
        "total_prompts": len(demos) * len(PINNED_CONDITIONS),
        "repetitions_per_prompt": repetitions,
        "total_generations": total_generations,
        "model_name": PINNED_MODEL_NAME,
        "model_revision": PINNED_REVISION,
        "precision": "bfloat16",
        "cpu_threads": runner.threads,
        "seed": PINNED_SEED,
        "max_new_tokens": PINNED_MAX_NEW_TOKENS,
        "git_commit": git_commit,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "discrepancies": discrepancies,
        "prompts_data": results_by_prompt,
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(audit_summary, f, indent=2)
        print(f"\nDeterminism audit report saved to: {out_p}")

    print("\n" + "=" * 70)
    print(f"DETERMINISM CHECK COMPLETED: {'PASSED (100% Deterministic)' if all_identical else 'FAILED'}")
    print("=" * 70)

    if not all_identical:
        raise RuntimeError(
            f"Determinism Check FAILED! Discrepancies detected across runs: {discrepancies}"
        )

    return audit_summary


def run_pilot_inference(
    runner: ModelRunner,
    test_tasks_path: Path,
    demo_tasks_path: Path,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute full pilot model inference across all 3 conditions (§5):
    - 90 evaluation tasks (30 each for d in {1, 3, 5}).
    - 3 conditions: A, B, C.
    - 270 total model generations in greedy decoding mode (max_new_tokens=512, seed=42).
    - Records raw generations, token sequences, and SHA-256 hashes.
    """
    with open(demo_tasks_path, encoding="utf-8") as f:
        demo_payload = json.load(f)
    demos = demo_payload if isinstance(demo_payload, list) else demo_payload.get("tasks", [])

    with open(test_tasks_path, encoding="utf-8") as f:
        test_payload = json.load(f)
    tasks = test_payload if isinstance(test_payload, list) else test_payload.get("tasks", [])

    if len(tasks) != 90:
        raise ValueError(f"Expected 90 evaluation tasks per §3, found {len(tasks)}")

    git_commit = get_git_commit_hash()
    timestamp_start = datetime.now(timezone.utc).isoformat()

    # Check for existing checkpoint to resume from interruptions
    completed_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if output_path:
        out_p = Path(output_path)
        if out_p.exists():
            try:
                with open(out_p, "r", encoding="utf-8") as f:
                    existing_payload = json.load(f)
                for r in existing_payload.get("generations", []):
                    tid = r.get("task_id")
                    cond = r.get("condition")
                    if tid and cond and "output_sha256" in r:
                        completed_map[(tid, cond)] = r
                if completed_map:
                    print(f"Found checkpoint at {out_p}: Loaded {len(completed_map)} existing generations. Resuming...")
            except Exception as e:
                print(f"Warning: Could not load existing checkpoint from {out_p} ({e}). Starting fresh.")

    print("\n" + "=" * 70)
    print(f"STARTING OFFICIAL PILOT INFERENCE (§5): {len(tasks)} tasks × 3 conditions = {len(tasks) * 3} generations")
    print(f"Git Commit: {git_commit}")
    print("=" * 70)

    records = []
    total_generations = len(tasks) * len(PINNED_CONDITIONS)
    gen_idx = 0

    def save_checkpoint(final: bool = False) -> None:
        if not output_path:
            return
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "metadata": {
                "pilot": "H2 Spatial Tax Pilot",
                "git_commit": git_commit,
                "timestamp_start": timestamp_start,
                "timestamp_end": datetime.now(timezone.utc).isoformat() if final else None,
                "status": "COMPLETED" if final else "IN_PROGRESS",
                "model_name": PINNED_MODEL_NAME,
                "model_revision": PINNED_REVISION,
                "precision": "bfloat16",
                "cpu_threads": runner.threads,
                "seed": PINNED_SEED,
                "max_new_tokens": PINNED_MAX_NEW_TOKENS,
                "total_tasks": len(tasks),
                "total_generations": len(records),
                "conditions": PINNED_CONDITIONS,
            },
            "generations": records,
        }
        tmp_p = out_p.with_suffix(".tmp")
        with open(tmp_p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        tmp_p.replace(out_p)

    for task in tasks:
        task_id = task["task_id"]
        depth = task["depth"]

        for condition in PINNED_CONDITIONS:
            gen_idx += 1
            prompt_key = (task_id, condition)

            # Skip generation if already present in checkpoint
            if prompt_key in completed_map:
                record = completed_map[prompt_key]
                records.append(record)
                print(f"[{gen_idx:03d}/{total_generations:03d}] Task {task_id} (d={depth}, Cond {condition})... [RESUMED from checkpoint]")
                continue

            prompt_text = build_prompt(demos, task, condition)
            prompt_hash = sha256_text(prompt_text)

            print(f"[{gen_idx:03d}/{total_generations:03d}] Task {task_id} (d={depth}, Cond {condition})...", end="", flush=True)

            gen_result = runner.generate(prompt_text)

            record = {
                "task_id": task_id,
                "depth": depth,
                "condition": condition,
                "prompt_sha256": prompt_hash,
                "prompt_text": prompt_text,
                "output_sha256": gen_result["output_sha256"],
                "output_token_count": gen_result["output_token_count"],
                "output_tokens": gen_result["output_tokens"],
                "output_raw": gen_result["output_raw"],
                "generation_time_sec": gen_result["generation_time_sec"],
            }
            records.append(record)
            completed_map[prompt_key] = record

            print(f" -> {gen_result['output_token_count']} tokens ({gen_result['generation_time_sec']:.2f}s)")
            save_checkpoint(final=False)

    save_checkpoint(final=True)
    if output_path:
        print(f"\nAll {len(records)} raw generations saved to: {output_path}")

    return {
        "metadata": {
            "pilot": "H2 Spatial Tax Pilot",
            "git_commit": git_commit,
            "timestamp_start": timestamp_start,
            "timestamp_end": datetime.now(timezone.utc).isoformat(),
            "status": "COMPLETED",
            "model_name": PINNED_MODEL_NAME,
            "model_revision": PINNED_REVISION,
            "precision": "bfloat16",
            "cpu_threads": runner.threads,
            "seed": PINNED_SEED,
            "max_new_tokens": PINNED_MAX_NEW_TOKENS,
            "total_tasks": len(tasks),
            "total_generations": len(records),
            "conditions": PINNED_CONDITIONS,
        },
        "generations": records,
    }


def main():
    parser = argparse.ArgumentParser(description="Model Inference and Determinism Runner per §4 & §5.")
    parser.add_argument(
        "--determinism-check",
        action="store_true",
        help="Run pre-run determinism check on 3 demonstration tasks (9 prompts × 3 runs).",
    )
    parser.add_argument(
        "--run-pilot",
        action="store_true",
        help="Execute official pilot inference on test tasks across all 3 conditions (270 generations).",
    )
    parser.add_argument(
        "--test-tasks",
        type=str,
        default="tasks/pilot/test_tasks.json",
        help="Path to evaluation test tasks JSON file (default: tasks/pilot/test_tasks.json).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of repetitions per prompt in determinism check (default: 3).",
    )
    parser.add_argument(
        "--demos-path",
        type=str,
        default="tasks/pilot/demo_tasks.json",
        help="Path to demonstration tasks JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/pilot",
        help="Output directory for results and logs.",
    )
    args = parser.parse_args()

    demo_path = Path(args.demos_path)
    if not demo_path.is_absolute():
        demo_path = PROJECT_ROOT / demo_path

    runner = ModelRunner()

    if args.determinism_check:
        output_report = Path(args.output_dir) / "determinism_check.json"
        if not output_report.is_absolute():
            output_report = PROJECT_ROOT / output_report
        run_determinism_check(
            runner=runner,
            demo_tasks_path=demo_path,
            repetitions=args.repetitions,
            output_path=output_report,
        )
    elif args.run_pilot:
        test_path = Path(args.test_tasks)
        if not test_path.is_absolute():
            test_path = PROJECT_ROOT / test_path
        output_file = Path(args.output_dir) / "raw_generations.json"
        if not output_file.is_absolute():
            output_file = PROJECT_ROOT / output_file
        run_pilot_inference(
            runner=runner,
            test_tasks_path=test_path,
            demo_tasks_path=demo_path,
            output_path=output_file,
        )
    else:
        print("Model loaded successfully. Specify --determinism-check or --run-pilot to execute inference.")


if __name__ == "__main__":
    main()
