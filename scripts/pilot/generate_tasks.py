#!/usr/bin/env python3
"""
Official Test Task Generator (S1_prereg_pilot.md §3, §12).

Generates the 90 official evaluation tasks (30 tasks per depth for d in {1, 3, 5})
under pre-registered seed 1234, applying the single-move scope mutant catch-rate filter:
- Each task enforced to achieve >= 80% catch rate on single-move scope mutants
  against the 3 fixed unit test inputs ([], [1, 2, 3], [-1, 5, 0, 2]).
- Tasks failing the mutant filter are regenerated under the same rule.
- Regeneration counts and mutant statistics are recorded per task.
- Outputs are saved to tasks/pilot/test_tasks.json.
"""

import argparse
import json
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

# Support direct script and module execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spatial_tax.generator import (
    generate_task,
    DEPTHS,
    TASKS_PER_DEPTH,
    TASK_SEED,
    DEMO_SEED,
    DEMO_DEPTHS,
    TaskSpec,
)


def generate_official_test_tasks(
    seed: int = TASK_SEED,
    depths: List[int] = DEPTHS,
    tasks_per_depth: int = TASKS_PER_DEPTH,
    min_catch_rate: float = 0.80,
) -> List[Dict[str, Any]]:
    """
    Generate official test tasks with pre-registered parameters (§3).
    Seed is initialized once to produce an ordered sequence across all depths.
    """
    rng = random.Random(seed)
    all_tasks = []

    print(f"Initializing official task generation with seed={seed}...")
    print(f"Target depths: {depths}, Tasks per depth: {tasks_per_depth} (Total: {len(depths) * tasks_per_depth})")
    print(f"Mutant catch-rate threshold: {min_catch_rate:.0%}\n")

    for depth in depths:
        print(f"--- Generating Depth {depth} ({tasks_per_depth} tasks) ---")
        for i in range(tasks_per_depth):
            task_id = f"task_d{depth}_{i:03d}"
            task: TaskSpec = generate_task(
                task_id=task_id,
                depth=depth,
                rng=rng,
                min_catch_rate=min_catch_rate,
            )
            task_dict = asdict(task)
            all_tasks.append(task_dict)
            print(
                f"  [{task_id}] mutant catch rate: {task.mutant_catch_rate:.1%} "
                f"({task.caught_mutants}/{task.total_mutants}), retries: {task.regeneration_count}"
            )

    return all_tasks


def main():
    parser = argparse.ArgumentParser(description="Official Test Task Generator per §3 & §12.")
    parser.add_argument(
        "--seed",
        type=int,
        default=TASK_SEED,
        help=f"Master random seed for task generation (default: {TASK_SEED} per §3).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="tasks/pilot/test_tasks.json",
        help="Path to output test tasks JSON file (default: tasks/pilot/test_tasks.json).",
    )
    parser.add_argument(
        "--min-catch-rate",
        type=float,
        default=0.80,
        help="Minimum mutant catch-rate threshold (default: 0.80 per §3).",
    )
    args = parser.parse_args()

    out_path = Path(args.output_file)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = generate_official_test_tasks(
        seed=args.seed,
        depths=DEPTHS,
        tasks_per_depth=TASKS_PER_DEPTH,
        min_catch_rate=args.min_catch_rate,
    )

    # Compute overall mutant summary
    total_mutants = sum(t["total_mutants"] for t in tasks)
    caught_mutants = sum(t["caught_mutants"] for t in tasks)
    total_retries = sum(t["regeneration_count"] for t in tasks)
    overall_catch_rate = caught_mutants / max(total_mutants, 1)

    payload = {
        "metadata": {
            "pilot": "H2 Spatial Tax Pilot",
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "depths": DEPTHS,
            "tasks_per_depth": TASKS_PER_DEPTH,
            "total_tasks": len(tasks),
            "min_mutant_catch_rate": args.min_catch_rate,
            "overall_mutant_catch_rate": round(overall_catch_rate, 4),
            "total_mutants_evaluated": total_mutants,
            "total_mutants_caught": caught_mutants,
            "total_task_regenerations": total_retries,
        },
        "tasks": tasks,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 70)
    print(f"OFFICIAL TASK GENERATION COMPLETED: {len(tasks)} tasks saved to {out_path}")
    print(f"Overall Mutant Catch Rate: {overall_catch_rate:.2%} ({caught_mutants}/{total_mutants})")
    print(f"Total Regenerations Needed: {total_retries}")
    print("=" * 70)


if __name__ == "__main__":
    main()

