#!/usr/bin/env python3
"""
Task Generator with Single-Move Scope Mutant Filter (S1_prereg_pilot.md §3, §8)

Generates test and demonstration tasks across conditions (A, B, C).
Enforces structural rules and the single-move scope mutant catch-rate filter:
- Depths: d in {1, 3, 5} for test; d in {1, 2, 3} for demonstrations
- Tasks per depth: 30
- Demonstrations: 3 tasks at d = 1, 2, 3, seed 9000 (excluded from test set)
- Unit tests per task: 3 fixed inputs: [], [1, 2, 3], [-1, 5, 0, 2]
- Post-exit statements after child blocks
- Unique constants 1-99 per task
- Mutant check: >= 80% of single-move scope mutants must fail unit tests.
  Failed tasks are regenerated under the same rule and regeneration counts recorded.
"""

import json
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

# Enable direct script execution
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.spatial_tax.mutants import check_task_mutants, MutantCheckResult


@dataclass
class TaskSpec:
    """Specification for a single task (condition-agnostic)."""
    task_id: str
    depth: int
    spec_text: str  # The # comment specification block
    signature_a: str  # def process_data(data):
    signature_bc: str  # def process_data(data) {
    canonical_python: str  # Full Python function (Condition A syntax)
    constants_used: List[int]  # Unique constants from pool 1-99
    unit_test_inputs: List[List[int]]
    regeneration_count: int = 0
    mutant_catch_rate: float = 1.0
    total_mutants: int = 0
    caught_mutants: int = 0


# §3 Fixed parameters
DEPTHS = [1, 3, 5]
TASKS_PER_DEPTH = 30
TASK_SEED = 1234
DEMO_SEED = 9000
DEMO_DEPTHS = [1, 2, 3]
UNIT_TEST_INPUTS = [[], [1, 2, 3], [-1, 5, 0, 2]]
CONSTANT_POOL = list(range(1, 100))  # 1-99 unique integers

# Block header templates per §3
LEVEL1_HEADERS = [
    "if len(data) > 0:",
    "for v1 in data:",
]
DEEPER_HEADERS = [
    "if {v} > 0:",
    "if {v} % 2 == 0:",
    "if total % 2 == 0:",
    "for v2 in data:",
]

# Statements per block per depth (pre-child updates)
# d=1: 3, d=2: 2, d=3: 2, d=5: 1
# Every non-innermost block also has 1 post-exit update statement
STATEMENTS_PER_BLOCK = {1: 3, 2: 2, 3: 2, 5: 1}


def describe_op_in_words(op: str, const: int) -> str:
    """Describe arithmetic update in words with no syntax cues or quotes."""
    if op == "+=":
        return f"add {const} to total"
    elif op == "-=":
        return f"subtract {const} from total"
    elif op == "*=":
        return f"multiply total by {const}"
    raise ValueError(f"Unknown operator: {op}")


def generate_spec_text(
    depth: int,
    headers: List[str],
    pre_ops: List[List[Tuple[str, int]]],
    post_ops: List[List[Tuple[str, int]]],
) -> str:
    """Generate the # comment specification block in words (no syntax cues, braces, or quotes)."""
    lines = ["# Specification:"]
    lines.append(f"# Function process_data(data) with nesting depth {depth}.")

    def emit_spec(level: int) -> None:
        indent = "  " * (level + 1)
        header_desc = headers[level].rstrip(":").replace("{v}", "the loop variable")
        lines.append(f"# {indent}Level {level + 1}: {header_desc}")
        for j, (op, const) in enumerate(pre_ops[level], 1):
            desc = describe_op_in_words(op, const)
            lines.append(f"# {indent}  Statement {j}: {desc}")

        if level < depth - 1:
            emit_spec(level + 1)
            op, const = post_ops[level][0]
            desc = describe_op_in_words(op, const)
            lines.append(f"# {indent}  Post-exit statement after level {level + 2} block ends: {desc}")

    emit_spec(0)
    lines.append("# Final: return total")
    return "\n".join(lines)


def build_canonical_python(
    depth: int,
    headers: List[str],
    pre_ops: List[List[Tuple[str, int]]],
    post_ops: List[List[Tuple[str, int]]],
) -> str:
    """
    Build the canonical Python function (Condition A syntax).
    Correctly nests child blocks and places post-exit statements after child blocks close.
    """
    lines = ["def process_data(data):"]
    lines.append("    total = 0")

    def emit_block(level: int) -> None:
        indent = "    " * (level + 1)
        body_indent = "    " * (level + 2)
        lines.append(f"{indent}{headers[level]}")

        # Pre-child statements
        for op, const in pre_ops[level]:
            lines.append(f"{body_indent}total {op} {const}")

        # If non-innermost, recursively emit child block and then post-exit statement
        if level < depth - 1:
            emit_block(level + 1)
            op, const = post_ops[level][0]
            lines.append(f"{body_indent}total {op} {const}")

    emit_block(0)
    lines.append("    return total")
    return "\n".join(lines)


def generate_candidate_task(task_id: str, depth: int, rng: random.Random) -> TaskSpec:
    """Generate a single raw candidate task before the mutant filter."""
    headers = []
    pre_ops: List[List[Tuple[str, int]]] = [[] for _ in range(depth)]
    post_ops: List[List[Tuple[str, int]]] = [[] for _ in range(depth - 1)]
    constants: List[int] = []

    # Level 1 header
    level1 = rng.choice(LEVEL1_HEADERS)
    headers.append(level1)

    # Deeper levels
    for level in range(1, depth):
        if level == 1 and "if" in level1:
            headers.append("for v1 in data:")
        else:
            available = DEEPER_HEADERS.copy()

            # Enforce §3: (at most once more per task) for v2 in data:
            if any("for v2 in data:" in h for h in headers):
                available = [h for h in available if h != "for v2 in data:"]

            # Enforce §3: draw without consecutive repeats
            if level > 0:
                last = headers[-1]
                if "total % 2 == 0" in last:
                    available = [h for h in available if h != "if total % 2 == 0:"]
                elif "% 2 == 0" in last:
                    available = [h for h in available if h != "if {v} % 2 == 0:"]
                elif "> 0" in last and "len" not in last:
                    available = [h for h in available if h != "if {v} > 0:"]
                elif "for v2 in data:" in last:
                    available = [h for h in available if h != "for v2 in data:"]

            header_template = rng.choice(available)

            # Enforce §3: <v> is the innermost active loop variable
            if "{v}" in header_template:
                if any("for v2 in data:" in h for h in headers):
                    active_v = "v2"
                else:
                    active_v = "v1"
                header_template = header_template.replace("{v}", active_v)

            headers.append(header_template)

    # Generate statements per block with unique constants
    stmts_per_block = STATEMENTS_PER_BLOCK[depth]
    const_pool = CONSTANT_POOL.copy()
    rng.shuffle(const_pool)
    const_iter = iter(const_pool)

    for level in range(depth):
        for _ in range(stmts_per_block):
            const = next(const_iter)
            constants.append(const)
            op = rng.choice(["+=", "-=", "*="])
            pre_ops[level].append((op, const))

        # Post-exit statement for non-innermost
        if level < depth - 1:
            const = next(const_iter)
            constants.append(const)
            op = rng.choice(["+=", "-=", "*="])
            post_ops[level].append((op, const))

    spec_text = generate_spec_text(depth, headers, pre_ops, post_ops)
    canonical = build_canonical_python(depth, headers, pre_ops, post_ops)

    return TaskSpec(
        task_id=task_id,
        depth=depth,
        spec_text=spec_text,
        signature_a="def process_data(data):",
        signature_bc="def process_data(data) {",
        canonical_python=canonical,
        constants_used=constants,
        unit_test_inputs=UNIT_TEST_INPUTS,
    )


def generate_task(
    task_id: str,
    depth: int,
    rng: random.Random,
    min_catch_rate: float = 0.80,
    max_retries: int = 100,
) -> TaskSpec:
    """
    Generate a task and apply the single-move scope mutant catch rate filter (§3, §8).
    If catch rate < min_catch_rate, regenerate under the same rule and record the count.
    """
    regeneration_count = 0
    while True:
        task = generate_candidate_task(task_id, depth, rng)
        check_result = check_task_mutants(
            canonical_python=task.canonical_python,
            unit_test_inputs=task.unit_test_inputs,
            task_id=task_id,
            min_catch_rate=min_catch_rate,
        )

        if check_result.passed:
            task.regeneration_count = regeneration_count
            task.mutant_catch_rate = check_result.catch_rate
            task.total_mutants = check_result.total_mutants
            task.caught_mutants = check_result.caught_mutants
            return task

        regeneration_count += 1
        if regeneration_count > max_retries:
            raise RuntimeError(
                f"Task {task_id} (depth {depth}) exceeded maximum regeneration attempts ({max_retries}). "
                f"Last catch rate was {check_result.catch_rate:.1%}."
            )


def main():
    """Dry-run generation for demonstration tasks (seed 9000) only."""
    output_dir = Path("tasks/pilot")
    output_dir.mkdir(parents=True, exist_ok=True)

    demo_rng = random.Random(DEMO_SEED)
    demo_tasks = []

    for i, depth in enumerate(DEMO_DEPTHS):
        task_id = f"demo_d{depth}_{i:03d}"
        task = generate_task(task_id, depth, demo_rng)
        demo_tasks.append(asdict(task))

    demo_path = output_dir / "demo_tasks.json"
    with open(demo_path, "w") as f:
        json.dump(demo_tasks, f, indent=2)

    print(f"Generated {len(demo_tasks)} demonstration tasks with mutant check passed.")
    for dt in demo_tasks:
        print(f"  {dt['task_id']} (depth {dt['depth']}): catch rate = {dt['mutant_catch_rate']:.1%}, retries = {dt['regeneration_count']}")


if __name__ == "__main__":
    main()
