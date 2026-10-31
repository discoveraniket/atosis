#!/usr/bin/env python3
"""
Task Generator for H2 Spatial Tax Pilot (S1_prereg_pilot.md §3)

Generates 90 test tasks (30 per depth d=1,3,5) across 3 conditions (A, B, C)
plus 3 demonstration tasks (d=1,2,3) with a separate seed.

Parameters fixed per §3:
- Depths: d = 1, 3, 5
- Tasks per depth: 30
- Total test tasks: 90 (× 3 conditions = 270 model generations)
- Demonstrations: 3 tasks at d = 1, 2, 3, seed 9000 (excluded from test set)
- Unit tests per task: 3 fixed inputs: [], [1, 2, 3], [-1, 5, 0, 2]
- Task seed: 1234
- Demonstration seed: 9000
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any


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
# If level 1 is 'if', level 2 must be 'for v1 in data:'
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


def generate_spec_text(depth: int, headers: List[str], statements_per_block: List[List[str]]) -> str:
    """Generate the # comment specification block in words (no syntax cues)."""
    lines = ["# Specification:"]
    lines.append(f"# Function process_data(data) with nesting depth {depth}.")
    for i, (header, stmts) in enumerate(zip(headers, statements_per_block), 1):
        indent = "  " * i
        header_desc = header.replace(":", "").replace("{v}", "the loop variable")
        lines.append(f"# Level {i}: {header_desc}")
        for j, stmt in enumerate(stmts, 1):
            lines.append(f"#   Statement {j}: {stmt}")
        if i < depth:
            lines.append(f"#   Post-exit statement after level {i} block ends")
    lines.append("# Final: return total")
    return "\n".join(lines)


def build_canonical_python(depth: int, headers: List[str], 
                           pre_stmts: List[List[str]], 
                           post_stmts: List[List[str]],
                           constants: List[int]) -> str:
    """Build the canonical Python function (Condition A syntax)."""
    lines = ["def process_data(data):"]
    lines.append("    total = 0")
    const_idx = 0
    
    for level in range(depth):
        indent = "    " * (level + 1)
        lines.append(f"{indent}{headers[level]}")
        
        # Pre-child statements
        for stmt in pre_stmts[level]:
            const = constants[const_idx]
            const_idx += 1
            lines.append(f"{indent}    {stmt.format(const=const)}")
        
        # Post-exit statement (for non-innermost blocks)
        if level < depth - 1:
            const = constants[const_idx]
            const_idx += 1
            post_stmt = post_stmts[level][0].format(const=const)
            lines.append(f"{indent}    {post_stmt}")
    
    # Innermost block has no post-exit (handled above)
    # Return statement at outermost level
    lines.append("    return total")
    return "\n".join(lines)


def generate_task(task_id: str, depth: int, rng: random.Random) -> TaskSpec:
    """Generate a single task specification."""
    # Select headers for this depth
    headers = []
    pre_stmts = [[] for _ in range(depth)]
    post_stmts = [[] for _ in range(depth - 1)]
    constants = []
    
    # Level 1 header
    level1 = rng.choice(LEVEL1_HEADERS)
    headers.append(level1)
    
    # Deeper levels
    for level in range(1, depth):
        if level == 1 and "if" in level1:
            headers.append("for v1 in data:")
        else:
            # Choose from deeper headers, avoiding consecutive repeats
            available = DEEPER_HEADERS.copy()
            if level > 1:
                # Avoid consecutive repeats of same template type
                last = headers[-1]
                if "if" in last and "% 2 == 0" in last:
                    available = [h for h in available if not ("% 2 == 0" in h and "if" in h)]
                elif "if" in last and "> 0" in last:
                    available = [h for h in available if not ("> 0" in h and "if" in h)]
            header_template = rng.choice(available)
            # Fill in loop variable
            if "{v}" in header_template:
                # Use innermost active loop variable
                if "v1" in " ".join(headers):
                    header_template = header_template.replace("{v}", "v1")
                else:
                    header_template = header_template.replace("{v}", "v2")
            headers.append(header_template)
    
    # Generate statements per block
    stmts_per_block = STATEMENTS_PER_BLOCK[depth]
    const_pool = CONSTANT_POOL.copy()
    rng.shuffle(const_pool)
    const_iter = iter(const_pool)
    
    for level in range(depth):
        for _ in range(stmts_per_block):
            const = next(const_iter)
            constants.append(const)
            # Randomly choose statement type
            stmt_type = rng.choice(["+=", "-=", "*="])
            pre_stmts[level].append(f"total {stmt_type} {{const}}")
        
        # Post-exit statement for non-innermost
        if level < depth - 1:
            const = next(const_iter)
            constants.append(const)
            stmt_type = rng.choice(["+=", "-=", "*="])
            post_stmts[level].append(f"total {stmt_type} {{const}}")
    
    spec_text = generate_spec_text(depth, headers, pre_stmts)
    canonical = build_canonical_python(depth, headers, pre_stmts, post_stmts, constants)
    
    return TaskSpec(
        task_id=task_id,
        depth=depth,
        spec_text=spec_text,
        signature_a="def process_data(data):",
        signature_bc="def process_data(data) {",
        canonical_python=canonical,
        constants_used=constants,
        unit_test_inputs=UNIT_TEST_INPUTS
    )


def main():
    output_dir = Path("tasks")
    output_dir.mkdir(exist_ok=True)
    
    # Generate test tasks
    rng = random.Random(TASK_SEED)
    test_tasks = []
    
    for depth in DEPTHS:
        for i in range(TASKS_PER_DEPTH):
            task_id = f"test_d{depth}_{i:03d}"
            task = generate_task(task_id, depth, rng)
            test_tasks.append(asdict(task))
    
    # Generate demonstration tasks
    demo_rng = random.Random(DEMO_SEED)
    demo_tasks = []
    
    for i, depth in enumerate(DEMO_DEPTHS):
        task_id = f"demo_d{depth}_{i:03d}"
        task = generate_task(task_id, depth, demo_rng)
        demo_tasks.append(asdict(task))
    
    # Save
    with open(output_dir / "test_tasks.json", "w") as f:
        json.dump(test_tasks, f, indent=2)
    
    with open(output_dir / "demo_tasks.json", "w") as f:
        json.dump(demo_tasks, f, indent=2)
    
    print(f"Generated {len(test_tasks)} test tasks and {len(demo_tasks)} demo tasks")
    print(f"Saved to {output_dir}/")


if __name__ == "__main__":
    main()