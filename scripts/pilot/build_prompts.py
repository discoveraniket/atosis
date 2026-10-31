#!/usr/bin/env python3
"""
Prompt Builder & Demonstration Format Validator (S1_prereg_pilot.md §5).

Constructs raw completion prompts for Conditions A, B, and C.
Enforces the exact layout and few-shot format consistency rules from §5:
1. Demonstrations: in order d = 1, 2, 3, separated by a blank line ('\\n\\n').
   Each demonstration is a # comment specification block followed by a complete function.
2. Target task: # comment specification block followed by condition signature:
   - Condition A: 'def process_data(data):'
   - Condition B & C: 'def process_data(data) {'
3. Prompt ends with exactly one newline ('\\n') after the signature line.
4. Format validator: Verifies every block-opening line in B/C ends with '{'
   and contains no Python ':' headers before inference.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Enable execution directly and as module
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.spatial_tax.converter import (
    python_to_condition_b,
    python_to_condition_c,
)


SIGNATURE_A = "def process_data(data):"
SIGNATURE_BC = "def process_data(data) {"


class FormatValidationError(ValueError):
    """Raised when demonstration or prompt violates §5 format rules."""
    pass


def validate_demonstration_format(demo_code: str, condition: str, demo_id: str = "") -> None:
    """
    Validate few-shot format consistency per §5:
    - Condition A: Standard Python syntax, colons on block headers, no braces.
    - Condition B & C: Every block-opening line must end with '{'.
      No Python ':' headers allowed on block-opening lines.
      Braces must be balanced.
      Condition C must have zero indentation.
    """
    lines = demo_code.splitlines()
    open_braces = 0
    close_braces = 0

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if condition == "A":
            if "{" in stripped or "}" in stripped:
                raise FormatValidationError(
                    f"[{demo_id}] Condition A demonstration contains brace on line {line_num}: {line}"
                )
            if stripped.startswith(("def ", "if ", "for ")):
                if not stripped.endswith(":"):
                    raise FormatValidationError(
                        f"[{demo_id}] Condition A block-opening line must end with ':' on line {line_num}: {line}"
                    )

        elif condition in ("B", "C"):
            # Check for Python colon before '{' (e.g. 'if len(data) > 0: {')
            if stripped.endswith("{"):
                header = stripped[:-1].rstrip()
                if header.endswith(":"):
                    raise FormatValidationError(
                        f"[{demo_id}] Condition {condition} contains illegal ':' before '{{' on line {line_num}: {line}"
                    )
                open_braces += 1
            elif stripped == "}":
                close_braces += 1
            elif stripped.startswith(("if ", "for ")):
                # Block header missing '{'
                raise FormatValidationError(
                    f"[{demo_id}] Condition {condition} block header does not end with '{{' on line {line_num}: {line}"
                )

            # Condition C must have no indentation
            if condition == "C" and line != line.lstrip():
                raise FormatValidationError(
                    f"[{demo_id}] Condition C must have zero indentation, found leading spaces on line {line_num}: {line}"
                )

    if condition in ("B", "C"):
        if open_braces != close_braces:
            raise FormatValidationError(
                f"[{demo_id}] Condition {condition} has unbalanced braces: {open_braces} open vs {close_braces} close"
            )


def format_demonstration_code(canonical_python: str, condition: str) -> str:
    """Convert canonical Python to the specified condition's complete function code."""
    if condition == "A":
        return canonical_python
    elif condition == "B":
        return python_to_condition_b(canonical_python)
    elif condition == "C":
        return python_to_condition_c(canonical_python)
    else:
        raise ValueError(f"Unknown condition: {condition}. Must be 'A', 'B', or 'C'.")


def build_demonstrations_block(demonstrations: List[Dict[str, Any]], condition: str) -> str:
    """
    Build the demonstrations block from 3 demonstrations (d=1, 2, 3) per §5.
    Separated by a blank line ('\\n\\n').
    Each demonstration is: # comment specification block followed by complete function.
    """
    if len(demonstrations) != 3:
        raise FormatValidationError(
            f"Expected exactly 3 demonstrations per §5, got {len(demonstrations)}"
        )

    # Sort strictly by depth: d = 1, 2, 3
    sorted_demos = sorted(demonstrations, key=lambda d: d.get("depth", 0))
    expected_depths = [1, 2, 3]
    actual_depths = [d.get("depth", 0) for d in sorted_demos]
    if actual_depths != expected_depths:
        raise FormatValidationError(
            f"Demonstrations must have depths {expected_depths}, got {actual_depths}"
        )

    demo_sections = []
    for demo in sorted_demos:
        demo_id = demo.get("task_id", f"demo_d{demo.get('depth')}")
        spec_text = demo["spec_text"].strip()
        code = format_demonstration_code(demo["canonical_python"], condition)
        
        # Validate demonstration syntax
        validate_demonstration_format(code, condition, demo_id)

        demo_sections.append(f"{spec_text}\n{code}")

    return "\n\n".join(demo_sections)


def get_condition_signature(condition: str) -> str:
    """Return the condition-specific function signature line."""
    if condition == "A":
        return SIGNATURE_A
    elif condition in ("B", "C"):
        return SIGNATURE_BC
    raise ValueError(f"Unknown condition: {condition}")


def build_prompt(demonstrations: List[Dict[str, Any]], task: Dict[str, Any], condition: str) -> str:
    """
    Construct the raw completion prompt for a given task and condition per §5.
    
    Layout:
    1. 3 demonstrations (d=1, 2, 3), separated by '\\n\\n'.
    2. '\\n\\n'
    3. Target task specification block ('# Specification:...').
    4. '\\n'
    5. Signature line ('def process_data(data):' or 'def process_data(data) {').
    6. Trailing '\\n' (prompt ends with one newline after signature).
    """
    demo_block = build_demonstrations_block(demonstrations, condition)
    task_spec = task["spec_text"].strip()
    signature = get_condition_signature(condition)

    # Prompt layout strictly per §5:
    prompt = f"{demo_block}\n\n{task_spec}\n{signature}\n"
    return prompt


def main():
    """CLI / Dry-run validation of demonstration prompts across Conditions A, B, and C."""
    demo_path = Path("tasks/pilot/demo_tasks.json")
    if not demo_path.exists():
        demo_path = Path("tasks/demo_tasks.json")

    if not demo_path.exists():
        print(f"Error: Demonstrations not found at {demo_path}")
        sys.exit(1)

    with open(demo_path) as f:
        demos = json.load(f)

    print(f"Loaded {len(demos)} demonstration tasks from {demo_path}")

    # Validate each condition
    conditions = ["A", "B", "C"]
    for cond in conditions:
        print(f"\n--- Validating Condition {cond} Demonstration Block ---")
        try:
            demo_block = build_demonstrations_block(demos, cond)
            print(f"PASS: Condition {cond} demonstrations validated successfully.")
            print(f"Total lines: {len(demo_block.splitlines())}, Characters: {len(demo_block)}")
        except FormatValidationError as e:
            print(f"FAIL: Condition {cond} format validation error: {e}")
            sys.exit(1)

    # Test prompt building for demo task 1 as target
    print("\n--- Sample Prompt for Condition B (Target: demo_d1_000) ---")
    sample_prompt = build_prompt(demos, demos[0], "B")
    print("=" * 60)
    print(sample_prompt)
    print("=" * 60)
    print(f"Prompt ends with newline: {sample_prompt.endswith(chr(10))}")
    print("All format checks PASSED.")


if __name__ == "__main__":
    main()
