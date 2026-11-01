#!/usr/bin/env python3
"""
Output Extraction and Test Evaluation Script (S1_prereg_pilot.md §6).

Implements extraction and evaluation pipeline:
1. Special tokens cut: cuts at first '<|...|>' token (<|endoftext|>, <|im_end|>).
2. Condition-specific scope extraction:
   - Condition A: cut at first non-empty line with zero indentation, or EOS.
   - Conditions B & C: depth starts at 1 (from prompt's '{'); cut after line returning depth to 0.
3. Truncation detection: token_count >= 512 and scope unclosed.
4. Overrun recording: flags any non-whitespace text discarded after the cut.
5. Structural scoring: Category 1 (Unparsable), Category 2 (Skeleton Error), Category 3 (Structurally Correct).
6. Unit testing: Pass@1 cross-check on unit test inputs ([], [1, 2, 3], [-1, 5, 0, 2]).
7. Content error auditing: detailed operator, constant, predicate checks on Category 3 generations.
"""

import ast
import json
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Support direct script and module execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spatial_tax.converter import (
    convert_brace_to_python,
    strip_trailing_comment,
    classify_line,
    ConversionError,
)
from src.spatial_tax.scorer import (
    score_generation,
    Category,
    ParseErrorCode,
    SkeletonErrorCode,
    ScoreResult,
)
from src.spatial_tax.mutants import execute_function


SPECIAL_TOKEN_REGEX = re.compile(r"<\|.*?\|>")


@dataclass
class ExtractionResult:
    """Detailed output of the §6 extraction process."""
    raw_text: str
    extracted_body: str
    full_function_code: str
    is_truncated: bool
    has_overrun: bool
    overrun_text: str
    special_tokens_seen: List[str]
    cut_reason: str
    depth_at_cut: int


@dataclass
class EvaluationRecord:
    """Complete evaluation record for a single model generation per §5 & §6."""
    task_id: str
    depth: int
    condition: str
    prompt_sha256: str
    output_sha256: str
    raw_token_count: int
    extraction: Dict[str, Any]
    category: int  # 1, 2, or 3
    category_name: str
    parse_error: Optional[str] = None
    skeleton_error: Optional[str] = None
    colon_brace_flag: bool = False
    pass_at_1: bool = False
    content_errors: List[str] = field(default_factory=list)
    unit_test_outputs: List[Any] = field(default_factory=list)
    reference_outputs: List[Any] = field(default_factory=list)


def extract_generation(
    raw_output: str,
    condition: str,
    token_count: int,
    max_tokens: int = 512,
) -> ExtractionResult:
    """
    Apply Section 6 extraction rules:
    1. Cut at first special token (<|...|>).
    2. Condition A: Cut at first non-empty line with zero indentation.
    3. Condition B/C: Depth starts at 1; cut after line returning depth to 0.
    4. Truncation: token_count >= max_tokens without scope closure.
    5. Overrun: record non-whitespace text after the cut.
    """
    special_tokens_seen = SPECIAL_TOKEN_REGEX.findall(raw_output)

    # 1. Cut at first special token
    match = SPECIAL_TOKEN_REGEX.search(raw_output)
    if match:
        text_before_special = raw_output[:match.start()]
        text_after_special = raw_output[match.end():]
        special_cut_reason = f"special_token:{match.group(0)}"
    else:
        text_before_special = raw_output
        text_after_special = ""
        special_cut_reason = "eos_or_length"

    extracted_lines = []
    discarded_lines = []
    scope_closed = False
    cut_reason = special_cut_reason
    depth_at_cut = 0

    lines = text_before_special.splitlines(keepends=True)

    if condition == "A":
        # Condition A: Cut at first non-empty line with 0 indentation
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                extracted_lines.append(line)
                continue

            indent = len(line) - len(line.lstrip())
            if indent == 0:
                # Reached scope end (or extraneous outer statement)
                discarded_lines.extend(lines[idx:])
                scope_closed = True
                cut_reason = "condition_a_zero_indent"
                break
            else:
                extracted_lines.append(line)

        # Condition A reassembled function
        body_text = "".join(extracted_lines)
        full_code = f"def process_data(data):\n{body_text}".strip()

    elif condition in ("B", "C"):
        # Conditions B & C: Depth starts at 1
        current_depth = 1
        for idx, line in enumerate(lines):
            line_no_comment = strip_trailing_comment(line)
            line_type, content = classify_line(line_no_comment)

            extracted_lines.append(line)

            if line_type == "open":
                current_depth += 1
            elif line_type == "close":
                current_depth -= 1
                if current_depth <= 0:
                    current_depth = 0
                    scope_closed = True
                    discarded_lines.extend(lines[idx + 1:])
                    cut_reason = "condition_bc_depth_zero"
                    break

        depth_at_cut = current_depth
        body_text = "".join(extracted_lines)
        full_code = f"def process_data(data) {{\n{body_text}".strip()

    else:
        raise ValueError(f"Unknown condition: {condition}")

    # Check truncation (§6: token_count >= 512 and scope was never closed)
    is_truncated = (token_count >= max_tokens and not scope_closed)

    # Discarded text after cut
    all_discarded = "".join(discarded_lines) + text_after_special
    has_overrun = bool(all_discarded.strip())
    overrun_text = all_discarded if has_overrun else ""

    return ExtractionResult(
        raw_text=raw_output,
        extracted_body=body_text,
        full_function_code=full_code,
        is_truncated=is_truncated,
        has_overrun=has_overrun,
        overrun_text=overrun_text,
        special_tokens_seen=special_tokens_seen,
        cut_reason=cut_reason,
        depth_at_cut=depth_at_cut,
    )


def extract_content_elements(node: ast.AST) -> Dict[str, Any]:
    """Extract constants, operators, and variable identifiers from an AST statement."""
    elements = {}
    if isinstance(node, ast.AugAssign):
        elements["op"] = type(node.op).__name__
        if isinstance(node.value, ast.Constant):
            elements["const"] = node.value.value
    elif isinstance(node, ast.Assign):
        if isinstance(node.value, ast.Constant):
            elements["const"] = node.value.value
    elif isinstance(node, (ast.If, ast.For)):
        elements["type"] = "if" if isinstance(node, ast.If) else "for"
    return elements


def find_content_errors(gen_code: str, ref_code: str) -> List[str]:
    """
    Examine Category 3 generations for content differences per §6:
    operator, constant, predicate, or iterable differs from reference at matching nodes.
    """
    errors = []
    try:
        gen_tree = ast.parse(gen_code)
        ref_tree = ast.parse(ref_code)

        def walk_statements(body_list: List[ast.AST]) -> List[ast.AST]:
            stmts = []
            for item in body_list:
                if isinstance(item, (ast.If, ast.For)):
                    stmts.append(item)
                    stmts.extend(walk_statements(item.body))
                elif isinstance(item, (ast.Assign, ast.AugAssign)):
                    stmts.append(item)
            return stmts

        gen_stmts = walk_statements(gen_tree.body[0].body)
        ref_stmts = walk_statements(ref_tree.body[0].body)

        if len(gen_stmts) != len(ref_stmts):
            errors.append(f"Statement count mismatch: {len(gen_stmts)} vs {len(ref_stmts)}")
            return errors

        for idx, (g_node, r_node) in enumerate(zip(gen_stmts, ref_stmts), 1):
            g_elem = extract_content_elements(g_node)
            r_elem = extract_content_elements(r_node)

            if g_elem != r_elem:
                errors.append(f"Node {idx} mismatch: expected {r_elem}, got {g_elem}")

    except Exception as e:
        errors.append(f"Content audit error: {e}")

    return errors


def evaluate_single_generation(
    raw_output: str,
    reference_python: str,
    condition: str,
    token_count: int,
    unit_test_inputs: List[List[int]],
    task_id: str = "",
    depth: int = 0,
    prompt_sha256: str = "",
    output_sha256: str = "",
) -> EvaluationRecord:
    """
    End-to-end evaluation of a raw generated output against reference task per §6:
    1. Extract output.
    2. Score structural category and sub-codes.
    3. Execute unit tests for Pass@1.
    4. Audit content errors if Category 3.
    """
    # 1. Extraction
    extraction = extract_generation(
        raw_output=raw_output,
        condition=condition,
        token_count=token_count,
        max_tokens=512,
    )

    # 2. Structural scoring
    score_result: ScoreResult = score_generation(
        generated_text=extraction.full_function_code,
        reference_python=reference_python,
        condition=condition,
        is_truncated=extraction.is_truncated,
    )

    # Get converted Python code for execution
    if condition in ("B", "C"):
        conv = convert_brace_to_python(extraction.full_function_code)
        python_for_eval = conv.python_code
        colon_brace_flag = conv.colon_brace_flag
    else:
        python_for_eval = extraction.full_function_code
        colon_brace_flag = False

    # 3. Unit test execution (Pass@1)
    ref_outputs = execute_function(reference_python, unit_test_inputs)
    gen_outputs = execute_function(python_for_eval, unit_test_inputs)
    pass_at_1 = (gen_outputs == ref_outputs)

    # 4. Content errors (Category 3 only)
    content_errors = []
    if score_result.category == Category.STRUCTURALLY_CORRECT:
        content_errors = find_content_errors(python_for_eval, reference_python)

    return EvaluationRecord(
        task_id=task_id,
        depth=depth,
        condition=condition,
        prompt_sha256=prompt_sha256,
        output_sha256=output_sha256,
        raw_token_count=token_count,
        extraction=asdict(extraction),
        category=score_result.category.value,
        category_name=score_result.category.name,
        parse_error=score_result.parse_error.value if score_result.parse_error else None,
        skeleton_error=score_result.skeleton_error.value if score_result.skeleton_error else None,
        colon_brace_flag=colon_brace_flag,
        pass_at_1=pass_at_1,
        content_errors=content_errors,
        unit_test_outputs=gen_outputs,
        reference_outputs=ref_outputs,
    )


def evaluate_generations_file(
    input_file: Path,
    tasks_file: Path,
    output_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Process an entire inference runs file and generate evaluation records per cell.
    """
    with open(input_file, encoding="utf-8") as f:
        infer_data = json.load(f)

    with open(tasks_file, encoding="utf-8") as f:
        tasks_data = json.load(f)

    if isinstance(tasks_data, dict) and "tasks" in tasks_data:
        task_list = tasks_data["tasks"]
    elif isinstance(tasks_data, list):
        task_list = tasks_data
    else:
        raise ValueError(
            f"Unexpected format for tasks file {tasks_file}: expected list or dict with 'tasks' key."
        )

    tasks_by_id = {t["task_id"]: t for t in task_list}

    eval_records = []
    generations = infer_data.get("generations", [])

    for gen in generations:
        task_id = gen["task_id"]
        task = tasks_by_id[task_id]

        record = evaluate_single_generation(
            raw_output=gen["output_raw"],
            reference_python=task["canonical_python"],
            condition=gen["condition"],
            token_count=gen.get("output_token_count", 0),
            unit_test_inputs=task.get("unit_test_inputs", [[], [1, 2, 3], [-1, 5, 0, 2]]),
            task_id=task_id,
            depth=gen.get("depth", task.get("depth", 0)),
            prompt_sha256=gen.get("prompt_sha256", ""),
            output_sha256=gen.get("output_sha256", ""),
        )
        eval_records.append(asdict(record))

    summary = {
        "total_evaluated": len(eval_records),
        "records": eval_records,
    }

    if output_file:
        out_p = Path(output_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Evaluated {len(eval_records)} generations. Saved to {out_p}")

    return summary


def main():
    """CLI for standalone evaluation of raw generation files."""
    if len(sys.argv) < 3:
        print("Usage: python scripts/pilot/extract_and_evaluate.py <generations.json> <tasks.json> [output.json]")
        sys.exit(1)

    gen_path = Path(sys.argv[1])
    tasks_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    evaluate_generations_file(gen_path, tasks_path, out_path)


if __name__ == "__main__":
    main()
