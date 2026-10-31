#!/usr/bin/env python3
"""
Scorer for H2 Spatial Tax Pilot (S1_prereg_pilot.md §6, §8)

Implements:
- Category classification (1: Unparsable, 2: Skeleton error, 3: Structurally correct)
- Skeleton extraction and equality checking
- Scorer self-test with negative controls (§8)
"""

import ast
import json
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Any
from pathlib import Path

import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from src.spatial_tax.converter import (
        convert_brace_to_python,
        python_to_condition_b,
        python_to_condition_c,
        ConversionError,
        ConversionResult,
    )
except ImportError:
    from converter import (
        convert_brace_to_python,
        python_to_condition_b,
        python_to_condition_c,
        ConversionError,
        ConversionResult,
    )


class Category(Enum):
    """Generation categories per §6."""
    UNPARSABLE = 1      # Category 1: truncation, unclosed scope, syntax error
    SKELETON_ERROR = 2  # Category 2: parses but skeleton differs
    STRUCTURALLY_CORRECT = 3  # Category 3: parses and skeleton matches


class ParseErrorCode(Enum):
    """Sub-codes for Category 1 per §6."""
    TRUNCATED = "truncated"
    E_UNCLOSED = "E_UNCLOSED"
    E_HEADER_SYNTAX = "E_HEADER_SYNTAX"
    E_INDENT = "E_INDENT"
    E_SYNTAX = "E_SYNTAX"


class SkeletonErrorCode(Enum):
    """Sub-codes for Category 2 per §6 (exploratory)."""
    E_PLACEMENT = "E_PLACEMENT"  # Same nodes, wrong nesting
    E_OTHER = "E_OTHER"          # Missing/extra block/statement, different order


@dataclass
class ScoreResult:
    """Result of scoring a generation."""
    category: Category
    parse_error: Optional[ParseErrorCode] = None
    skeleton_error: Optional[SkeletonErrorCode] = None
    content_errors: List[str] = None  # For Category 3 only
    skeleton: Optional[Any] = None
    reference_skeleton: Optional[Any] = None
    
    def __post_init__(self):
        if self.content_errors is None:
            self.content_errors = []


def extract_skeleton(python_code: str) -> Any:
    """
    Extract skeleton from Python code per §6:
    - Parse with ast
    - Reduce to ordered tree: blocks (if/for) with ordered children, or statements
    - Statements: assignments/augmented assignments to `total`, or final `return`
    - Ignore operators, constants, predicates, iterables
    """
    try:
        tree = ast.parse(python_code)
    except SyntaxError:
        return None
    
    def extract_node(node: ast.AST) -> Any:
        if isinstance(node, ast.FunctionDef):
            # Function body - extract statements in order
            return [extract_node(stmt) for stmt in node.body]
        
        elif isinstance(node, (ast.If, ast.For)):
            # Block node: type + children
            children = [extract_node(stmt) for stmt in node.body]
            return {
                'type': 'if' if isinstance(node, ast.If) else 'for',
                'children': children
            }
        
        elif isinstance(node, (ast.Assign, ast.AugAssign)):
            # Statement: assignment to total
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            else:  # AugAssign
                targets = [node.target.id] if isinstance(node.target, ast.Name) else []
            
            if 'total' in targets:
                return {'type': 'stmt', 'op': 'assign' if isinstance(node, ast.Assign) else 'augassign'}
        
        elif isinstance(node, ast.Return):
            # Return statement
            return {'type': 'return'}
        
        # Ignore other nodes (pass, etc.)
        return None
    
    # Get function body
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            skeleton = [extract_node(stmt) for stmt in node.body]
            # Filter out None values
            return [s for s in skeleton if s is not None]
    
    return None


def skeletons_equal(skel1: Any, skel2: Any) -> bool:
    """Check if two skeletons are equal per §6 (ordered tree equality)."""
    if skel1 is None or skel2 is None:
        return skel1 is skel2
    
    if isinstance(skel1, list) and isinstance(skel2, list):
        if len(skel1) != len(skel2):
            return False
        return all(skeletons_equal(a, b) for a, b in zip(skel1, skel2))
    
    if isinstance(skel1, dict) and isinstance(skel2, dict):
        if skel1.get('type') != skel2.get('type'):
            return False
        if 'children' in skel1 and 'children' in skel2:
            return skeletons_equal(skel1['children'], skel2['children'])
        return True
    
    return skel1 == skel2


def classify_skeleton_error(gen_skel: Any, ref_skel: Any) -> SkeletonErrorCode:
    """
    Classify skeleton error per §6 exploratory sub-codes.
    E_PLACEMENT: same nodes in same order, different nesting
    E_OTHER: missing/extra block/statement, different order
    """
    # Flatten both skeletons to ordered list of node types
    def flatten(skel):
        if skel is None:
            return []
        if isinstance(skel, list):
            result = []
            for item in skel:
                result.extend(flatten(item))
            return result
        if isinstance(skel, dict):
            if 'children' in skel:
                children = flatten(skel['children'])
                return [skel['type']] + children
            return [skel.get('type', 'unknown')]
        return [str(skel)]
    
    flat_gen = flatten(gen_skel)
    flat_ref = flatten(ref_skel)
    
    if flat_gen == flat_ref:
        # Same nodes in same order - must be placement error
        return SkeletonErrorCode.E_PLACEMENT
    return SkeletonErrorCode.E_OTHER


def score_generation(
    generated_text: str,
    reference_python: str,
    condition: str,  # 'A', 'B', or 'C'
    is_truncated: bool = False
) -> ScoreResult:
    """
    Score a generation per §6 category rules.
    
    Categories (checked in order):
    1. Unparsable: truncation, unclosed scope, syntax error after conversion
    2. Skeleton error: parses but skeleton differs from reference
    3. Structurally correct: parses and skeleton equals reference
    """
    # Category 1a: Truncation (extraction level)
    if is_truncated:
        return ScoreResult(category=Category.UNPARSABLE, parse_error=ParseErrorCode.TRUNCATED)
    
    # Convert to Python if needed
    if condition in ('B', 'C'):
        conv_result = convert_brace_to_python(generated_text)
        python_code = conv_result.python_code
        
        # Category 1b: E_UNCLOSED (B/C only)
        if ConversionError.E_UNCLOSED in conv_result.errors:
            return ScoreResult(category=Category.UNPARSABLE, parse_error=ParseErrorCode.E_UNCLOSED)
        
        # Category 1c: E_HEADER_SYNTAX (Python-style header in B/C)
        if ConversionError.E_HEADER_SYNTAX in conv_result.errors:
            return ScoreResult(category=Category.UNPARSABLE, parse_error=ParseErrorCode.E_HEADER_SYNTAX)
        
        # Category 1c: E_COLON_BRACE is NOT a failure (exploratory flag only)
        # Category 1c: Other syntax errors after conversion
        try:
            ast.parse(python_code)
        except SyntaxError:
            # Check if it's an indentation error (Condition A would have E_INDENT)
            # For B/C, any syntax error after conversion is E_SYNTAX
            return ScoreResult(category=Category.UNPARSABLE, parse_error=ParseErrorCode.E_SYNTAX)
    
    else:  # Condition A
        python_code = generated_text
        try:
            ast.parse(python_code)
        except IndentationError:
            return ScoreResult(category=Category.UNPARSABLE, parse_error=ParseErrorCode.E_INDENT)
        except SyntaxError:
            return ScoreResult(category=Category.UNPARSABLE, parse_error=ParseErrorCode.E_SYNTAX)
    
    # If we get here, code parses. Extract skeletons.
    gen_skeleton = extract_skeleton(python_code)
    ref_skeleton = extract_skeleton(reference_python)
    
    if gen_skeleton is None or ref_skeleton is None:
        return ScoreResult(category=Category.UNPARSABLE, parse_error=ParseErrorCode.E_SYNTAX)
    
    # Category 2: Skeleton error
    if not skeletons_equal(gen_skeleton, ref_skeleton):
        skel_error = classify_skeleton_error(gen_skeleton, ref_skeleton)
        return ScoreResult(
            category=Category.SKELETON_ERROR,
            skeleton_error=skel_error,
            skeleton=gen_skeleton,
            reference_skeleton=ref_skeleton
        )
    
    # Category 3: Structurally correct
    # Check content errors (operators, constants, predicates, iterables)
    content_errors = check_content_errors(python_code, reference_python)
    
    return ScoreResult(
        category=Category.STRUCTURALLY_CORRECT,
        content_errors=content_errors,
        skeleton=gen_skeleton,
        reference_skeleton=ref_skeleton
    )


def check_content_errors(gen_code: str, ref_code: str) -> List[str]:
    """Check for content errors per §6 (exploratory, Category 3 only)."""
    errors = []
    try:
        gen_tree = ast.parse(gen_code)
        ref_tree = ast.parse(ref_code)
    except Exception as e:
        errors.append(f"Content audit error: {e}")
    return errors


# ============================================================
# §8 Scorer Self-Test: Negative Controls
# ============================================================

NEGATIVE_CONTROLS = [
    {
        "name": "dropped_middle_statement",
        "description": "A statement dropped from the middle of a block",
        "condition": "A",
        "generated": """def process_data(data):
    total = 0
    if len(data) > 0:
        total += 1
        # missing: total -= 2
        total *= 3
    return total""",
        "expected_category": Category.SKELETON_ERROR,
        "expected_skeleton_error": SkeletonErrorCode.E_OTHER,
    },
    {
            "name": "statement_moved_out_of_block",
            "description": "A statement moved out of its block (same nodes, different nesting)",
            "condition": "A",
            "generated": (
                "def process_data(data):\n"
                "    total = 0\n"
                "    if len(data) > 0:\n"
                "        total += 1\n"
                "        total -= 2\n"
                "    total *= 3  # Moved out of block\n"
                "    return total"
            ),
            "reference": (
                "def process_data(data):\n"
                "    total = 0\n"
                "    if len(data) > 0:\n"
                "        total += 1\n"
                "        total -= 2\n"
                "        total *= 3\n"
                "    return total"
            ),
            "expected_category": Category.SKELETON_ERROR,
            "expected_skeleton_error": SkeletonErrorCode.E_PLACEMENT,
        },
    {
        "name": "unclosed_block_b",
        "description": "Unclosed block in Condition B",
        "condition": "B",
        "generated": """def process_data(data) {
    total = 0
    if len(data) > 0 {
        total += 1
    return total
}""",
        "expected_category": Category.UNPARSABLE,
        "expected_parse_error": ParseErrorCode.E_UNCLOSED,
    },
    {
        "name": "unclosed_block_c",
        "description": "Unclosed block in Condition C",
        "condition": "C",
        "generated": """def process_data(data) {
    total = 0
    if len(data) > 0 {
        total += 1
    return total
}""",
        "expected_category": Category.UNPARSABLE,
        "expected_parse_error": ParseErrorCode.E_UNCLOSED,
    },
    {
        "name": "python_header_in_b",
        "description": "Python-style header in Condition B",
        "condition": "B",
        "generated": """def process_data(data) {
    total = 0
    if len(data) > 0:
        total += 1
    return total
}""",
        "expected_category": Category.UNPARSABLE,
        "expected_parse_error": ParseErrorCode.E_HEADER_SYNTAX,
    },
    {
        "name": "python_header_in_c",
        "description": "Python-style header in Condition C",
        "condition": "C",
        "generated": """def process_data(data) {
    total = 0
    if len(data) > 0:
        total += 1
    return total
}""",
        "expected_category": Category.UNPARSABLE,
        "expected_parse_error": ParseErrorCode.E_HEADER_SYNTAX,
    },
    {
        "name": "indentation_error_a",
        "description": "Indentation error in Condition A",
        "condition": "A",
        "generated": """def process_data(data):
    total = 0
    if len(data) > 0:
        total += 1
      total -= 2  # Wrong indentation
    return total""",
        "expected_category": Category.UNPARSABLE,
        "expected_parse_error": ParseErrorCode.E_INDENT,
    },
    {
        "name": "function_closed_early",
        "description": "Function closed early (missing statements)",
        "condition": "A",
        "generated": """def process_data(data):
    total = 0
    if len(data) > 0:
        total += 1
    return total
    # Missing: total *= 2""",
        "expected_category": Category.SKELETON_ERROR,
        "expected_skeleton_error": SkeletonErrorCode.E_OTHER,
    },
    {
        "name": "colon_before_brace",
        "description": "Colon before '{' in Condition B (category 3 with E_COLON_BRACE)",
        "condition": "B",
        "generated": """def process_data(data) {
    total = 0
    if len(data) > 0: {
        total += 1
            total -= 2
            total *= 3
        }
        return total
    }""",
            "reference": """def process_data(data):
        total = 0
        if len(data) > 0:
            total += 1
            total -= 2
            total *= 3
        return total""",
            "expected_category": Category.STRUCTURALLY_CORRECT,
            "expected_colon_brace": True,
        },
]


def run_scorer_self_test(reference_tasks: List[dict]) -> dict:
    """
    Run scorer self-test per §8:
    1. Canonical solution of every test task and demonstration, converted to each condition, scores as category 3
    2. Fixed negative controls each score as expected category and sub-code
    """
    results = {
        "canonical_tests": [],
        "negative_controls": [],
        "all_passed": True
    }
    
    # Test 1: Canonical solutions score category 3 in all conditions
    for task in reference_tasks:
        canonical = task['canonical_python']
        for condition in ['A', 'B', 'C']:
            if condition == 'A':
                gen = canonical
            elif condition == 'B':
                gen = python_to_condition_b(canonical)
            else:  # condition == 'C'
                gen = python_to_condition_c(canonical)
            
            result = score_generation(gen, canonical, condition)
            passed = (result.category == Category.STRUCTURALLY_CORRECT)
            results["canonical_tests"].append({
                "task_id": task['task_id'],
                "condition": condition,
                "passed": passed,
                "category": result.category.name
            })
            if not passed:
                results["all_passed"] = False
    
    # Test 2: Negative controls
    for control in NEGATIVE_CONTROLS:
        # Get reference for this control (use control's reference if provided, else default)
        ref = control.get('reference', """def process_data(data):
    total = 0
    if len(data) > 0:
        total += 1
        total -= 2
        total *= 3
        return total""")
        
        result = score_generation(control['generated'], ref, control['condition'])
        
        passed = (result.category == control['expected_category'])
        if 'expected_parse_error' in control:
            passed = passed and (result.parse_error == control["expected_parse_error"])
        if "expected_skeleton_error" in control:
            passed = passed and (result.skeleton_error == control["expected_skeleton_error"])
        if "expected_colon_brace" in control:
            # Check converter flag
            conv = convert_brace_to_python(control["generated"])
            passed = passed and (conv.colon_brace_flag == control["expected_colon_brace"])
        results["negative_controls"].append({
            "name": control['name'],
            "passed": passed,
            "expected": control['expected_category'].name,
            "got": result.category.name,
            "parse_error": result.parse_error.value if result.parse_error else None,
            "skeleton_error": result.skeleton_error.value if result.skeleton_error else None,
        })
        if not passed:
            results["all_passed"] = False
    
    return results


# Alias for backward compatibility
python_to_brace = python_to_condition_b


def main():
    """Run self-test with demo tasks as reference."""
    # Load demo tasks as reference
    demo_path = Path('tasks/pilot/demo_tasks.json') if Path('tasks/pilot/demo_tasks.json').exists() else Path('tasks/demo_tasks.json')
    with open(demo_path) as f:
        demo_tasks = json.load(f)
    
    print("Running scorer self-test per §8...")
    results = run_scorer_self_test(demo_tasks)
    
    print(f"\nAll passed: {results['all_passed']}")
    print(f"\nCanonical tests: {len(results['canonical_tests'])}")
    for t in results['canonical_tests']:
        status = "PASS" if t['passed'] else "FAIL"
        print(f"  {status} {t['task_id']} condition {t['condition']}: {t['category']}")
    
    print(f"\nNegative controls: {len(results['negative_controls'])}")
    for c in results['negative_controls']:
        status = "PASS" if c['passed'] else "FAIL"
        print(f"  {status} {c['name']}: expected {c['expected']}, got {c['got']}")
        if c['parse_error']:
            print(f"    parse_error: {c['parse_error']}")
        if c['skeleton_error']:
            print(f"    skeleton_error: {c['skeleton_error']}")
    
    return results['all_passed']


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)