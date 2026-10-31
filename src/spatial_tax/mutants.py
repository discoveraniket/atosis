#!/usr/bin/env python3
"""
Single-Move Scope Mutant Generator and Evaluator (S1_prereg_pilot.md §3, §8)

Implements the single-move scope mutant generator and evaluation filter:
- For each task, all single-move scope mutants are generated; none is sampled.
- Outward move: moves the last statement of a block to just after that block.
- Inward move: moves the first update statement after a block into that block as its last statement.
- Invariants strictly preserved:
    1. Document order is preserved.
    2. `total = 0` and `return total` never move.
    3. Moves that would leave a block empty are never generated.
- Unit tests: 3 fixed inputs: [], [1, 2, 3], [-1, 5, 0, 2].
- Threshold: At least 80% of a task's mutants must fail the unit tests (i.e. caught).
"""

import ast
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional


@dataclass
class Mutant:
    """A single-move scope mutant."""
    name: str
    move_type: str  # "outward" or "inward"
    target_level: int
    mutated_python: str


@dataclass
class MutantCheckResult:
    """Results of running the mutant catch-rate check on a task."""
    task_id: str
    total_mutants: int
    caught_mutants: int
    catch_rate: float
    passed: bool
    details: List[Dict[str, Any]] = field(default_factory=list)


def is_update_stmt(stmt: ast.AST) -> bool:
    """
    Check if an AST statement is an accumulator update statement.
    Allowed update statements are assignments or augmented assignments to `total`.
    Excludes the initial initialization `total = 0` and `return total`.
    """
    if isinstance(stmt, ast.AugAssign):
        if isinstance(stmt.target, ast.Name) and stmt.target.id == "total":
            return True
        return False
    if isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            if isinstance(target, ast.Name) and target.id == "total":
                # Pin initial total = 0
                if isinstance(stmt.value, ast.Constant) and stmt.value.value == 0:
                    return False
                return True
    return False


def find_blocks_with_parents(tree: ast.AST) -> List[Tuple[ast.AST, List[ast.AST], int, int]]:
    """
    Traverse the AST to find all compound blocks (ast.If, ast.For) with parent list references.
    Returns a list of tuples: (block_node, parent_body_list, index_in_parent, nesting_level).
    """
    blocks = []

    def traverse(body_list: List[ast.AST], level: int) -> None:
        for idx, stmt in enumerate(body_list):
            if isinstance(stmt, (ast.If, ast.For)):
                blocks.append((stmt, body_list, idx, level))
                traverse(stmt.body, level + 1)

    # FunctionDef body is the root block list
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            traverse(node.body, 0)
            break

    return blocks


def generate_scope_mutants(canonical_python: str) -> List[Mutant]:
    """
    Generate all valid single-move scope mutants (inward and outward) for a task per §3.
    Preserves document order, pins `total = 0` and `return total`, and disallows moves
    that would leave any block empty.
    """
    base_tree = ast.parse(canonical_python)
    blocks_meta = find_blocks_with_parents(base_tree)
    mutants = []

    for block_idx in range(len(blocks_meta)):
        # 1. Outward move: Move last statement of block to just after that block
        tree_out = ast.parse(canonical_python)
        blocks_out = find_blocks_with_parents(tree_out)
        block_node, parent_list, idx_in_parent, level = blocks_out[block_idx]

        # Invariant check: Cannot leave block empty (needs at least 2 statements)
        # And last statement must be an update statement (not return, not block)
        if len(block_node.body) > 1 and is_update_stmt(block_node.body[-1]):
            stmt_to_move = block_node.body.pop()
            parent_list.insert(idx_in_parent + 1, stmt_to_move)
            mutated_code = ast.unparse(tree_out)
            mutants.append(
                Mutant(
                    name=f"mutant_outward_lvl{level}_block{block_idx}",
                    move_type="outward",
                    target_level=level,
                    mutated_python=mutated_code,
                )
            )

        # 2. Inward move: Move first update statement after a block into that block as its last statement
        tree_in = ast.parse(canonical_python)
        blocks_in = find_blocks_with_parents(tree_in)
        block_node, parent_list, idx_in_parent, level = blocks_in[block_idx]

        if idx_in_parent + 1 < len(parent_list):
            candidate_stmt = parent_list[idx_in_parent + 1]
            # Must be an update statement (not return total, not total = 0, not another block)
            if is_update_stmt(candidate_stmt):
                stmt_to_move = parent_list.pop(idx_in_parent + 1)
                block_node.body.append(stmt_to_move)
                mutated_code = ast.unparse(tree_in)
                mutants.append(
                    Mutant(
                        name=f"mutant_inward_lvl{level}_block{block_idx}",
                        move_type="inward",
                        target_level=level,
                        mutated_python=mutated_code,
                    )
                )

    return mutants


def execute_function(python_code: str, test_inputs: List[List[int]]) -> List[Any]:
    """
    Execute a python function string safely against unit test inputs.
    Returns the list of outputs for each input, or string error representation if execution fails.
    """
    safe_globals: Dict[str, Any] = {
        "__builtins__": {
            "len": len,
            "range": range,
            "print": print,
        }
    }
    local_scope: Dict[str, Any] = {}
    try:
        compiled = compile(python_code, "<string>", "exec")
        exec(compiled, safe_globals, local_scope)
        fn = local_scope.get("process_data")
        if not callable(fn):
            return ["ERROR_FUNCTION_NOT_CALLABLE"] * len(test_inputs)
        
        results = []
        for inp in test_inputs:
            # Pass a copy to avoid mutation side effects
            res = fn(inp.copy())
            results.append(res)
        return results
    except Exception as e:
        return [f"ERROR_{type(e).__name__}_{e}"] * len(test_inputs)


def check_task_mutants(
    canonical_python: str,
    unit_test_inputs: List[List[int]],
    task_id: str = "",
    min_catch_rate: float = 0.80,
) -> MutantCheckResult:
    """
    Run all single-move scope mutants against unit tests.
    A mutant is caught if its outputs on unit test inputs differ from the canonical solution.
    Returns MutantCheckResult indicating whether catch_rate >= min_catch_rate (default 0.80 per §3 & §8).
    """
    ref_outputs = execute_function(canonical_python, unit_test_inputs)
    mutants = generate_scope_mutants(canonical_python)

    if not mutants:
        # If no mutants could be generated, the task cannot satisfy the catch rate requirement
        return MutantCheckResult(
            task_id=task_id,
            total_mutants=0,
            caught_mutants=0,
            catch_rate=0.0,
            passed=False,
            details=[{"error": "No valid single-move mutants generated"}],
        )

    caught_count = 0
    details = []

    for mutant in mutants:
        mutant_outputs = execute_function(mutant.mutated_python, unit_test_inputs)
        is_caught = (mutant_outputs != ref_outputs)
        if is_caught:
            caught_count += 1
        details.append({
            "name": mutant.name,
            "move_type": mutant.move_type,
            "target_level": mutant.target_level,
            "is_caught": is_caught,
            "mutant_outputs": mutant_outputs,
            "ref_outputs": ref_outputs,
        })

    catch_rate = caught_count / len(mutants)
    passed = catch_rate >= min_catch_rate

    return MutantCheckResult(
        task_id=task_id,
        total_mutants=len(mutants),
        caught_mutants=caught_count,
        catch_rate=catch_rate,
        passed=passed,
        details=details,
    )
