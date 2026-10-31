#!/usr/bin/env python3
"""
Pilot Health Check & Analysis Reporter (S1_prereg_pilot.md §8).

Computes cell-by-cell metrics and evaluates the fixed pilot health check gates:
1. Headline failure rates:
   - Upper bound: Category 1 + Category 2 (scope-closing failures)
   - Lower bound: Category 1 + E_PLACEMENT
   - E_OTHER rate
2. Fixed health check gates (§8):
   - Pass rate, d = 1, floor: >= 40% in best-performing condition
   - Pass rate, d = 1, ceiling: <= 90% pooled across conditions
   - Pass rate, d = 5, floor: > 5% in best-performing condition
   - Parse failures, d = 1: < 10% per cell (at most 2 of 30)
   - Truncation rate: < 5% per cell
3. Stratified extraction hand-check sampler (§8):
   - Samples 20 generations per condition with seed 5678 (6 from d=1, 7 from d=3, 7 from d=5).
   - Exports manual review sheet for the >= 19/20 agreement check.
"""

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Support direct script and module execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


HAND_CHECK_SEED = 5678
HAND_CHECK_STRATA = {1: 6, 3: 7, 5: 7}  # 20 per condition


def compute_cell_metrics(cell_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute structural metrics and failure rates for a single cell (depth × condition)."""
    n = len(cell_records)
    if n == 0:
        return {"total_generations": 0}

    cat1 = sum(1 for r in cell_records if r.get("category") == 1)
    cat2 = sum(1 for r in cell_records if r.get("category") == 2)
    cat3 = sum(1 for r in cell_records if r.get("category") == 3)

    # Category 1 breakdown
    truncated = sum(1 for r in cell_records if r.get("parse_error") == "truncated" or r.get("extraction", {}).get("is_truncated"))
    parse_failures = cat1 - truncated  # Non-truncated category 1 per §8

    # Category 2 breakdown
    e_placement = sum(1 for r in cell_records if r.get("skeleton_error") == "E_PLACEMENT")
    e_other = sum(1 for r in cell_records if r.get("skeleton_error") == "E_OTHER")

    # Secondary metrics
    pass_at_1 = sum(1 for r in cell_records if r.get("pass_at_1", False))
    colon_brace = sum(1 for r in cell_records if r.get("colon_brace_flag", False))
    content_errors = sum(len(r.get("content_errors", [])) for r in cell_records if r.get("category") == 3)

    # Rates
    upper_bound = (cat1 + cat2) / n
    lower_bound = (cat1 + e_placement) / n
    e_other_rate = e_other / n
    parse_failure_rate = parse_failures / n
    truncation_rate = truncated / n
    pass_at_1_rate = pass_at_1 / n

    return {
        "total_generations": n,
        "category_1_count": cat1,
        "category_2_count": cat2,
        "category_3_count": cat3,
        "truncated_count": truncated,
        "parse_failure_count": parse_failures,
        "e_placement_count": e_placement,
        "e_other_count": e_other,
        "pass_at_1_count": pass_at_1,
        "colon_brace_count": colon_brace,
        "content_error_count": content_errors,
        "upper_bound_structural_failure_rate": round(upper_bound, 4),
        "lower_bound_structural_failure_rate": round(lower_bound, 4),
        "e_other_rate": round(e_other_rate, 4),
        "parse_failure_rate": round(parse_failure_rate, 4),
        "truncation_rate": round(truncation_rate, 4),
        "pass_at_1_rate": round(pass_at_1_rate, 4),
    }


def evaluate_health_gates(cells: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate the fixed pilot health check gates defined in §8:
    - Pass rate, d = 1, floor: >= 40% in best-performing condition
    - Pass rate, d = 1, ceiling: <= 90% pooled across conditions
    - Pass rate, d = 5, floor: > 5% in best-performing condition
    - Parse failures, d = 1: < 10% per cell (at most 2 of 30)
    - Truncation rate: < 5% per cell across all cells
    """
    # 1. d=1 Pass rate floor: >= 40% in best-performing condition
    d1_pass_rates = {
        cond: cells.get(f"d1_{cond}", {}).get("pass_at_1_rate", 0.0)
        for cond in ("A", "B", "C")
    }
    best_d1_rate = max(d1_pass_rates.values()) if d1_pass_rates else 0.0
    gate_d1_floor_passed = (best_d1_rate >= 0.40)

    # 2. d=1 Pass rate ceiling: <= 90% pooled across conditions
    d1_total = sum(cells.get(f"d1_{cond}", {}).get("total_generations", 0) for cond in ("A", "B", "C"))
    d1_passed = sum(cells.get(f"d1_{cond}", {}).get("pass_at_1_count", 0) for cond in ("A", "B", "C"))
    pooled_d1_rate = d1_passed / max(d1_total, 1)
    gate_d1_ceiling_passed = (pooled_d1_rate <= 0.90)

    # 3. d=5 Pass rate floor: > 5% in best-performing condition
    d5_pass_rates = {
        cond: cells.get(f"d5_{cond}", {}).get("pass_at_1_rate", 0.0)
        for cond in ("A", "B", "C")
    }
    best_d5_rate = max(d5_pass_rates.values()) if d5_pass_rates else 0.0
    gate_d5_floor_passed = (best_d5_rate > 0.05)

    # 4. Parse failures at d=1: < 10% per cell (at most 2 of 30)
    d1_parse_failures = {
        cond: cells.get(f"d1_{cond}", {}).get("parse_failure_rate", 0.0)
        for cond in ("A", "B", "C")
    }
    gate_d1_parse_passed = all(rate < 0.10 for rate in d1_parse_failures.values())

    # 5. Truncation rate: < 5% per cell across all cells
    truncation_rates = {
        cell_id: metrics.get("truncation_rate", 0.0)
        for cell_id, metrics in cells.items()
    }
    gate_truncation_passed = all(rate < 0.05 for rate in truncation_rates.values())

    all_gates_passed = (
        gate_d1_floor_passed
        and gate_d1_ceiling_passed
        and gate_d5_floor_passed
        and gate_d1_parse_passed
        and gate_truncation_passed
    )

    return {
        "all_gates_passed": all_gates_passed,
        "gates": {
            "d1_pass_rate_floor": {
                "target": ">= 0.40 in best-performing condition",
                "observed_rates": d1_pass_rates,
                "best_rate": round(best_d1_rate, 4),
                "passed": gate_d1_floor_passed,
            },
            "d1_pass_rate_ceiling": {
                "target": "<= 0.90 pooled across conditions",
                "pooled_rate": round(pooled_d1_rate, 4),
                "passed": gate_d1_ceiling_passed,
            },
            "d5_pass_rate_floor": {
                "target": "> 0.05 in best-performing condition",
                "observed_rates": d5_pass_rates,
                "best_rate": round(best_d5_rate, 4),
                "passed": gate_d5_floor_passed,
            },
            "d1_parse_failures": {
                "target": "< 0.10 per cell at d=1",
                "observed_rates": d1_parse_failures,
                "passed": gate_d1_parse_passed,
            },
            "truncation_rate": {
                "target": "< 0.05 per cell",
                "max_observed_rate": max(truncation_rates.values()) if truncation_rates else 0.0,
                "passed": gate_truncation_passed,
            },
        },
    }


def sample_extraction_hand_check(
    records: List[Dict[str, Any]],
    seed: int = HAND_CHECK_SEED,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Sample 20 generations per condition stratified by depth:
    6 from d=1, 7 from d=3, 7 from d=5 (seed 5678 per §8).
    Prepares review sample for researcher hand-check.
    """
    rng = random.Random(seed)
    samples_by_condition: Dict[str, List[Dict[str, Any]]] = {}

    for condition in ("A", "B", "C"):
        cond_records = [r for r in records if r.get("condition") == condition]
        selected_samples = []

        for depth, target_count in HAND_CHECK_STRATA.items():
            stratum = [r for r in cond_records if r.get("depth") == depth]
            if len(stratum) <= target_count:
                chosen = stratum
            else:
                chosen = rng.sample(stratum, target_count)

            for sample_idx, r in enumerate(chosen, 1):
                raw_text = r.get("extraction", {}).get("raw_text", "")
                cut_body = r.get("extraction", {}).get("extracted_body", "")

                selected_samples.append({
                    "sample_id": f"handcheck_{condition}_d{depth}_{sample_idx:02d}",
                    "task_id": r.get("task_id", ""),
                    "condition": condition,
                    "depth": depth,
                    "raw_output": raw_text,
                    "scorer_extracted_text": cut_body,
                    "scorer_cut_reason": r.get("extraction", {}).get("cut_reason", ""),
                    "researcher_verified_end_position": None,
                    "agreement": None,  # To be marked by researcher
                    "notes": "",
                })

        samples_by_condition[condition] = selected_samples

    return samples_by_condition


def generate_health_summary(
    eval_records_file: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Load evaluation records, compute metrics per cell, evaluate gates,
    sample hand-check, and generate structured report files.
    """
    with open(eval_records_file, encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])
    if not records:
        raise ValueError(
            f"No evaluation records found in {eval_records_file}. "
            "Cannot evaluate pilot health on empty or missing records (Guideline §5.7)."
        )

    # Group by cell (depth × condition)
    cells_data: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        depth = r.get("depth", 0)
        cond = r.get("condition", "")
        cell_id = f"d{depth}_{cond}"
        cells_data.setdefault(cell_id, []).append(r)

    # Compute metrics per cell
    cells_metrics: Dict[str, Dict[str, Any]] = {}
    for cell_id, cell_recs in sorted(cells_data.items()):
        cells_metrics[cell_id] = compute_cell_metrics(cell_recs)

    # Evaluate health check gates
    gate_results = evaluate_health_gates(cells_metrics)

    # Sample hand-check
    hand_check_samples = sample_extraction_hand_check(records, seed=HAND_CHECK_SEED)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_generations_evaluated": len(records),
        "cells": cells_metrics,
        "health_check_gates": gate_results,
        "overall_status": "PASS" if gate_results["all_gates_passed"] else "FAIL",
    }

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "pilot_health_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    hand_check_path = output_dir / "extraction_hand_check_samples.json"
    with open(hand_check_path, "w", encoding="utf-8") as f:
        json.dump(hand_check_samples, f, indent=2)

    print(f"Pilot health summary saved to: {summary_path}")
    print(f"Extraction hand-check samples saved to: {hand_check_path}")
    print(f"Overall Health Gates Status: {summary['overall_status']}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Pilot Health Check and Reporter per §8.")
    parser.add_argument("eval_records", type=str, help="Path to evaluation records JSON file.")
    parser.add_argument("--output-dir", type=str, default="results/pilot", help="Output directory.")
    args = parser.parse_args()

    eval_p = Path(args.eval_records)
    out_p = Path(args.output_dir)

    generate_health_summary(eval_p, out_p)


if __name__ == "__main__":
    main()
