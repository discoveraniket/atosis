#!/usr/bin/env python3
"""
Boundary-Span Surprisal Feasibility Checker (S1_prereg_pilot.md §7).

Performs the feasibility check confirming the Transformers CPU setup can compute:
1. Per-token log-probabilities on reference text spans.
2. Tokenizer audit for whitespace runs: 4, 8, 12, 16, and 20 spaces (raw and per-char surprisal).
3. Boundary-span surprisals across Conditions A, B, and C:
   - Primary span: from last token of inner block's last body statement through
     first token of next statement in enclosing block.
   - Sensitivity span: ending before the next statement's first token.
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Support direct script and module execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spatial_tax.converter import (
    python_to_condition_b,
    python_to_condition_c,
)
from scripts.pilot.build_prompts import build_prompt
from scripts.pilot.run_inference import (
    PINNED_MODEL_NAME,
    PINNED_REVISION,
    PINNED_CPU_THREADS,
    DEFAULT_SNAPSHOT_DIR,
    get_git_commit_hash,
)


SPACE_RUNS = [4, 8, 12, 16, 20]


def audit_space_tokenization(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    space_counts: List[int] = SPACE_RUNS,
) -> List[Dict[str, Any]]:
    """
    Tokenizer audit per §7:
    Record how the tokenizer splits runs of 4, 8, 12, 16, and 20 spaces.
    Reports raw surprisal and per-character surprisal following a newline context.
    """
    results = []

    for count in space_counts:
        # Evaluate whitespace following a newline context (\n + spaces)
        text = "\n" + (" " * count)
        tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids

        with torch.no_grad():
            logits = model(tokens).logits
            log_probs = torch.log_softmax(logits, dim=-1)

        # Sum negative log-likelihood for all space tokens (after \n)
        raw_nll = 0.0
        tok_records = []
        for idx in range(1, tokens.shape[1]):
            tok_id = tokens[0, idx].item()
            lp = log_probs[0, idx - 1, tok_id].item()
            tok_str = tokenizer.decode([tok_id])
            raw_nll += -lp
            tok_records.append({
                "token_id": tok_id,
                "token_repr": repr(tok_str),
                "log_prob": round(lp, 4),
                "surprisal_nats": round(-lp, 4),
            })

        per_char_nll = raw_nll / count

        results.append({
            "space_count": count,
            "token_count": len(tok_records),
            "tokens": tok_records,
            "raw_surprisal_nats": round(raw_nll, 4),
            "per_character_surprisal_nats": round(per_char_nll, 4),
            "raw_surprisal_bits": round(raw_nll / math.log(2), 4),
            "per_character_surprisal_bits": round(per_char_nll / math.log(2), 4),
        })

    return results


def compute_token_logprobs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Compute per-token log-probabilities and negative log-likelihood for a full text sequence.
    """
    encoding = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
    input_ids = encoding["input_ids"]
    offset_mapping = encoding["offset_mapping"][0].tolist()

    with torch.no_grad():
        logits = model(input_ids).logits
        log_probs = torch.log_softmax(logits, dim=-1)

    token_list = input_ids[0].tolist()
    token_records = []

    for t in range(len(token_list)):
        tok_id = token_list[t]
        tok_str = tokenizer.decode([tok_id])
        char_span = offset_mapping[t]

        if t == 0:
            lp = 0.0
            surprisal = 0.0
        else:
            lp = log_probs[0, t - 1, tok_id].item()
            surprisal = -lp

        token_records.append({
            "index": t,
            "token_id": tok_id,
            "token_str": tok_str,
            "char_span": char_span,
            "log_prob": lp,
            "surprisal": surprisal,
        })

    return input_ids, token_records


def locate_boundary_span_tokens(
    token_records: List[Dict[str, Any]],
    full_text: str,
    inner_last_stmt: str,
    next_enclosing_stmt: str,
) -> Tuple[int, int]:
    """
    Locate the token index span per §7:
    From the last token of inner_last_stmt through the first token of next_enclosing_stmt.
    """
    pos_inner_start = full_text.find(inner_last_stmt)
    pos_inner_end = pos_inner_start + len(inner_last_stmt)
    pos_next_start = full_text.find(next_enclosing_stmt)

    if pos_inner_start == -1 or pos_next_start == -1:
        raise ValueError(
            f"Could not locate boundary statements in text:\n"
            f"inner_last_stmt: '{inner_last_stmt}', next_enclosing_stmt: '{next_enclosing_stmt}'"
        )

    tok1_idx = None
    tok2_idx = None

    for r in token_records:
        start, end = r["char_span"]
        if start < pos_inner_end <= end:
            tok1_idx = r["index"]
        if tok2_idx is None and end > pos_next_start:
            tok2_idx = r["index"]

    if tok1_idx is None or tok2_idx is None:
        raise ValueError(
            f"Failed to map boundary characters ({pos_inner_end}, {pos_next_start}) to token indices."
        )

    return tok1_idx, tok2_idx


def compute_span_surprisal(
    token_records: List[Dict[str, Any]],
    start_idx: int,
    end_idx: int,
) -> Dict[str, Any]:
    """
    Compute primary and sensitivity boundary span surprisals per §7:
    - Primary: [start_idx, end_idx] (through first token of next statement)
    - Sensitivity: [start_idx, end_idx - 1] (before first token of next statement)
    """
    primary_records = token_records[start_idx : end_idx + 1]
    primary_nll = sum(r["surprisal"] for r in primary_records)

    sensitivity_records = token_records[start_idx : end_idx]
    sensitivity_nll = sum(r["surprisal"] for r in sensitivity_records)

    return {
        "start_token_idx": start_idx,
        "end_token_idx": end_idx,
        "primary_span": {
            "token_count": len(primary_records),
            "tokens": [r["token_str"] for r in primary_records],
            "token_ids": [r["token_id"] for r in primary_records],
            "total_surprisal_nats": round(primary_nll, 4),
            "mean_surprisal_nats": round(primary_nll / max(len(primary_records), 1), 4),
            "total_surprisal_bits": round(primary_nll / math.log(2), 4),
        },
        "sensitivity_span": {
            "token_count": len(sensitivity_records),
            "tokens": [r["token_str"] for r in sensitivity_records],
            "token_ids": [r["token_id"] for r in sensitivity_records],
            "total_surprisal_nats": round(sensitivity_nll, 4),
            "mean_surprisal_nats": round(sensitivity_nll / max(len(sensitivity_records), 1), 4),
            "total_surprisal_bits": round(sensitivity_nll / math.log(2), 4),
        },
    }


def run_surprisal_feasibility(
    snapshot_dir: Optional[Path] = None,
    demo_tasks_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute §7 feasibility check:
    1. Loads Qwen2.5-Coder-1.5B-Instruct in bfloat16 on CPU pinned to 16 threads.
    2. Runs whitespace tokenizer audit (4, 8, 12, 16, 20 spaces).
    3. Computes boundary-span surprisal across Conditions A, B, and C on demo_d2_001.
    4. Saves audit report and confirms feasibility.
    """
    torch.set_num_threads(PINNED_CPU_THREADS)
    model_dir = Path(snapshot_dir or DEFAULT_SNAPSHOT_DIR)

    print(f"Loading model & tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16)
    model.eval()

    print("\n--- Running Tokenizer Audit (§7) ---")
    space_audit = audit_space_tokenization(model, tokenizer, SPACE_RUNS)
    for entry in space_audit:
        print(f"  Spaces {entry['space_count']:2d}: tokens={len(entry['tokens'])}, "
              f"NLL={entry['raw_surprisal_nats']:.4f} nats ({entry['per_character_surprisal_nats']:.4f} nats/char)")

    # Load demonstrations for boundary-span check
    demo_p = Path(demo_tasks_path or (PROJECT_ROOT / "tasks" / "pilot" / "demo_tasks.json"))
    with open(demo_p) as f:
        demos = json.load(f)

    # Use demo_d2_001 (depth 2) for boundary span exit feasibility
    target_task = [d for d in demos if d.get("depth") == 2][0]
    canonical = target_task["canonical_python"]

    # Boundary statements in demo_d2_001:
    # Inner block last statement: "total *= 34"
    # Next enclosing statement: "total += 33"
    inner_last = "total *= 34"
    next_enclosing = "total += 33"

    print("\n--- Running Boundary-Span Surprisal Feasibility (§7) ---")
    condition_results = {}

    for cond in ("A", "B", "C"):
        prompt_text = build_prompt(demos, target_task, cond)

        if cond == "A":
            solution_body = canonical.replace("def process_data(data):", "").lstrip("\n")
        elif cond == "B":
            b_code = python_to_condition_b(canonical)
            solution_body = b_code.replace("def process_data(data) {", "").lstrip("\n")
        else:
            c_code = python_to_condition_c(canonical)
            solution_body = c_code.replace("def process_data(data) {", "").lstrip("\n")

        full_eval_text = prompt_text + solution_body

        input_ids, token_records = compute_token_logprobs(model, tokenizer, full_eval_text)
        tok1, tok2 = locate_boundary_span_tokens(token_records, full_eval_text, inner_last, next_enclosing)

        span_data = compute_span_surprisal(token_records, tok1, tok2)
        condition_results[cond] = span_data

        print(f"  Condition {cond}: span={span_data['primary_span']['tokens']}")
        print(f"    Primary NLL: {span_data['primary_span']['total_surprisal_nats']:.4f} nats "
              f"({span_data['primary_span']['token_count']} tokens)")
        print(f"    Sensitivity NLL: {span_data['sensitivity_span']['total_surprisal_nats']:.4f} nats "
              f"({span_data['sensitivity_span']['token_count']} tokens)")

    summary = {
        "status": "PASS",
        "feasibility_confirmed": True,
        "model_name": PINNED_MODEL_NAME,
        "model_revision": PINNED_REVISION,
        "cpu_threads": torch.get_num_threads(),
        "precision": "bfloat16",
        "git_commit": get_git_commit_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tokenizer_space_audit": space_audit,
        "boundary_span_surprisal": condition_results,
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSurprisal feasibility report saved to: {out_p}")

    print("\n" + "=" * 70)
    print("SURPRISAL FEASIBILITY CHECK: PASSED (§7 requirements confirmed)")
    print("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Boundary-Span Surprisal Feasibility Checker (§7).")
    parser.add_argument(
        "--output",
        type=str,
        default="results/pilot/surprisal_feasibility.json",
        help="Path to save feasibility JSON report.",
    )
    args = parser.parse_args()

    out_p = Path(args.output)
    if not out_p.is_absolute():
        out_p = PROJECT_ROOT / out_p

    run_surprisal_feasibility(output_path=out_p)


if __name__ == "__main__":
    main()
