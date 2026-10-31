# Pre-Registration: H2 Spatial Tax Pilot

| Field | Value |
| :---- | :---- |
| Version | 1.0 |
| Date written | 2026-10-31 |
| Status | Draft. Not frozen. Freeze before generating any task. |
| Researcher | Independent (no institution) |
| Target model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Model revision (commit hash) | `2e1fd397ee46e1388853d2af2c993145b0f1098a` |
| External timestamp | Recorded in `manifest.json` (`external_timestamp_doi`) after depositing the tagged file |

**Purpose of the pilot.** Check that the experiment works: the generator, converters, scorer, extraction rule, and prompt balance. The pilot does not test H2. Pilot results are not used to support or reject the hypothesis, and they are not pooled into the main study.

---

## 1. Research Question

Does indentation-based scoping increase structural failures in a causal decoder as nesting depth grows? Do explicit delimiters (braces) reduce that failure rate when task logic and depth are held fixed?

---

## 2. Conditions

All conditions share identical logic, identifiers, and control flow. Only the scoping syntax differs. Semicolons are not used. Tasks contain no dict or set literals, f-strings, or braces inside strings or comments, so braces appear only as scope delimiters.

**Condition A: indentation only**

```py
def process_data(data):
    total = 0
    if len(data) > 0:
        for val in data:
            if val > 0:
                total += val
            total -= 1
        total *= 2
    return total
```

**Condition B: braces with indentation**

```py
def process_data(data) {
    total = 0
    if len(data) > 0 {
        for val in data {
            if val > 0 {
                total += val
            }
            total -= 1
        }
        total *= 2
    }
    return total
}
```

**Condition C: braces, no indentation**

```py
def process_data(data) {
total = 0
if len(data) > 0 {
for val in data {
if val > 0 {
total += val
}
total -= 1
}
total *= 2
}
return total
}
```

**Contrasts (all pre-specified):**

| Contrast | Question |
| :---- | :---- |
| C1: A vs. C | Do braces replace indentation? (primary) |
| C2: A vs. B | Do braces help as an added signal? |
| C3: B vs. C | Once braces exist, does indentation still help? |

B and C are never pooled into one "braced" group.

---

## 3. Tasks

**Allowed constructs:** `=`, `+=`, `-=`, `*=` on one accumulator `total`; `if <predicate>:`; `for <var> in <iterable>:`; one final `return total`. No `else`, `elif`, `while`, nested functions, or comprehensions.

**Structural rules**

1. **Post-exit statements.** Every block at level k contains at least one state-changing statement after its child block ends and before its own end (except the innermost block, which has no child block). This rules out the trivial "close everything at the end" pattern.  
2. **Data-dependent predicates and iterables.** Predicates depend on running state or input data, for example `total % 2 == 0` or `v > 0`. Placing a statement inside or outside a block must change the output. Trivial wrappers such as `if True` are not used.  
3. **Unique statements.** Every state-changing update statement uses a distinct constant from the task's constant pool, so generated statements can be matched to the reference unambiguously.
4. **Unambiguous specification.** Each prompt has a `#` comment specification block immediately above the function signature that states the nesting and ordering in words, identical across conditions and containing no syntax cues, braces, or quotes.

**Parameters (fixed before task generation):**

| Parameter | Value |
| :---- | :---- |
| Depths | d = 1, 3, 5 |
| Tasks per depth | 30 |
| Total test tasks | 90 (× 3 conditions = 270 model generations) |
| Demonstrations | 3 tasks at d = 1, 2, 3, generated with a separate seed and excluded from the test set |
| Unit tests per task | 3 fixed inputs passed as `data`: `[]`, `[1, 2, 3]`, and `[-1, 5, 0, 2]` |
| Block headers | Level 1 is `if len(data) > 0:` or `for v1 in data:`. If level 1 is `if`, level 2 is `for v1 in data:`. Deeper levels draw without consecutive repeats from `if <v> > 0:`, `if <v> % 2 == 0:`, `if total % 2 == 0:`, and (at most once more per task) `for v2 in data:`, where `<v>` is the innermost active loop variable. |
| Statements per block | Pre-child update statements per block: 3 at d = 1; 2 at d = 2 and d = 3; 1 at d = 5. Every non-innermost block also has 1 post-exit update statement after its child block ends. |
| Constants | Unique integers 1 to 99 for arithmetic update statements (`+=`, `-=`, `*=`); no repeats within a task. Initializations (`total = 0`) and predicate comparands (`> 0`, `% 2 == 0`) are fixed structural elements and exempt from the unique pool. |
| Task seed | 1234 |
| Demonstration seed | 9000 |

**Mutant check.** For each task, all single-move scope mutants are generated; none is sampled. A mutant moves either the last statement of a block to just after that block (outward), or the first update statement after a block into that block as its last statement (inward). Document order is preserved. `total = 0` and `return total` never move, and a move that would leave a block empty is not generated. At least 80% of a task's mutants must fail the unit tests. A task that fails this check is regenerated under the same rule, and the regeneration count is recorded.

---

## 4. Environment

| Item | Value |
| :---- | :---- |
| Hardware | CPU only. Intel Core Ultra 7 265K (20 cores; pinned to 16 threads via `torch.set_num_threads(16)`). No discrete GPU. |
| Operating system | Windows 11 (10.0.26300) |
| Python | 3.12.10 |
| PyTorch | 2.15.0+cpu |
| Transformers | 5.21.1 |
| Tokenizers | 0.23.3 |
| Safetensors | 0.8.0 |
| NumPy | 2.5.4 |
| Precision | bfloat16 (native model weight precision) |
| Model storage | Hugging Face release stored and loaded natively as bfloat16 (`dtype=torch.bfloat16`) |
| Virtual environment | `.venv` in the project folder; `requirements.txt` pins the versions above |

Record the model revision hash, the tokenizer files, and the SHA-256 of the safetensors file in `manifest.json` before task generation.

---

## 5. Inference

- **Model loading:** Transformers, `bfloat16` (`dtype=torch.bfloat16`), CPU (`torch.set_num_threads(16)`).  
- **Prompt mode:** raw completion. No chat template.  
- **Decoding:** greedy (`do_sample=False`), `max_new_tokens=512`.  
- **Seed:** `torch.manual_seed(42)`. Recorded, though greedy decoding is deterministic.  
- **Determinism check (before the pilot):** run the 3 demonstration tasks across all 3 conditions (9 prompts) three times at the final settings. All outputs must be identical across runs. If not, stop and record the cause.

**Prompt layout (same for every condition)**

1. The three demonstrations, in order d = 1, 2, 3, separated by a blank line (`\n\n`). Each demonstration is a `#` comment specification block followed by a complete function in that condition's syntax.  
2. The test task: its `#` comment specification block followed by its signature (`def process_data(data):` in A; `def process_data(data) {` in B and C).  
3. The prompt ends with one newline (`\n`) after the signature line. The model writes the entire function body (and the closing `}` in B and C).

**Few-shot format consistency (B and C).** The demonstrations in B and C use braced headers on every line that opens a block. No format reminder is added to the prompt. A script checks every demonstration for this before the run. Whether this is enough is tested in the pilot health check (Section 8), not assumed here.

---

## 6. Output Extraction and Scoring

**Extraction**

- **Special tokens (all conditions):** generations are decoded with special tokens kept. Before any condition-specific rule below, the text is cut at the first special token (any string of the form `<|...|>`, including the end-of-sequence tokens `<|endoftext|>` and `<|im_end|>`). Any special token seen is recorded.
- **A:** cut the output at the first non-empty line with zero indentation, or at the end-of-sequence token.  
- **B and C:** the prompt supplies the opening `{`, so depth starts at 1. Depth is counted with the line rules in "Conversion to Python (B and C)" below, not by counting raw brace characters. Cut the output after the closing line that returns depth to 0, or at end-of-sequence.
- **Truncation** is defined at the extraction level: a generation is truncated if no scope end and no end-of-sequence was reached within 512 tokens. Hitting the token limit after the scope has closed is not truncation. The raw token count is recorded separately.  
- Text after the cut is discarded. Record an overrun flag (exploratory) for any non-whitespace text after the cut.

**Conversion to Python (B and C)**

Each line is classified after removing any trailing `#` comment (comment text never counts as structure) and surrounding whitespace:

- **Empty line:** skipped.  
- **A line that is exactly `}`:** closes one block (depth − 1).  
- **A line that ends with `{`:** opens a block. The header is the text before `{`. A single `:` immediately before `{` is removed and the generation is flagged `E_COLON_BRACE` (exploratory; not a failure). The converter writes the header followed by `:` and increases depth by 1.  
- **Any other line:** a statement. Braces anywhere else in it are left in place, so the Python parser rejects the output (category 1).  

The output is re-indented by depth, 4 spaces per level. Indentation in the generated text is ignored, so B and C convert identically. A Python-style header without `{` is left as is and fails to parse (`E_HEADER_SYNTAX`). If depth is above 0 at the end of the extracted text, the generation is category 1 (`E_UNCLOSED`) even when the converted code parses.

**Stop sequence.** None. Generation runs to 512 tokens or end-of-sequence. Any stop sequence would require a new version of this document, with its equivalence test specified before use.

**Categories (each generation falls into exactly one; checked in this order)**

1. **Unparsable:** truncation, an unclosed scope, or any syntax error after conversion to Python. Each such generation gets exactly one label, assigned in this order: (a) `truncated`: no scope end and no end-of-sequence within 512 tokens; (b) `E_UNCLOSED` (B and C only): end-of-sequence reached while a block is still open; (c) a syntax error after conversion, labelled `E_HEADER_SYNTAX` when caused by a Python-style header in B or C, `E_INDENT` when caused by indentation in A, and `E_SYNTAX` otherwise. A truncated generation is never also labelled `E_UNCLOSED`. Condition A has no unclosed scope: its scope-closing errors appear as `E_INDENT` or as category 2.
2. **Skeleton error:** parses, but the generated skeleton differs from the reference skeleton.  
3. **Structurally correct:** parses, and the generated skeleton equals the reference skeleton.

**Skeleton.** The converted output is parsed with Python's `ast` module and reduced to an ordered tree. A node is either a block (`if` or `for`) with its ordered children, or a statement (an assignment or augmented assignment to `total`, or the final `return`). Operators, constants, predicates, and iterables are ignored. The reference skeleton is built from the canonical solution with the same function. A missing block, a missing or extra statement, a statement under a different parent, and statements in a different order all change the skeleton.

**Headline metric:** structural failure rate = category 1 + category 2, computed over all generations. Parse failures are included, not dropped.

**Category 2 sub-codes (exploratory)**

- `E_PLACEMENT`: the generated skeleton lists the same node kinds (block types and statements) in the same document order as the reference, but at different nesting depths. Every piece is present, and at least one scope boundary is in the wrong place.  
- `E_OTHER`: any other skeleton difference. This includes a missing or extra block or statement, a different order of block types, a combination of these, or a function that ends before the reference does.  

**Bounds on the structural failure rate.** Structure alone cannot tell a scope error from an omission when a function ends early: a premature closing `}` or an early dedent to column 0 gives the same skeleton as a model that leaves out its last statements. The headline rate (category 1 + category 2) is therefore the upper bound for scope-closing failures. The lower bound is category 1 + `E_PLACEMENT`. Both bounds, and the `E_OTHER` rate, are reported for every cell (depth × condition). The text discarded after the cut is kept in the raw output files so that early-end cases can be inspected by hand.

**Secondary metrics**

- Pass@1 on the unit tests (cross-check).  
- Content errors (operator, constant, predicate, or iterable differs from the reference at a matching node), reported separately for category 3 generations.

---

## 7. Boundary-Span Surprisal (main study; pilot checks feasibility only)

Surprisal is computed on the canonical reference solutions, fed to the model in each condition's syntax with the same demonstrations.

- **Span:** from the last token of the inner block's last body statement, through the first token of the next statement in the enclosing block.  
- **Summary:** total negative log-likelihood over the span, per scope exit. The outermost exit is excluded.  
- **Sensitivity:** the same calculation with the span ending before the next statement's first token.  
- **Tokenizer audit:** record how the tokenizer splits runs of 4, 8, 12, 16, and 20 spaces. Report raw and per-character surprisal.

Before the main study, confirm that the Transformers setup returns per-token log-probabilities for a supplied reference text. This is a feasibility check and is not evaluated on pilot results.

---

## 8. Pilot Health Checks

These check the pipeline, not the hypothesis. Checks on parsing and truncation are evaluated per cell (depth × condition), and the extraction hand-check per condition, because they measure whether output is read correctly. Pass-rate checks measure whether the tasks are solvable and leave room to differ, so they use the best-performing condition (floors) or pooled results (ceiling) and never a single condition's outcome. Pass rates for every cell are reported in all cases.

| Check | Target | If it fails |
| :---- | :---- | :---- |
| Pass rate, d = 1, floor | At least 40% in the best-performing condition | Simplify base tasks, or inspect prompt formatting (new version) |
| Pass rate, d = 1, ceiling | At most 90%, pooled across the three conditions | Make base tasks harder (new version) |
| Pass rate, d = 5, floor | Above 5% in the best-performing condition | Simplify base logic (new version) |
| Parse failures, d = 1 | Below 10% per cell, so at most 2 of 30. A parse failure is a category 1 generation that was not truncated: a syntax error after conversion, or an unclosed scope. Truncation is checked in its own row and is not counted here. | Read the error sub-codes first: many `E_HEADER_SYNTAX` points to the demonstrations; many `E_UNCLOSED` or `E_INDENT` points to extraction or the converter. Fix and record a new version. |
| Converter round-trip | 100% on all 90 tasks and 3 demonstrations. For each: (1) the canonical Python source is converted to B and to C; (2) both are converted back to Python with the Section 6 rules; (3) both results equal the canonical source exactly; (4) C equals B with leading whitespace removed from every line; (5) the 3 unit tests give identical outputs for the canonical source and both converted-back versions. | Stop. Fix the converter and record a new document version under the Revision rule before any model generation. |
| Scorer self-test | 100% agreement on both parts. (1) The canonical solution of every test task and demonstration, converted to each condition, scores as category 3. (2) A fixed set of hand-written negative controls, stored with the pipeline scripts and committed before the freeze, each scores as its expected category and sub-code. The controls cover: a dropped middle statement; a statement moved out of its block; an unclosed block (B and C); a Python-style header in B and C; an indentation error in A; a function closed early; and a colon before `{` (category 3 with `E_COLON_BRACE`). | Stop. Fix the scorer and record a new document version under the Revision rule before any model generation. |
| Mutant catch rate | at least 80% per task | Regenerate the task |
| Prompt length, per depth | Reported, not gated: mean token count per depth × condition. Differences are expected, since braces add tokens in B and removed indentation removes tokens in C. | Not applicable. Counts are recorded in the pilot report. |
| Truncation (extraction level) | below 5% | Investigate; do not raise the token limit silently |
| Extraction hand-check | At least 19 of 20 per condition agree with the rule. For each condition, 20 generations are sampled at random with seed 5678, stratified by depth (6 from d = 1, 7 from d = 3, 7 from d = 5). The raw outputs are shown without any cut. The researcher marks where the function ends by applying the Section 6 extraction rule, then compares the marks with the scorer's cut. Agreement means the same end position. | Fix extraction (new version) |

**Revision rule.** A failed check triggers a new version of this document with the change and its reason recorded before any new model generation. The one exception is the mutant check, which is a per-task generation filter specified in Section 3: a task that fails it is regenerated under the same rule, the regeneration count is recorded and reported, and no new document version is needed. If the pilot still fails after two revisions, stop, record the result, and reassess the design before continuing.

---

## 9. Main Study (not yet specified)

The main study will be pre-registered in a separate document after the pilot is frozen. It will cover depths 1 to 6, about 1,000 tasks, a new seed, the mixed-effects model below, and a minimum detectable effect set by simulation from pilot variance.

**Planned model (binomial mixed-effects logistic regression):**

```
logit P(failure) = b0 + b1·depth + b2·[B] + b3·[C]
                   + b4·depth·[B] + b5·depth·[C] + (1 | task)
```

Reference condition: A. C1 = b5, C2 = b4, C3 = b5 − b4. Holm correction across the three contrasts. Equivalence-based null: the 95% CI lies within ±MDE. Any result that fits neither support nor null is reported as inconclusive.

---

## 10. Interpretation Rules

- Pilot differences between conditions are not evidence about H2.  
- Do not change tasks or prompts to make an effect appear. Change them only when a health check fails.  
- Record unexpected observations as exploratory, with the date noticed.  
- Report all health-check results, including failures.

---

## 11. Known Limitations

1. Pre-training exposure: standard Python is common in training data; braced Python is not. Shared demonstrations reduce the gap but do not remove it.  
2. Depth and length are confounded. Token counts are reported per depth.  
3. The dedent and closing-brace boundaries are different token types. Surprisal is summed over the whole span to compensate.  
4. Raw-completion prompting is not how these models are usually used.  
5. One small model in greedy mode. Results do not generalize to other sizes, sampling, or reasoning modes.  
6. Synthetic tasks. External validity requires testing on natural code.  
7. Tokenizer-specific. Results describe the Qwen tokenizer until replicated.

---

## 12. Freeze Procedure

1. Complete all pipeline scripts, the fixed negative controls for the scorer self-test (Section 8), and `manifest.json` (Section 4).  
2. Update the header of this document: set `Status` to `Frozen` and update `Date written` to the freeze date.  
3. Commit this completed file, the pipeline scripts, the negative controls, and `manifest.json`.  
4. Create an annotated tag on that commit: `git tag -a prereg-pilot-frozen -m "Pilot pre-registration frozen before task and model generation"`.  
5. Deposit the exact tagged version of this file with an external timestamp service (OSF or Zenodo). Record the resulting DOI or registration URL in `manifest.json` under `external_timestamp_doi` in a follow-up metadata commit without modifying this file.  
6. Generate tasks and run model generations only after steps 1 to 5 are complete.

After the freeze, any change to conditions, tasks, prompts, metrics, or thresholds requires a new document version, with its own date and a stated reason.

---

## 13. Change Log

| Date | Version | Change |
| :---- | :---- | :---- |
| 2026-10-31 | 1.0 | Initial version. |