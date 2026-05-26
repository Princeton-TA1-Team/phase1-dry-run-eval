"""End-of-round summary writer.

Reads a solve JSONL whose per-response dicts already carry `correctness` and
`extracted_answer` (populated inline by `verifier.Verifier`) and emits the
three artifacts the upstream `eval.py` would have written:

    evaluated_<model>.jsonl              (same content; correctness already there)
    evaluated_<model>_flattened.jsonl    (one row per generation, with
                                          `<response_column>_<key>` keys —
                                          the input shape `flatten.solve_jsonl_to_flat_ds`
                                          produces, written via .to_json)
    evaluated_<model>_error_analysis.json (pass@1, pass@N, finish_reason stats)

We re-use `flatten.solve_jsonl_to_flat_ds` for the flattened file rather than
re-implement the same flattening twice. The error-analysis JSON is a minimal
dict with the keys downstream code might inspect; we don't reproduce the
full upstream pass-at-k sliding-window analysis since AIME is single-source
and we just need pass@1 / pass@N.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .flatten import solve_jsonl_to_flat_ds


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_artifacts(solve_dir: Path,
                    model_alias: str,
                    round_num: int) -> dict:
    """Write the three evaluated_* artifacts and return the summary dict.

    The solve JSONL may contain multiple rows per problem id when the makeup
    loop appended additional samples for problems that lacked parsable+stop
    coverage on the main pass. We group by id when computing per-problem
    metrics so pass@N reflects "any correct across all samples for this
    problem" rather than "any correct in this JSONL row".
    """
    solve_dir = Path(solve_dir)
    solve_jsonl = solve_dir / "dataset.jsonl"
    if not solve_jsonl.exists():
        raise FileNotFoundError(f"missing solve jsonl: {solve_jsonl}")

    rk = f"round{round_num}_response"
    response_column = f"{rk}_generations"

    rows = _read_jsonl(solve_jsonl)

    # 1. evaluated_<model>.jsonl — same shape as solve.jsonl since correctness
    #    is already inlined per generation by the streaming verifier.
    eval_jsonl = solve_dir / f"evaluated_{model_alias}.jsonl"
    _write_jsonl(eval_jsonl, rows)

    # 2. evaluated_<model>_flattened.jsonl — one row per generation, with
    #    `<response_column>_<key>` columns and parsed-thinking columns.
    flat_ds = solve_jsonl_to_flat_ds(solve_jsonl, round_num=round_num)
    flat_jsonl = solve_dir / f"evaluated_{model_alias}_flattened.jsonl"
    _write_jsonl(flat_jsonl, [dict(r) for r in flat_ds])

    # 3. evaluated_<model>_error_analysis.json — small summary.
    # Group rows by problem id so makeup rows merge into the same problem.
    by_id: dict = {}
    for r in rows:
        pid = r.get("id")
        by_id.setdefault(pid, []).append(r)
    n_problems = len(by_id)
    n_makeup_rows = len(rows) - n_problems
    per_problem_correct = []     # any correct across all samples (pass@N)
    per_problem_first = []       # first sample correctness (pass@1 proxy)
    n_samples_seen = []
    finish_counter: Counter = Counter()
    finish_correctness: dict = {}
    extracted_none = 0
    correctness_none = 0
    total_gens = 0

    for pid, problem_rows in by_id.items():
        gens = []
        for r in problem_rows:
            gens.extend(r.get(response_column) or [])
        n_samples_seen.append(len(gens))
        any_correct = False
        first_correct = None
        for g in gens:
            total_gens += 1
            ok = g.get("correctness")
            fr = g.get("finish_reason")
            finish_counter[fr] = finish_counter.get(fr, 0) + 1
            slot = finish_correctness.setdefault(
                str(fr), {"correct": 0, "incorrect": 0, "unparsable": 0})
            if ok is True:
                slot["correct"] += 1
                any_correct = True
            elif ok is False:
                slot["incorrect"] += 1
            else:
                slot["unparsable"] += 1
                correctness_none += 1
            if g.get("extracted_answer") is None:
                extracted_none += 1
            if first_correct is None:
                first_correct = ok
        per_problem_correct.append(any_correct)
        per_problem_first.append(first_correct is True)

    pass_at_1 = (sum(per_problem_first) / n_problems) if n_problems else 0.0
    pass_at_n = (sum(per_problem_correct) / n_problems) if n_problems else 0.0
    n_samples_per_problem = max(n_samples_seen) if n_samples_seen else 0

    def _all_gens(problem_rows: list[dict]) -> list[dict]:
        out: list[dict] = []
        for r in problem_rows:
            out.extend(r.get(response_column) or [])
        return out

    questions_all_unparsable = sum(
        1 for prs in by_id.values()
        if (lambda gens: bool(gens) and all(g.get("correctness") is None for g in gens))(_all_gens(prs))
    )
    questions_all_incorrect = sum(
        1 for prs in by_id.values()
        if (lambda gens: bool(gens) and all(g.get("correctness") is False for g in gens))(_all_gens(prs))
    )

    summary: dict[str, Any] = {
        "model": model_alias,
        "round_num": round_num,
        "n_problems": n_problems,
        "n_samples_per_problem": n_samples_per_problem,
        "total_generations": total_gens,
        "n_makeup_rows": n_makeup_rows,
        "pass_at_1": pass_at_1,
        f"pass_at_{n_samples_per_problem}": pass_at_n,
        "questions_with_all_unparsable": questions_all_unparsable,
        "questions_with_all_incorrect": questions_all_incorrect,
        "unparsable_responses_count": extracted_none,
        "finish_reason_stats": dict(finish_counter),
        "finish_reason_correctness": finish_correctness,
        "overall_stats": {
            "pass_at_1": pass_at_1,
            f"pass_at_{n_samples_per_problem}": pass_at_n,
            "n_problems": n_problems,
        },
    }

    err_json = solve_dir / f"evaluated_{model_alias}_error_analysis.json"
    with open(err_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    fr = " ".join(f"{k}={v}" for k, v in sorted(finish_counter.items()))
    makeup_note = f"  makeup_rows={n_makeup_rows}" if n_makeup_rows else ""
    print(f"  pass@1={pass_at_1:.3f}  pass@{n_samples_per_problem}={pass_at_n:.3f}  "
          f"unparsable={correctness_none}/{total_gens}  finish={{{fr}}}{makeup_note}",
          flush=True)
    return summary
