"""End-of-cell summary writer for the cm pipeline.

Reads the solve cell's JSONL — whose per-response dicts already carry
`correctness` and `extracted_answer` from streaming verification — and emits
the three artifacts the registry's `evaluated` status expects:

    evaluated_completions.jsonl              (= solve.jsonl, with verifier
                                              annotations already inlined)
    evaluated_completions_flattened.jsonl    (one row per generation)
    evaluated_completions_error_analysis.json (pass@1, pass@N, finish stats)

Filenames match the `<setting>/<task>_<modelid>/` cell layout used downstream.
Mirrors the structure of the recursive_filter1 summary writer but parameterised
by `task_name` (we use `solve_response`, matching upstream's `${STEP}_response`
JSONL key prefix).
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


def write_artifacts(cell_dir: Path,
                    solve_jsonl: Path,
                    task_name: str,
                    model_alias: str) -> dict:
    """Write the three evaluated_completions* artifacts beside `solve_jsonl`.

    Returns the summary dict (also persisted as JSON).
    """
    cell_dir = Path(cell_dir)
    solve_jsonl = Path(solve_jsonl)
    if not solve_jsonl.exists():
        raise FileNotFoundError(f"missing solve jsonl: {solve_jsonl}")

    response_column = f"{task_name}_generations"
    rows = _read_jsonl(solve_jsonl)

    eval_jsonl = cell_dir / "evaluated_completions.jsonl"
    _write_jsonl(eval_jsonl, rows)

    flat_ds = solve_jsonl_to_flat_ds(solve_jsonl, task_name=task_name)
    flat_jsonl = cell_dir / "evaluated_completions_flattened.jsonl"
    _write_jsonl(flat_jsonl, [dict(r) for r in flat_ds])

    by_id: dict = {}
    for r in rows:
        pid = r.get("id")
        by_id.setdefault(pid, []).append(r)
    n_problems = len(by_id)
    per_problem_correct: list[bool] = []
    per_problem_first: list[bool] = []
    n_samples_seen: list[int] = []
    finish_counter: Counter = Counter()
    finish_correctness: dict = {}
    extracted_none = 0
    correctness_none = 0
    total_gens = 0

    for pid, problem_rows in by_id.items():
        gens: list[dict] = []
        for r in problem_rows:
            gens.extend(r.get(response_column) or [])
        n_samples_seen.append(len(gens))
        any_correct = False
        first_correct: bool | None = None
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

    questions_all_unparsable = sum(
        1 for prs in by_id.values()
        if (lambda gens: bool(gens) and all(g.get("correctness") is None for g in gens))(
            [g for r in prs for g in (r.get(response_column) or [])])
    )
    questions_all_incorrect = sum(
        1 for prs in by_id.values()
        if (lambda gens: bool(gens) and all(g.get("correctness") is False for g in gens))(
            [g for r in prs for g in (r.get(response_column) or [])])
    )

    summary: dict[str, Any] = {
        "model": model_alias,
        "task_name": task_name,
        "n_problems": n_problems,
        "n_samples_per_problem": n_samples_per_problem,
        "total_generations": total_gens,
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
    err_json = cell_dir / "evaluated_completions_error_analysis.json"
    with open(err_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    fr = " ".join(f"{k}={v}" for k, v in sorted(finish_counter.items()))
    print(f"  pass@1={pass_at_1:.3f}  pass@{n_samples_per_problem}={pass_at_n:.3f}  "
          f"unparsable={correctness_none}/{total_gens}  finish={{{fr}}}",
          flush=True)
    return summary
