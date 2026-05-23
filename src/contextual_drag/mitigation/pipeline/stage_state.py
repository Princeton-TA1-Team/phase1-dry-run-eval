"""Per-stage artifact-presence checks. Each function returns True when a
stage's output is on disk and well-formed enough that the stage can be
skipped on resume.

These are the resume-decision boundaries. Every stage's first action is to
ask "is my output already there?" — if yes, skip. The checks are deliberately
strict: a partial or malformed artifact returns False and the stage re-runs.
"""

from __future__ import annotations

import json
from pathlib import Path


def ds_complete(p: Path | str) -> bool:
    """HF dataset is fully written: state.json + dataset_info.json present."""
    p = Path(p)
    return (p.is_dir()
            and (p / "state.json").is_file()
            and (p / "dataset_info.json").is_file())


def jsonl_nonempty(p: Path | str) -> bool:
    p = Path(p)
    return p.is_file() and p.stat().st_size > 0


def count_jsonl_with_hash(p: Path | str, task_name: str) -> int:
    """Number of lines that successfully parsed AND carry a prompt_hash under
    `<task_name>_generations_metadata.prompt_hash`. Matches the resume key
    written by inference_step.handle_row_with_verify.
    """
    p = Path(p)
    if not p.is_file():
        return 0
    meta_key = f"{task_name}_generations_metadata"
    n = 0
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (obj.get(meta_key) or {}).get("prompt_hash"):
                n += 1
    return n


def solve_raw_complete(jsonl_path: Path | str,
                       task_name: str,
                       expected_rows: int,
                       n_samples: int) -> bool:
    """Solve cell finished: every input problem has at least one JSONL row
    with at least one generation. Used to skip the solve stage on resume.

    n_samples is accepted for API stability but no longer constrains row size
    — the makeup loop appends rows with `makeup_batch_size` generations, which
    can differ from the main pass's `n_samples`. A row is "complete enough"
    if it has any generations and a prompt_hash; distinct prompt_hashes count
    distinct problems covered.
    """
    del n_samples  # no longer constrains row size after makeup support
    p = Path(jsonl_path)
    if not p.is_file():
        return False
    seen_hashes: set = set()
    gens_key = f"{task_name}_generations"
    meta_key = f"{task_name}_generations_metadata"
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                return False
            gens = obj.get(gens_key)
            if not isinstance(gens, list) or not gens:
                return False
            h = (obj.get(meta_key) or {}).get("prompt_hash")
            if h:
                seen_hashes.add(h)
    return len(seen_hashes) >= expected_rows


def eval_artifacts_complete(solve_dir: Path | str, model_alias: str) -> bool:
    """All three summary artifacts exist, are non-empty, and parse."""
    solve_dir = Path(solve_dir)
    eval_jsonl = solve_dir / f"evaluated_{model_alias}.jsonl"
    flat_jsonl = solve_dir / f"evaluated_{model_alias}_flattened.jsonl"
    err_json = solve_dir / f"evaluated_{model_alias}_error_analysis.json"

    for p in (eval_jsonl, flat_jsonl):
        if not p.is_file() or p.stat().st_size == 0:
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        json.loads(line)
        except Exception:
            return False
    if not err_json.is_file() or err_json.stat().st_size == 0:
        return False
    try:
        with open(err_json, "r", encoding="utf-8") as f:
            json.load(f)
    except Exception:
        return False
    return True
