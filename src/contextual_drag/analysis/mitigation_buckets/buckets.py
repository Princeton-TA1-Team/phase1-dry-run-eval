"""Outcome-bucket assignment + derived metrics.

Each (problem, draft, mitigation) observation is assigned to one of 8 buckets
covering the 2^3 outcome space (direct × 1f × mitigation), each binarized
at ``threshold`` (default 0.5 — majority of n=8 samples).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

# (direct, 1f, mit) -> bucket name. Names ordered for nice display.
BUCKETS: list[tuple[str, tuple[bool, bool, bool]]] = [
    ("unaffected",       (True,  True,  True )),
    ("iatrogenic_harm",  (True,  True,  False)),
    ("recoverable_drag", (True,  False, True )),
    ("persistent_drag",  (True,  False, False)),
    ("drag_helps_kept",  (False, True,  True )),
    ("drag_helps_lost",  (False, True,  False)),
    ("new_gain",         (False, False, True )),
    ("robust_failure",   (False, False, False)),
]
BUCKET_NAMES: list[str] = [n for n, _ in BUCKETS]
_BUCKET_BY_KEY: dict[tuple[bool, bool, bool], str] = {k: n for n, k in BUCKETS}


def empty_counts() -> dict[str, int]:
    return {b: 0 for b in BUCKET_NAMES}


def assign_bucket(direct_pr: float, f1_pr: float, mit_pr: float,
                  threshold: float = 0.5) -> str:
    return _BUCKET_BY_KEY[(direct_pr >= threshold, f1_pr >= threshold, mit_pr >= threshold)]


def add_counts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {k: a.get(k, 0) + b.get(k, 0) for k in BUCKET_NAMES}


def _safe(num: float, den: float) -> Optional[float]:
    return (num / den) if den > 0 else None


def derive_metrics(counts: dict[str, int]) -> dict:
    """Derived rates from a bucket-count dict.

    See ``../mitigation_analysis/buckets.py`` for the full metric glossary.
    """
    n = sum(counts.values())
    if n == 0:
        return {"n": 0}

    c = counts
    direct_correct = c["unaffected"] + c["iatrogenic_harm"] + c["recoverable_drag"] + c["persistent_drag"]
    f1_correct     = c["unaffected"] + c["iatrogenic_harm"] + c["drag_helps_kept"] + c["drag_helps_lost"]
    mit_correct    = c["unaffected"] + c["recoverable_drag"] + c["drag_helps_kept"] + c["new_gain"]

    drag_failed_den = c["recoverable_drag"] + c["persistent_drag"]
    drag_kept_den   = c["unaffected"] + c["iatrogenic_harm"]
    direct_failed_den = c["drag_helps_kept"] + c["drag_helps_lost"] + c["new_gain"] + c["robust_failure"]
    both_failed_den   = c["new_gain"] + c["robust_failure"]
    pre_correct_den   = drag_kept_den + c["drag_helps_kept"] + c["drag_helps_lost"]
    pre_correct_kept  = c["unaffected"] + c["drag_helps_kept"]

    direct_acc = _safe(direct_correct, n)
    f1_acc     = _safe(f1_correct,     n)
    mit_acc    = _safe(mit_correct,    n)

    return {
        "n":                 n,
        "direct_acc":        direct_acc,
        "f1_acc":            f1_acc,
        "mit_acc":           mit_acc,
        "recovery_rate":     _safe(c["recoverable_drag"], drag_failed_den),
        "iatrogenic_rate":   _safe(c["iatrogenic_harm"],  drag_kept_den),
        "preservation_rate": _safe(pre_correct_kept,      pre_correct_den),
        "new_gain_rate":     _safe(c["new_gain"],         both_failed_den),
        "net_vs_1f":         (mit_acc - f1_acc) if (mit_acc is not None and f1_acc is not None) else None,
        "net_vs_direct":     (mit_acc - direct_acc) if (mit_acc is not None and direct_acc is not None) else None,
        "drag_failed_den":   drag_failed_den,
        "drag_kept_den":     drag_kept_den,
        "direct_failed_den": direct_failed_den,
        "both_failed_den":   both_failed_den,
    }


# --------------------------------------------------------------------------- local-file CLI helpers

def _md5(s: str) -> str:
    return hashlib.md5((s or "").encode()).hexdigest()


def _row_pass_rate(row: dict, gen_col: str) -> Optional[float]:
    correct = total = 0
    for g in row.get(gen_col, []) or []:
        c = g.get("correctness") if isinstance(g, dict) else None
        if c is not None:
            correct += int(bool(c))
            total += 1
    return (correct / total) if total else None


def _load_traj_keyed(path: str | Path, gen_col: str) -> dict[tuple[str, str], float]:
    """``{(problem_id, md5(traj1)): pass_rate}`` for a draft-keyed jsonl."""
    out: dict[tuple[str, str], float] = {}
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                continue
            pr = _row_pass_rate(row, gen_col)
            if pr is None:
                continue
            out[(row["id"], _md5(row.get("traj1", "")))] = pr
    return out


def _load_id_keyed(path: str | Path, gen_col: str) -> dict[str, float]:
    """``{problem_id: pass_rate}`` for the direct jsonl."""
    out: dict[str, float] = {}
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                continue
            pr = _row_pass_rate(row, gen_col)
            if pr is None:
                continue
            out[row["id"]] = pr
    return out


def count_observations_from_jsonls(
    *,
    direct_jsonl: str | Path,
    onef_jsonl: str | Path,
    mit_jsonl: str | Path,
    threshold: float = 0.5,
) -> tuple[dict[str, int], int]:
    """Load three jsonls, pair on (id, md5(traj1)), and accumulate bucket counts.

    Returns ``(counts_dict, n_observations)``.
    """
    direct = _load_id_keyed(direct_jsonl, "init_response_generations")
    onef   = _load_traj_keyed(onef_jsonl, "init_response_generations")
    mit    = _load_traj_keyed(mit_jsonl, "solve_response_generations")

    counts = empty_counts()
    n = 0
    for (pid, h), mit_pr in mit.items():
        if pid not in direct:
            continue
        if (pid, h) not in onef:
            continue
        d_pr = direct[pid]
        f_pr = onef[(pid, h)]
        b = assign_bucket(d_pr, f_pr, mit_pr, threshold)
        counts[b] += 1
        n += 1
    return counts, n
