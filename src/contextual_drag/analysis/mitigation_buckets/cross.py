"""Joint filter-vs-revise contingency on the same (problem, draft) drafts.

Filter and Revise are applied to the *same* 1F-flattened draft pool, so they
can be paired per-draft. This module computes the 2x2 contingency
{filter ✓/✗} × {revise ✓/✗} conditioned on each (direct, 1f) cell.
"""
from __future__ import annotations

from typing import Optional

REGIMES = ["drag_failed", "drag_kept", "both_failed", "drag_helps"]


def _regime_key(direct: bool, f1: bool) -> str:
    if direct and not f1:
        return "drag_failed"
    if direct and f1:
        return "drag_kept"
    if not direct and not f1:
        return "both_failed"
    return "drag_helps"


def empty_joint() -> dict:
    return {r: {"FF": 0, "FT": 0, "TF": 0, "TT": 0} for r in REGIMES}


def compute_cross(direct: dict, f1: dict, fl: dict, rv: dict, threshold: float) -> dict:
    """direct/f1/fl/rv are ``{key -> (pass_rate, n_samples)}`` maps from loaders."""
    j = empty_joint()
    n = 0
    n_filter_correct = n_revise_correct = n_both_correct = 0
    common = set(fl.keys()) & set(rv.keys()) & set(f1.keys())
    for (pid, h) in common:
        if pid not in direct:
            continue
        d_pr, _ = direct[pid]
        f_pr, _ = f1[(pid, h)]
        fl_pr, _ = fl[(pid, h)]
        rv_pr, _ = rv[(pid, h)]
        d_b  = d_pr  >= threshold
        f_b  = f_pr  >= threshold
        fl_b = fl_pr >= threshold
        rv_b = rv_pr >= threshold
        key = ("T" if fl_b else "F") + ("T" if rv_b else "F")
        j[_regime_key(d_b, f_b)][key] += 1
        n += 1
        n_filter_correct += int(fl_b)
        n_revise_correct += int(rv_b)
        n_both_correct   += int(fl_b and rv_b)

    derived: dict = {}
    for r, c in j.items():
        rn = c["FF"] + c["FT"] + c["TF"] + c["TT"]
        derived[r] = {
            "n":               rn,
            "filter_only_fix": c["TF"],
            "revise_only_fix": c["FT"],
            "both_fix":        c["TT"],
            "neither_fix":     c["FF"],
            "filter_acc":      ((c["TF"] + c["TT"]) / rn) if rn else None,
            "revise_acc":      ((c["FT"] + c["TT"]) / rn) if rn else None,
        }

    phi: Optional[float] = None
    if n > 0:
        a = sum(j[r]["TT"] for r in REGIMES)
        b = sum(j[r]["TF"] for r in REGIMES)
        c = sum(j[r]["FT"] for r in REGIMES)
        d = sum(j[r]["FF"] for r in REGIMES)
        denom_sq = (a + b) * (c + d) * (a + c) * (b + d)
        if denom_sq > 0:
            phi = (a * d - b * c) / (denom_sq ** 0.5)

    return {
        "n_observations": n,
        "regimes":        j,
        "derived":        derived,
        "filter_acc":     (n_filter_correct / n) if n else None,
        "revise_acc":     (n_revise_correct / n) if n else None,
        "agreement_rate": ((n_both_correct + sum(j[r]["FF"] for r in REGIMES)) / n) if n else None,
        "phi":            phi,
    }
