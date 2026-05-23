"""Per-cell metric computation for §3 error conditioning.

For each conditioned + direct jsonl pair we compute four numbers, all
averaged at the *problem* level (matching
``analysis/correct_verification_conditioning/analysis.py``):

  correctness_raw                     — follow-up accuracy on every conditioning
                                        sample (no verdict filter)
  correctness_filtered                — follow-up accuracy on samples that
                                        survived the verdict filter
                                        (== correctness_raw for framing)
  correctness_raw_init_sampling       — direct-baseline accuracy on the *same
                                        problems* that appeared in the conditioned
                                        set (matches denominator for raw)
  correctness_filtered_init_sampling  — direct-baseline accuracy on the problems
                                        that survived the verdict filter

The "delta" plotted in figures is then either:
  - external (framing): correctness_raw - correctness_raw_init_sampling
  - internal (1f/2f) : correctness_filtered - correctness_filtered_init_sampling
"""
from __future__ import annotations

import json
from typing import Iterable, Optional

import numpy as np

from contextual_drag.analysis.error_conditioning.verification_parsing import (
    annotate_responses_1f,
    annotate_responses_2f,
    filter_responses_1f,
    filter_responses_2f,
)


# --------------------------------------------------------------------------- loaders

def _load_jsonl(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # Tolerate a torn last line (the inference appender writes
                # append+flush+fsync per row but a torn write is possible
                # under preemption).
                continue
    return out


def _merge_ids(entries: Iterable[dict]) -> list[dict]:
    """Collapse duplicate-id records by concatenating their generations."""
    by_id: dict[str, dict] = {}
    for entry in entries:
        eid = entry["id"]
        if eid not in by_id:
            by_id[eid] = dict(entry)
            by_id[eid]["init_response_generations"] = list(
                entry.get("init_response_generations", [])
            )
        else:
            by_id[eid]["init_response_generations"].extend(
                entry.get("init_response_generations", [])
            )
    return list(by_id.values())


def _load_merged(path: str) -> list[dict]:
    return _merge_ids(_load_jsonl(path))


# --------------------------------------------------------------------------- metrics

def _problem_correctness(entries: list[dict]) -> list[float]:
    """Per-problem mean correctness over its responses."""
    out: list[float] = []
    for e in entries:
        gens = e.get("init_response_generations", [])
        if not gens:
            continue
        out.append(float(np.mean([g.get("correctness") is True for g in gens])))
    return out


def _mean(xs: list[float]) -> Optional[float]:
    return float(np.mean(xs)) if xs else None


def compute_cell_from_jsonls(
    *,
    setting: str,
    cond_jsonl: str,
    direct_jsonl: str,
) -> dict:
    """Compute §3 conditioning metrics from local jsonls.

    ``setting`` ∈ ``{"framing", "1f", "2f"}``. Returns a dict with at minimum
    correctness_raw, correctness_raw_init_sampling, correctness_filtered,
    correctness_filtered_init_sampling, num_problems_cond, num_problems_filtered.
    """
    if setting not in ("1f", "2f", "framing"):
        raise ValueError(f"Unsupported conditioning setting: {setting}")

    cond = _load_merged(cond_jsonl)
    direct = _load_merged(direct_jsonl)
    direct_by_id = {e["id"]: e for e in direct}

    # Restrict the direct baseline to problems that also appear in the
    # conditioned set (which may be a subset, e.g. only problems with at
    # least one incorrect draft).
    cond_ids = {e["id"] for e in cond}
    direct_matched = [direct_by_id[i] for i in cond_ids if i in direct_by_id]

    if setting == "framing":
        cond_filtered = cond
        direct_filtered_set = direct_matched
    else:
        annotate = annotate_responses_1f if setting == "1f" else annotate_responses_2f
        do_filter = filter_responses_1f if setting == "1f" else filter_responses_2f
        cond_annotated = [annotate(dict(e)) for e in cond]
        cond_with_valid = [e for e in cond_annotated if e.get("valid_verdict")]
        cond_filtered = [do_filter(dict(e)) for e in cond_with_valid]
        cond_filtered = [e for e in cond_filtered if e["init_response_generations"]]
        kept_ids = {e["id"] for e in cond_filtered}
        direct_filtered_set = [direct_by_id[i] for i in kept_ids if i in direct_by_id]

    raw_corr_cond = _problem_correctness(cond)
    filt_corr_cond = _problem_correctness(cond_filtered)
    raw_corr_direct = _problem_correctness(direct_matched)
    filt_corr_direct = _problem_correctness(direct_filtered_set)

    return {
        "correctness_raw":                    _mean(raw_corr_cond),
        "correctness_filtered":               _mean(filt_corr_cond),
        "correctness_raw_init_sampling":      _mean(raw_corr_direct),
        "correctness_filtered_init_sampling": _mean(filt_corr_direct),
        "num_problems_cond":                  len(cond),
        "num_problems_filtered":              len(cond_filtered),
        "num_problems_direct_matched":        len(direct_matched),
        "num_problems_direct_filtered":       len(direct_filtered_set),
        "cond_path":                          str(cond_jsonl),
        "direct_path":                        str(direct_jsonl),
        "setting":                            setting,
    }
