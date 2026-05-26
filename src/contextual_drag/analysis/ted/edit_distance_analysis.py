"""Build per-(anchored, init) jsonl TED caches and summarize them.

A cache file (JSON) is a dict ``{problem_id: entry}`` where ``entry`` keys
are stringified ``[traj1_answer, ...]`` lists and values carry
``anchored_responses`` and ``init_response`` lists, each populated with
``edit_distance.{levenshtein,tree,binary}`` per response.

The pipeline is invoked through :func:`build_cache_from_jsonls` with
explicit anchored + init jsonl paths (no external path resolution).
``summarize_cache`` reduces a cache to per-problem means.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Optional

import numpy as np
from joblib import Parallel, delayed

from contextual_drag.analysis.ted.edit_distances import edit_distance
from contextual_drag.analysis.ted.sample_preprocessing import (
    build_processed_anchored_data,
    load_init_responses,
)

# Number of in-context drafts in an anchored row for each conditioning phase.
PHASE_NUM_TRAJS: dict[str, int] = {"1f": 1, "2f": 2, "framing": 1}


def compute_edit_distance(processed_entry, metric: str = "levenshtein"):
    """Compute per-(traj_answers) edit distances against both the init-response
    and anchored-response sets.

    Stores the **raw per-draft list** ``[TED(d_1, r), ..., TED(d_N, r)]``;
    reduction (min/mean/max/median) is applied at read time by
    :func:`gather_distance_stats`.
    """
    for traj_answers in processed_entry:
        traj_parsed = ast.literal_eval(traj_answers)

        for init_response_metadata in processed_entry[traj_answers].get('init_response', []):
            init_response_answer = init_response_metadata[0]
            edit_distances = [
                float(edit_distance(traj_answer, init_response_answer, metric=metric))
                for traj_answer in traj_parsed
            ]
            init_response_metadata[1].setdefault("edit_distance", {})
            init_response_metadata[1]["edit_distance"][metric] = edit_distances

        for anchored_metadata in processed_entry[traj_answers].get('anchored_responses', []):
            anchored_answer = anchored_metadata[0]
            edit_distances = [
                float(edit_distance(traj_answer, anchored_answer, metric=metric))
                for traj_answer in traj_parsed
            ]
            anchored_metadata[1].setdefault("edit_distance", {})
            anchored_metadata[1]["edit_distance"][metric] = edit_distances

    return processed_entry


def batched_compute_edit_distance_parallel(processed_data, metric: str, n_jobs: int = 20):
    print(f"Computing edit distance ({metric}). N problems: {len(processed_data)}")
    items = list(processed_data.items())

    def _wrapper(problem_id, problem_entries):
        return (problem_id, compute_edit_distance(problem_entries, metric=metric))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_wrapper)(problem_id, problem_entries)
        for problem_id, problem_entries in items
    )
    return {problem_id: entry for problem_id, entry in results}


_DRAFT_REDUCTIONS = {
    "min":    np.min,
    "mean":   np.mean,
    "max":    np.max,
    "median": np.median,
}


def _reduce_per_draft(value, reduction: str) -> float:
    """Reduce per-draft list to scalar; legacy scalar caches pass through."""
    if isinstance(value, (int, float)):
        return float(value)
    return float(_DRAFT_REDUCTIONS[reduction](value))


def gather_distance_stats(
    entry,
    *,
    filter_stop: bool = True,
    filter_verification: bool = False,
    filter_incorrect: bool = False,
    metric: str = "tree",
    reduction: str = "min",
):
    """Apply filters in-place and reduce to per-anchor mean distance for both
    init_response and anchored_responses sets."""
    if reduction not in _DRAFT_REDUCTIONS:
        raise ValueError(
            f"unknown reduction {reduction!r}; known: {list(_DRAFT_REDUCTIONS)}"
        )
    for anchor in entry:
        for key in ("anchored_responses", "init_response"):
            remained = []
            for item in entry[anchor].get(key, []):
                if filter_stop and item[1]['finish_reason'] != 'stop':
                    continue
                if filter_verification and 'verdict' in item[1] and item[1]['verdict'] is not False:
                    continue
                if filter_incorrect and item[1]['correctness'] is not False:
                    continue
                remained.append(item)
            entry[anchor][key] = remained
            distances = [
                _reduce_per_draft(x[1]['edit_distance'][metric], reduction)
                for x in remained
            ]
            entry[anchor][f"{key}_distance_stats"] = float(np.mean(distances)) if distances else None

    anchored_means, init_means = [], []
    for anchor in entry:
        a = entry[anchor].get("anchored_responses_distance_stats")
        d = entry[anchor].get("init_response_distance_stats")
        if a is not None and d is not None:
            anchored_means.append(a)
            init_means.append(d)
    return {
        "anchored_responses": float(np.mean(anchored_means)) if anchored_means else None,
        "init_response":      float(np.mean(init_means))    if init_means    else None,
    }


def compute_stats_for_dataset(processed_data, **kwargs):
    return [gather_distance_stats(entry, **kwargs) for entry in processed_data.values()]


def _detect_anchored_fmt(anchored_jsonl: str | Path) -> str:
    """Peek at the first non-empty row: ``list`` if
    ``init_response_generations`` is a list, else ``flat``."""
    with open(anchored_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row.get("init_response_generations"), list):
                return "list"
            return "flat"
    return "flat"


def build_cache_from_jsonls(
    *,
    phase: str,
    anchored_jsonl: str | Path,
    init_jsonl: str | Path,
    metric: str = "tree",
    n_jobs: int = 4,
    cache_out: str | Path,
) -> Path:
    """Build a TED cache from explicit anchored + init jsonls and write to
    ``cache_out``. The anchored format (``flat``/``list``) is autodetected.

    ``metric="binary"`` caches only binary; anything else caches all three
    of ``levenshtein, tree, binary``.
    """
    if phase not in PHASE_NUM_TRAJS:
        raise ValueError(f"unknown phase {phase!r}; known: {list(PHASE_NUM_TRAJS)}")
    num_trajs = PHASE_NUM_TRAJS[phase]
    anchored_fmt = _detect_anchored_fmt(anchored_jsonl)

    cache_out = Path(cache_out)
    cache_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[building] {cache_out}  (n_jobs={n_jobs}, num_trajs={num_trajs}, fmt={anchored_fmt})")
    print(f"  anchored = {anchored_jsonl}")
    print(f"  init     = {init_jsonl}")
    processed = build_processed_anchored_data(
        str(anchored_jsonl), n_jobs=n_jobs, num_trajs=num_trajs, fmt=anchored_fmt,
    )
    processed = load_init_responses(processed, str(init_jsonl))

    metrics = ["binary"] if metric == "binary" else ["levenshtein", "tree", "binary"]
    for m in metrics:
        processed = batched_compute_edit_distance_parallel(
            processed, metric=m, n_jobs=n_jobs,
        )

    with open(cache_out, "w") as f:
        json.dump(processed, f)
    print(f"[done] wrote {cache_out}")
    return cache_out


def summarize_cache(
    cache_in: str | Path,
    *,
    metric: str = "tree",
    reduction: str = "min",
    filter_verification: bool = False,
) -> dict:
    """Read a TED cache file and return per-problem means.

    Returns ``{anchored_responses, init_response, n_problems_with_both,
    n_problems_anchored, n_problems_init, metric, reduction}``.
    """
    with open(cache_in, "r") as f:
        processed = json.load(f)
    stats = compute_stats_for_dataset(
        processed,
        metric=metric,
        reduction=reduction,
        filter_verification=filter_verification,
    )
    anchored = [s["anchored_responses"] for s in stats if s["anchored_responses"] is not None]
    init     = [s["init_response"]      for s in stats if s["init_response"]      is not None]
    n_both = sum(
        1 for s in stats
        if s["anchored_responses"] is not None and s["init_response"] is not None
    )
    return {
        "anchored_responses":   float(np.mean(anchored)) if anchored else None,
        "init_response":        float(np.mean(init))     if init     else None,
        "n_problems_with_both": n_both,
        "n_problems_anchored":  len(anchored),
        "n_problems_init":      len(init),
        "metric":               metric,
        "reduction":            reduction,
    }
