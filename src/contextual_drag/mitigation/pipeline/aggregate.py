"""Per-problem trajectory selection and strategy join.

`pick_one_per_problem` is the inlined replacement for
`stage3_generate_aggregation/aggregate_data_recursive.py` (single-N=1 path):
group rows by problem id, drop rows that don't meet completeness +
parsable-thinking + model-alias filters, and pick exactly one row per id with
a seeded RNG. Output schema: `{id, problem, answer, source, traj1, traj1_correctness}`.

`find_unsatisfied_ids` returns the subset of expected ids that have zero rows
passing the same filter set. Used to drive targeted re-generation in the
solve cell (see pipeline.py stage 2d) — without it, problems with all-bad
generations are silently dropped from the recursive cohort.

`join_strategy` is the inlined replacement for `add_strategy_column.py`:
inner-join two HF datasets on `id`, copying one strategy text column over.

Mitigation uses only `join_strategy`; the rest is ported for completeness.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from datasets import Dataset, load_from_disk


# Same cap as the upstream aggregator. Keeps round-2+ prompts from blowing up.
TRAJ_MAX_CHARS = 32768


def _resp_key(round_num: int) -> str:
    """Per-row response key prefix used by the flat dataset for a given round.
    r=0 → 'init_response'; r>=1 → 'round{r}_response'."""
    return "init_response" if round_num == 0 else f"round{round_num}_response"


def pick_one_per_problem(flat_ds_path: str | Path,
                         model_alias: str,
                         round_num: int,
                         seed: int,
                         id_column: str = "id") -> Dataset:
    """Filter, group, and sample one trajectory per problem from a flat
    response dataset.

    The input is a flattened HF dataset where each row corresponds to a single
    (problem, model, sample) triple. For round 0 the row keys come from the
    init dataset at `<init_data_root>/<task>/processed_flattened_init_responses.ds`
    (`init_response_*`). For round >=1, they're produced by `flatten.solve_jsonl_to_flat_ds`
    (`round{r}_response_*`).

    Filters applied (must all pass):
      - row's metadata `model_config_alias` matches `model_alias`
      - finish_reason == 'stop'
      - thinking_status == 'parsable_thinking'

    Output: HF Dataset with columns
      `{id, problem, answer, source, traj1, traj1_correctness}`
    where `traj1` is the post-thinking response text (truncated to 32k chars,
    matching the upstream aggregator).
    """
    ds = load_from_disk(str(flat_ds_path))

    rk = _resp_key(round_num)
    meta_col = f"{rk}_generations_metadata"
    finish_col = f"{rk}_generations_finish_reason"
    thinking_col = f"{rk}_thinking_status"
    final_col = f"{rk}_final"
    correctness_col = f"{rk}_generations_correctness"

    required = [id_column, "problem", "answer",
                meta_col, finish_col, thinking_col, final_col, correctness_col]
    missing = [c for c in required if c not in ds.column_names]
    if missing:
        raise ValueError(
            f"flat dataset at {flat_ds_path} is missing required columns: {missing}\n"
            f"have: {ds.column_names}")

    by_id: dict = defaultdict(list)
    metas = ds[meta_col]
    finishes = ds[finish_col]
    thinkings = ds[thinking_col]
    ids = ds[id_column]
    n_total = len(ds)
    kept = 0
    for i in range(n_total):
        m = metas[i] or {}
        if m.get("model_config_alias") != model_alias:
            continue
        if finishes[i] != "stop":
            continue
        if thinkings[i] != "parsable_thinking":
            continue
        by_id[ids[i]].append(i)
        kept += 1

    print(f"  aggregator: kept {kept}/{n_total} rows over {len(by_id)} problems "
          f"(model={model_alias} finish=stop thinking=parsable)", flush=True)
    if not by_id:
        raise ValueError(
            f"No rows survived filters for model={model_alias!r} round={round_num} "
            f"in {flat_ds_path}. Check that the model alias and round number match "
            f"the input data.")

    rng = np.random.default_rng(seed)
    rows = []
    for problem_id, candidate_inds in by_id.items():
        chosen = int(rng.choice(candidate_inds))
        src = ds[chosen]
        rows.append({
            "id": src[id_column],
            "problem": src["problem"],
            "answer": src["answer"],
            "source": src.get("source", ""),
            "traj1": (src[final_col] or "")[:TRAJ_MAX_CHARS],
            "traj1_correctness": src[correctness_col],
        })

    return Dataset.from_list(rows)


def find_unsatisfied_ids(flat_ds: "str | Path | Dataset",
                         model_alias: str,
                         round_num: int,
                         expected_ids: Iterable,
                         id_column: str = "id") -> set:
    """Return the subset of `expected_ids` that have zero rows passing the
    aggregator's standard filters (model_config_alias + finish_reason==stop +
    thinking_status==parsable_thinking).

    This is the cohort-shrink detector: any id returned here would silently
    disappear from the recursive cohort if we ran the aggregator as-is. The
    pipeline uses the result to drive a targeted re-generation loop.

    Accepts either a path to a saved HF dataset (loaded with load_from_disk)
    or an already-materialized Dataset. The Dataset path is used by the
    makeup loop, which holds a fresh Dataset in memory and shouldn't pay a
    disk roundtrip every iteration (and shouldn't have to special-case the
    empty-dataset save_to_disk schema-inference error).
    """
    if isinstance(flat_ds, Dataset):
        ds = flat_ds
    else:
        ds = load_from_disk(str(flat_ds))

    rk = _resp_key(round_num)
    meta_col = f"{rk}_generations_metadata"
    finish_col = f"{rk}_generations_finish_reason"
    thinking_col = f"{rk}_thinking_status"

    expected_set = set(expected_ids)
    if not expected_set:
        return set()

    if (len(ds) == 0
            or meta_col not in ds.column_names
            or finish_col not in ds.column_names
            or thinking_col not in ds.column_names):
        return set(expected_set)

    satisfied: set = set()
    metas = ds[meta_col]
    finishes = ds[finish_col]
    thinkings = ds[thinking_col]
    ids = ds[id_column]
    for i in range(len(ds)):
        m = metas[i] or {}
        if m.get("model_config_alias") != model_alias:
            continue
        if finishes[i] != "stop":
            continue
        if thinkings[i] != "parsable_thinking":
            continue
        satisfied.add(ids[i])

    return expected_set - satisfied


def join_strategy(draft_ds: Dataset, strategy_ds_path: str | Path,
                  id_column: str = "id",
                  strategy_column: str = "strategy",
                  output_column: str = "strategy") -> Dataset:
    """Add a per-problem strategy column to a draft dataset by joining on id."""
    strategy_ds = load_from_disk(str(strategy_ds_path))

    if id_column not in strategy_ds.column_names:
        raise ValueError(f"strategy ds missing id column {id_column!r}")
    if strategy_column not in strategy_ds.column_names:
        raise ValueError(f"strategy ds missing strategy column {strategy_column!r}")

    strategy_ids = list(strategy_ds[id_column])
    dups = [pid for pid, c in Counter(strategy_ids).items() if c > 1]
    if dups:
        raise ValueError(
            f"strategy ds has duplicate ids: {dups[:10]} (total {len(dups)})")

    id_to_strategy = {row[id_column]: row[strategy_column] for row in strategy_ds}

    missing = sorted({row[id_column] for row in draft_ds
                      if row[id_column] not in id_to_strategy})
    if missing:
        raise ValueError(
            f"strategy missing for {len(missing)} draft ids; first 10: {missing[:10]}")

    rows = []
    for row in draft_ds:
        new_row = dict(row)
        new_row[output_column] = id_to_strategy[row[id_column]]
        rows.append(new_row)
    return Dataset.from_list(rows)
