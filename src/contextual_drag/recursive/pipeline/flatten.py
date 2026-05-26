"""Convert inference JSONL outputs into the HF datasets the next stage expects.

`inference_jsonl_to_ds` handles n=1 cells (strategy, filter1): one row in →
one row out, with the model's response parsed to post-thinking text under a
chosen column name. Replaces `format_multiturn_data/postprocess.py`.

`solve_jsonl_to_flat_ds` handles n=N solve cells: one row in → N rows out
(one per generation), each with `round{r}_response_generations_<key>` columns
(matching the schema the upstream `eval.py` would produce with `--flatten`),
plus parsed-thinking columns for downstream aggregation. This is the input
the next round's `aggregate.pick_one_per_problem` reads.

Both functions assume the JSONL was written by our inference driver
(or the upstream one), so each row carries `<task_name>_prompt`,
`<task_name>_generations` (list of dicts), and `<task_name>_generations_metadata`.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from .thinking import parse_thinking_steps


def _read_jsonl(path: str | Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def inference_jsonl_to_ds(jsonl_path: str | Path,
                          task_name: str,
                          final_column: str,
                          max_response_length: int = 16384) -> Dataset:
    """Flatten an n=1 inference JSONL into an HF dataset, parsing thinking.

    Used after the strategy and filter1 cells. Each row of the JSONL must have
    `<task_name>_generations` as a list with at least one element; the first
    element's `generated_response` is parsed for thinking and stored under
    `<final_column>` (e.g. 'strategy' or 'filtered_traj1'). The parser's
    status is stored under `<task_name>_thinking_status`.

    The original record is preserved (so downstream code can still see the
    raw generation, prompt, metadata).
    """
    src_rows = _read_jsonl(jsonl_path)
    out_rows = []
    for entry in src_rows:
        gens = entry.get(f"{task_name}_generations") or []
        if not gens:
            # Skip rows where the inference cell produced nothing for this row.
            # The prompt-hash resume guarantees we'll re-generate later if needed.
            continue
        prompt = entry.get(f"{task_name}_prompt", "")
        text = gens[0].get("generated_response", "")
        post, status = parse_thinking_steps(text, prompt, max_response_length)
        new_row = dict(entry)
        new_row[final_column] = post
        new_row[f"{task_name}_thinking_status"] = status
        out_rows.append(new_row)
    return Dataset.from_list(out_rows)


def solve_jsonl_to_flat_ds(jsonl_path: str | Path,
                           round_num: int,
                           max_response_length: int = 16384) -> Dataset:
    """Flatten an n=N solve JSONL to one row per generation, with the schema
    the next round's aggregator expects.

    Round t solve JSONL row schema (per row):
        round{t}_response_prompt: str
        round{t}_response_generations: list[{response_id, generated_response,
            finish_reason, correctness, extracted_answer}]
        round{t}_response_generations_metadata: {model_config_alias, ...}
        ...plus problem fields (id, problem, answer, source, ...)

    Output (one row per generation) has, for each k in the per-response dict:
        round{t}_response_generations_{k}
    e.g. `round{t}_response_generations_correctness` (single value, not a list).
    Plus `round{t}_response_final` and `round{t}_response_thinking_status` from
    the parsed thinking. This is exactly what `aggregate.pick_one_per_problem`
    consumes when called with `round_num=t`.
    """
    rk = f"round{round_num}_response"
    src_rows = _read_jsonl(jsonl_path)
    out_rows = []
    for entry in src_rows:
        gens = entry.get(f"{rk}_generations") or []
        if not gens:
            continue
        prompt = entry.get(f"{rk}_prompt", "")
        # Strip the per-row generations list from the carry-over fields so the
        # flattened rows don't duplicate the giant list on every output row.
        carry = {k: v for k, v in entry.items()
                 if k != f"{rk}_generations"}
        for g in gens:
            flat_row = dict(carry)
            for gk, gv in g.items():
                flat_row[f"{rk}_generations_{gk}"] = gv
            text = g.get("generated_response", "")
            post, status = parse_thinking_steps(text, prompt, max_response_length)
            flat_row[f"{rk}_final"] = post
            flat_row[f"{rk}_thinking_status"] = status
            out_rows.append(flat_row)
    return Dataset.from_list(out_rows)
