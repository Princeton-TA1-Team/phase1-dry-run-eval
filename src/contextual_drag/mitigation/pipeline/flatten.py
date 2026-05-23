"""Flatteners for inference JSONL outputs.

`inference_jsonl_to_ds` handles n=1 cells (strategy, filter1, revise1): one
row in → one row out, with the model's response parsed to post-thinking text
under a chosen column name.

`solve_jsonl_to_flat_ds` handles the n=N solve cell: one row in → N rows out
(one per generation), each with `<task_name>_generations_<key>` columns
matching the schema upstream `eval.py --flatten` produces. Parameterised by
`task_name` (mitigation uses `solve_response`, matching upstream
`auto_context_manipulation_lib.sh`'s `${STEP}_response` convention).

`parse_thinking_steps` recognises three formats:

1. gpt-oss harmony WITH special tokens kept (vLLM's
   `skip_special_tokens=False`): post-thinking text is everything after the
   LAST `<|channel|>final<|message|>` marker, with `<|return|>` / `<|end|>`
   stripped.
2. gpt-oss harmony WITH special tokens stripped: post-thinking text is
   everything after the LAST `assistantfinal`.
3. DeepSeek/Qwen `<think>...</think>`: take the segment after the last
   `</think>`; mismatched counts across prompt+response → `malformed_thinking`.

If none match, tag `no_thinking` and return the response unchanged. Long
unparsable responses get truncated and prefixed `truncated_`.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset


_HARMONY_FINAL = "<|channel|>final<|message|>"
_HARMONY_ANALYSIS = "<|channel|>analysis<|message|>"
_HARMONY_END_TOKENS = ("<|return|>", "<|end|>", "<|endoftext|>")


def parse_thinking_steps(response: str, prompt: str,
                         max_response_length: int = 16384) -> tuple[str, str]:
    if _HARMONY_FINAL in response:
        final_part = response.split(_HARMONY_FINAL)[-1]
        for tok in _HARMONY_END_TOKENS:
            if final_part.endswith(tok):
                final_part = final_part[: -len(tok)]
        return final_part.strip(), "parsable_thinking"
    if _HARMONY_ANALYSIS in response:
        return response, "malformed_thinking"

    if response.startswith("analysis") and len(response) > 7 and response[7] != " ":
        if "assistantfinal" in response:
            return response.split("assistantfinal")[-1], "parsable_thinking"
        return response, "malformed_thinking"

    if "<think>" not in (prompt + response):
        return response, "no_thinking"

    non_thinking_response = response.split("</think>")[-1]
    concatenated = prompt + response
    if concatenated.count("<think>") != concatenated.count("</think>"):
        thinking_status = "malformed_thinking"
    else:
        thinking_status = "parsable_thinking"

    if (len(non_thinking_response) > max_response_length
            and thinking_status != "parsable_thinking"):
        non_thinking_response = non_thinking_response[:max_response_length]
        thinking_status = "truncated_" + thinking_status

    return non_thinking_response, thinking_status


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

    Used after the strategy, filter1, and revise1 cells. Each row of the JSONL
    must have `<task_name>_generations` as a list with at least one element;
    the first element's `generated_response` is parsed for thinking and stored
    under `<final_column>` (e.g. 'strategy' or 'filtered_traj1'). The parser's
    status is stored under `<task_name>_thinking_status`.
    """
    src_rows = _read_jsonl(jsonl_path)
    out_rows = []
    for entry in src_rows:
        gens = entry.get(f"{task_name}_generations") or []
        if not gens:
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
                           task_name: str,
                           max_response_length: int = 16384) -> Dataset:
    """Flatten a solve JSONL (n=N samples per row) to one row per generation.

    Mirrors the schema produced by upstream `eval.py --flatten` for a step
    whose `--task-name` was `task_name`: each output row carries
    `<task_name>_generations_<key>` columns plus the parsed-thinking columns.
    """
    rk = task_name
    src_rows = _read_jsonl(jsonl_path)
    out_rows = []
    for entry in src_rows:
        gens = entry.get(f"{rk}_generations") or []
        if not gens:
            continue
        prompt = entry.get(f"{rk}_prompt", "")
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
