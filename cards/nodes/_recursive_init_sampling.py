"""Stage -1 init-sampling helper for the recursive cards.

The recursive self-improvement pipeline's Stage 0 (`pick_one_per_problem` in
`contextual_drag.recursive.pipeline.aggregate`) consumes a *flattened
init-response* dataset with `init_response_generations_metadata` /
`init_response_generations_finish_reason` / `init_response_thinking_status` /
`init_response_final` / `init_response_generations_correctness` columns. A raw
benchmark `.ds` (e.g. `data/full_data/aime24/aime24.ds`) has only
`id, problem, answer, source, ...` — pointing the recursive pipeline at it
raises `ValueError("missing required columns: …")` from Stage 0.

This helper bridges the gap. Both rf1 and naive recursive card wrappers call
`ensure_init_responses(...)` before the recursive run subprocess. If the
caller's `data_path` already has the init-response columns it's a no-op pass-
through. Otherwise the helper runs the three existing CLI verbs in sequence
(the same chain `cards/nodes/run_error_conditioning.py` and `run_ted_experiment.py`
already use) —

    contextual-drag inference run --task_name init_response …
        # writes <init_dir>/inference/completions.jsonl (packed, N samples per row)

    contextual-drag eval <math|crux|game_of_24> --flatten_dataset …
        # writes <init_dir>/inference/*flattened.jsonl (one row per (problem, sample))
        # plus per-generation `correctness` annotations

    contextual-drag data initial-sampling-postprocess --input_file_template inference/*flattened.jsonl
        # writes <init_dir>/processed_flattened_init_responses.ds

— and returns the path of the produced
`<intermediate_dir>/init_sampling/processed_flattened_init_responses.ds`.

Resume: the produced `.ds` is itself the sentinel. If a complete one already
exists under `<intermediate_dir>/init_sampling/`, the helper short-circuits.
A kill mid-`inference run` is recovered by that command's per-row prompt-hash
JSONL resume; a kill mid-eval-flatten or mid-postprocess is recovered by
re-running (both are fast and idempotent: they re-derive from the JSONL).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# Per-task template-key dispatch (mirrors the upstream
# evals_final/initial_sampling_launch_large_vllmnew.sh logic).
_TASK_TEMPLATE_KEY = {
    "crux-i":  "question_only_prompt",
    "crux-o":  "question_only_prompt",
    "gpqa":    "qa_mc_prompt",
    "mmlu":    "qa_mc_prompt",
    # aime24/25, hmmt24/25, 24-game and any future math-style task fall
    # through to the default key.
}
_DEFAULT_TEMPLATE_KEY = "qwen_math_prompt"


# Per-task eval-subgroup dispatch — picks the correct `contextual-drag eval`
# verb to flatten the inference output. `eval math` handles aime/hmmt and the
# QA-MC tasks (gpqa/mmlu) for which there is no dedicated eval verb yet; the
# math evaluator is tolerant of non-boxed multiple-choice answers (`extracted_answer`
# falls through to a string compare on the parsed final token).
_TASK_EVAL_SUBGROUP = {
    "crux-i":  "crux",
    "crux-o":  "crux",
    "24-game": "game_of_24",
}
_DEFAULT_EVAL_SUBGROUP = "math"


# Column whose presence on `data_path` indicates the .ds is already an
# init-response dataset (no Stage -1 needed). Same column the recursive
# Stage 0 aggregator filters on.
_INIT_RESPONSE_PRESENCE_COL = "init_response_generations_metadata"


def _ds_complete(ds_dir: Path) -> bool:
    """A saved HF dataset directory is considered complete iff the two
    sentinel files written at the end of `save_to_disk` are present."""
    return (ds_dir.is_dir()
            and (ds_dir / "state.json").is_file()
            and (ds_dir / "dataset_info.json").is_file())


def template_key_for_task(task: str) -> str:
    """Public for the unit test — the same dispatch used internally."""
    return _TASK_TEMPLATE_KEY.get(task, _DEFAULT_TEMPLATE_KEY)


def eval_subgroup_for_task(task: str) -> str:
    """Public for the unit test — picks the right `contextual-drag eval` verb."""
    return _TASK_EVAL_SUBGROUP.get(task, _DEFAULT_EVAL_SUBGROUP)


def ensure_init_responses(
    *,
    data_path: Path,
    intermediate_dir: Path,
    init_alias: str,
    task: str,
    init_template_path: Path,
    init_n_samples: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
) -> tuple[Path, bool]:
    """Materialize an init-response dataset if `data_path` is raw.

    Returns `(input_ds_path_to_use, init_sampling_skipped)`. The first element
    is what the recursive `--input_ds` flag should consume; the second is a
    boolean for the wrapper to echo into `result.json` (`True` if Stage -1
    was a no-op, `False` if Stage -1 actually ran or hit the resume
    sentinel).

    Raises `RuntimeError` if the postprocess subprocess returns 0 but did
    not actually produce the expected output `.ds` — defensive check to
    catch silent failures.
    """
    from datasets import load_from_disk

    data_path = Path(data_path).resolve()
    intermediate_dir = Path(intermediate_dir).resolve()

    ds = load_from_disk(str(data_path))
    if _INIT_RESPONSE_PRESENCE_COL in ds.column_names:
        print(f"[card-node init-sampling] {data_path} already has "
              f"`{_INIT_RESPONSE_PRESENCE_COL}` — pass-through.", flush=True)
        return (data_path, True)

    init_dir = intermediate_dir / "init_sampling"
    out_ds = init_dir / "processed_flattened_init_responses.ds"

    if _ds_complete(out_ds):
        print(f"[card-node init-sampling] reusing existing {out_ds} "
              f"(Stage -1 resume hit).", flush=True)
        return (out_ds, False)

    init_dir.mkdir(parents=True, exist_ok=True)
    inference_dir = init_dir / "inference"
    template_key = template_key_for_task(task)
    eval_subgroup = eval_subgroup_for_task(task)
    cdrag = [sys.executable, "-m", "contextual_drag"]

    # (a) inference run — async vLLM init sampling. Output: inference_dir/completions.jsonl.
    print(f"[card-node init-sampling] step a/c: inference run "
          f"(model={init_alias}, task={task}, template_key={template_key}, "
          f"n={init_n_samples})", flush=True)
    subprocess.run(cdrag + [
        "inference", "run",
        "--data_path", str(data_path),
        "--prompt_template_path", str(init_template_path),
        "--prompt_template_key", template_key,
        "--task_name", "init_response",
        "--model_config", init_alias,
        "--output_dir", str(inference_dir),
        "--n", str(init_n_samples),
        "--tensor_parallel_size", str(tensor_parallel_size),
        "--gpu_memory_utilization", str(gpu_memory_utilization),
    ], check=True)

    # (b) eval <subgroup> --flatten_dataset — turns the packed `completions.jsonl`
    # (one row per problem, N generations in list fields) into `*flattened.jsonl`
    # (one row per (problem, sample) with per-generation `correctness`). Required
    # by the postprocessor's row-level schema check.
    print(f"[card-node init-sampling] step b/c: eval {eval_subgroup} "
          f"--flatten_dataset (dataset_dir={inference_dir})", flush=True)
    subprocess.run(cdrag + [
        "eval", eval_subgroup,
        "--dataset_dir", str(inference_dir),
        "--single_partition", "--n_jobs", "1",
        "--flatten_dataset",
    ], check=True)

    # (c) flatten + parse-thinking + save_to_disk. Output:
    # init_dir/processed_flattened_init_responses.ds/.
    print(f"[card-node init-sampling] step c/c: postprocess "
          f"(input_dir={init_dir}, template=inference/*flattened.jsonl)",
          flush=True)
    subprocess.run(cdrag + [
        "data", "initial-sampling-postprocess",
        "--input_dir", str(init_dir),
        "--input_file_template", "inference/*flattened.jsonl",
    ], check=True)

    if not _ds_complete(out_ds):
        raise RuntimeError(
            f"[card-node init-sampling] postprocess returned 0 but the "
            f"expected output `{out_ds}` is missing or incomplete. "
            f"Inspect {init_dir} for partial artifacts.")
    return (out_ds, False)
