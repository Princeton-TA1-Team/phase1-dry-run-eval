"""Shared, auto-reused initial-sampling cache for the single-step card nodes.

drag / error-conditioning / mitigation each begin with an *identical* init-sampling
pass for a given (model, dataset): clean-prompt ``inference run`` ->
``eval --flatten_dataset`` -> ``data initial-sampling-postprocess``, producing
``processed_flattened_init_responses.ds``. Re-running that per card wastes a full
generation pass. This module centralizes the pass and caches its output under
``<init_cache_root>/<model>/<dataset>/``, keyed by the init-determining config so
reuse is automatic *and* safe:

  * cache hit (manifest matches the current init config) -> reuse, skip generation;
  * cache present but config differs (or no manifest)    -> fall back to a card-local
    init (never clobbers the shared slot, never returns mismatched data);
  * no cache                                             -> generate into the shared
    dir and write a manifest.

``init_cache_root=""`` disables sharing (card-local init == legacy behavior).

The init-sampling ``inference run`` is itself resumable (per-row sha256(prompt)
checkpoint), so an interrupted shared generation simply resumes on the next card.

NOTE: concurrent *generation* of the same (model, dataset) cache from two processes
is not locked; run init generation serially (or pre-warm the cache) when launching
parallel cards. Concurrent *reuse* (read) is always safe.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

PROCESSED_DS_NAME = "processed_flattened_init_responses.ds"
MANIFEST_NAME = "init_manifest.json"
INIT_SUBDIR = "init_inference"

# Keys that fully determine init-sampling content. Two cards that agree on all of
# these produce identical init sampling and may share the cache.
MANIFEST_KEYS = (
    "model_config", "data_path", "init_template_path", "init_template_key",
    "max_questions", "n", "max_tokens",
)


@dataclass(frozen=True)
class InitSampling:
    processed_ds: Path   # feed to `data aggregate --input_dir`
    init_dir: Path       # holds completions.jsonl + evaluated_*.jsonl (EC/mit analysis)
    reused: bool


def dataset_key(dataset, data_path) -> str:
    """Sub-path component for the cache: the benchmark name when known, else the
    init dataset filename stem (so cards that omit `dataset` still get a key)."""
    d = (str(dataset) if dataset is not None else "").strip()
    return d if d else Path(str(data_path)).stem


def shared_init_dir(init_cache_root, model_config, dataset, data_path) -> Path:
    return Path(str(init_cache_root)) / str(model_config) / dataset_key(dataset, data_path)


def build_manifest(*, model_config, data_path, init_template_path, init_template_key,
                   max_questions, n, max_tokens) -> dict:
    return {
        "model_config": str(model_config),
        "data_path": str(data_path),
        "init_template_path": str(init_template_path),
        "init_template_key": str(init_template_key),
        "max_questions": int(max_questions),
        "n": int(n),
        "max_tokens": int(max_tokens),
    }


def cache_action(processed_ds: Path, manifest_path: Path, wanted: dict) -> str:
    """Pure, unit-testable decision: 'reuse' | 'fallback' | 'generate'."""
    if processed_ds.exists() and (processed_ds / "dataset_info.json").exists():
        if not manifest_path.exists():
            return "fallback"  # untrusted: a .ds with no manifest we wrote
        try:
            have = json.loads(manifest_path.read_text())
        except Exception:
            return "fallback"
        return "reuse" if have == wanted else "fallback"
    return "generate"


def ensure_init_sampling(
    *, cdrag: List[str], work_dir: Path, init_cache_root,
    model_config, data_path, init_template_path, init_template_key,
    max_questions, n, max_tokens, gpu_memory_utilization, eval_verb,
    dataset, tensor_parallel_size: int = 1,
) -> InitSampling:
    """Return init sampling for (model, dataset), reusing the shared cache when the
    init config matches. Runs the 3-step init pass only on a miss/fallback."""
    wanted = build_manifest(
        model_config=model_config, data_path=data_path,
        init_template_path=init_template_path, init_template_key=init_template_key,
        max_questions=max_questions, n=n, max_tokens=max_tokens)

    sharing = bool(str(init_cache_root).strip())
    manifest_path = None
    if sharing:
        base = shared_init_dir(init_cache_root, model_config, dataset, data_path)
        processed_ds = base / PROCESSED_DS_NAME
        manifest_path = base / MANIFEST_NAME
        action = cache_action(processed_ds, manifest_path, wanted)
        if action == "reuse":
            print(f"[init-cache] reuse {processed_ds}", flush=True)
            return InitSampling(processed_ds, base / INIT_SUBDIR, reused=True)
        if action == "fallback":
            print(f"[init-cache] WARN: shared init at {base} has a missing/mismatched "
                  f"manifest; generating card-local init (no sharing this run).",
                  flush=True)
            base, manifest_path = work_dir, None
            processed_ds = base / PROCESSED_DS_NAME
        else:
            print(f"[init-cache] generating shared init at {base}", flush=True)
    else:
        base = work_dir
        processed_ds = base / PROCESSED_DS_NAME

    init_dir = base / INIT_SUBDIR
    init_dir.mkdir(parents=True, exist_ok=True)

    # 1. clean-prompt inference (resumable: per-row sha256(prompt) checkpoint)
    subprocess.run(cdrag + [
        "inference", "run",
        "--model_config", str(model_config),
        "--data_path", str(data_path),
        "--prompt_template_path", str(init_template_path),
        "--prompt_template_key", str(init_template_key),
        "--output_dir", str(init_dir),
        "--task_name", "init_response",
        "--max_questions", str(max_questions),
        "--n", str(n),
        "--batch_size", str(min(8, int(max_questions))),
        "--tensor_parallel_size", str(tensor_parallel_size),
        "--gpu_memory_utilization", str(gpu_memory_utilization),
        "--max_tokens", str(max_tokens),
    ], check=True)

    # 2. eval (--flatten_dataset feeds the postprocess step)
    subprocess.run(cdrag + [
        "eval", eval_verb,
        "--dataset_dir", str(init_dir),
        "--single_partition", "--n_jobs", "1",
        "--flatten_dataset",
    ], check=True)

    # 3. postprocess flattened jsonl -> processed .ds (written under <base>/)
    subprocess.run(cdrag + [
        "data", "initial-sampling-postprocess",
        "--input_dir", str(base),
        "--input_file_template", f"{INIT_SUBDIR}/*flattened.jsonl",
    ], check=True)
    if not processed_ds.exists():
        raise FileNotFoundError(f"postprocess did not create {processed_ds}")

    if manifest_path is not None:
        manifest_path.write_text(json.dumps(wanted, indent=2, sort_keys=True))

    return InitSampling(processed_ds, init_dir, reused=False)
