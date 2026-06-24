"""Shared 1F conditioned-inference + analysis cache (approach A).

The 1F conditioned inference (aggregate T0 F1 -> conditioned `inference run` with the
`1f` template -> eval -> `analysis error_conditioning run --setting 1f`) is the single
expensive artifact behind BOTH:
  * drag-1f          -> uses the RAW (unfiltered) metric: correctness_raw vs ..._init_sampling
  * error-cond posthoc -> uses the verdict-FILTERED metric: correctness_filtered vs ..._init_sampling

The analysis emits both in one `ec_summary.json`, so the two tests differ only in which
field they read. This module caches that artifact under
``<cond_cache_root>/<model>/<dataset>/<regime>/`` keyed by a config manifest:

  * ``ensure_cond_eval(...)``  — drag-1f calls this; computes (init->aggregate->cond
    inference->eval->analysis) on a miss, reuses on a manifest match. Returns the
    ec_summary dict + flags.
  * ``read_cond_ec_summary(...)`` — error-cond posthoc calls this; PURE READ of the
    cached ec_summary (no GPU). Returns the dict or None (cache miss) so the posthoc
    card stays analysis-only and is never launched on the GPU sweep.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

EC_SUMMARY_NAME = "ec_summary.json"
COND_MANIFEST_NAME = "cond_manifest.json"

MANIFEST_KEYS = (
    "model_config", "data_path", "cond_template_path", "cond_template_key",
    "num_false", "min_num_true_sampling", "min_num_false_sampling",
    "n", "max_tokens", "setting",
)


def _dataset_key(dataset, data_path) -> str:
    d = (str(dataset) if dataset is not None else "").strip()
    return d if d else Path(str(data_path)).stem


def cond_cache_dir(cond_cache_root, model_config, dataset, data_path, regime) -> Path:
    return (Path(str(cond_cache_root)) / str(model_config)
            / _dataset_key(dataset, data_path) / str(regime))


def build_manifest(**kw) -> dict:
    return {k: (str(kw[k]) if k in ("model_config", "data_path", "cond_template_path",
                                    "cond_template_key", "setting") else int(kw[k]))
            for k in MANIFEST_KEYS}


def read_cond_ec_summary(cond_cache_root, model_config, dataset, data_path, regime="1f") -> dict | None:
    """PURE READ (no compute): the cached ec_summary for (model, dataset, regime), or None."""
    if not str(cond_cache_root).strip():
        return None
    p = cond_cache_dir(cond_cache_root, model_config, dataset, data_path, regime) / EC_SUMMARY_NAME
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _latest(dir_: Path, pattern: str, exclude_suffix: str | None = None) -> Path | None:
    cands = sorted(dir_.glob(pattern))
    if exclude_suffix is not None:
        cands = [p for p in cands if not str(p).endswith(exclude_suffix)]
    return cands[-1] if cands else None


def ensure_cond_eval(*, cdrag, work_dir: Path, cond_cache_root, regime, setting,
                     model_config, data_path, dataset, processed_init_ds: Path,
                     direct_eval_jsonl: Path, agg_cmd, cond_template_path, cond_template_key,
                     num_false, min_num_true_sampling, min_num_false_sampling,
                     n, max_tokens, gpu_memory_utilization, tensor_parallel_size,
                     eval_verb) -> dict:
    """Ensure the 1F conditioned inference + analysis exists (shared cache); compute on
    a miss/mismatch, reuse on a manifest match. Returns:
      {"ec_summary": dict|None, "aggregate_failed": bool, "reused": bool, "cache_dir": Path}.
    """
    wanted = build_manifest(
        model_config=model_config, data_path=data_path,
        cond_template_path=cond_template_path, cond_template_key=cond_template_key,
        num_false=num_false, min_num_true_sampling=min_num_true_sampling,
        min_num_false_sampling=min_num_false_sampling, n=n, max_tokens=max_tokens,
        setting=setting)

    sharing = bool(str(cond_cache_root).strip())
    base = (cond_cache_dir(cond_cache_root, model_config, dataset, data_path, regime)
            if sharing else (work_dir / "cond_local" / str(regime)))
    summary_path = base / EC_SUMMARY_NAME
    manifest_path = base / COND_MANIFEST_NAME

    if sharing and summary_path.is_file() and manifest_path.is_file():
        try:
            have = json.loads(manifest_path.read_text())
        except Exception:
            have = None
        if have == wanted:
            print(f"[cond-cache] reuse {summary_path}", flush=True)
            try:
                return {"ec_summary": json.loads(summary_path.read_text()),
                        "aggregate_failed": False, "reused": True, "cache_dir": base}
            except Exception:
                pass  # fall through and recompute
        else:
            print(f"[cond-cache] manifest mismatch at {base}; recomputing.", flush=True)

    base.mkdir(parents=True, exist_ok=True)
    agg_dir = base / "aggregate"
    cond_dir = base / "cond_inference"
    agg_dir.mkdir(parents=True, exist_ok=True)
    cond_dir.mkdir(parents=True, exist_ok=True)

    # aggregate T0 F{num_false}
    print(f"[cond-cache] aggregate -T 0 -F {num_false} -> {agg_dir}", flush=True)
    agg = subprocess.run(cdrag + [
        "data", agg_cmd, "--input_dir", str(processed_init_ds),
        "--num_true", "0", "--num_false", str(num_false),
        "--min_num_true_sampling", str(min_num_true_sampling),
        "--min_num_false_sampling", str(min_num_false_sampling),
        "--output_dir", str(agg_dir), "--init_response_models", str(model_config),
    ], check=False)
    cond_ds = agg_dir / f"minimal_aggregated_data_T0_F{num_false}.ds"
    if agg.returncode != 0 or not (cond_ds / "dataset_info.json").exists():
        return {"ec_summary": None, "aggregate_failed": True, "reused": False, "cache_dir": base}

    # conditioned inference (1f template)
    from datasets import load_from_disk
    n_kept = len(load_from_disk(str(cond_ds)))
    print(f"[cond-cache] conditioned inference ({cond_template_key}, {n_kept} rows)", flush=True)
    subprocess.run(cdrag + [
        "inference", "run", "--model_config", str(model_config),
        "--data_path", str(cond_ds), "--prompt_template_path", str(cond_template_path),
        "--prompt_template_key", str(cond_template_key), "--output_dir", str(cond_dir),
        "--task_name", "init_response", "--max_questions", str(n_kept),
        "--n", str(n), "--batch_size", str(min(8, n_kept)),
        "--tensor_parallel_size", str(tensor_parallel_size),
        "--gpu_memory_utilization", str(gpu_memory_utilization), "--max_tokens", str(max_tokens),
    ], check=True)

    print("[cond-cache] eval conditioned", flush=True)
    subprocess.run(cdrag + [
        "eval", eval_verb, "--dataset_dir", str(cond_dir), "--single_partition", "--n_jobs", "1",
    ], check=True)

    cond_jsonl = _latest(cond_dir, "evaluated_*.jsonl", exclude_suffix="_flattened.jsonl")
    if cond_jsonl is None:
        return {"ec_summary": None, "aggregate_failed": True, "reused": False, "cache_dir": base}

    print(f"[cond-cache] analysis error_conditioning run --setting {setting}", flush=True)
    subprocess.run(cdrag + [
        "analysis", "error_conditioning", "run", "--setting", setting,
        "--cond_jsonl", str(cond_jsonl), "--direct_jsonl", str(direct_eval_jsonl),
        "--out", str(summary_path),
    ], check=True)

    manifest_path.write_text(json.dumps(wanted, indent=2, sort_keys=True))
    try:
        ec = json.loads(summary_path.read_text())
    except Exception:
        ec = None
    return {"ec_summary": ec, "aggregate_failed": ec is None, "reused": False, "cache_dir": base}
