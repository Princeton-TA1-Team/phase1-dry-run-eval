"""
Magnet pipeline node: §4 mitigation (context-manipulation) experiment.

Subprocess chain (one long-running Python process on the host's GPU):

    inference run    (direct, clean prompt, task_name=init_response)
      -> eval math --flatten_dataset
      -> data initial-sampling-postprocess
      -> data aggregate -T 0 -F 2
      -> data minimal-aggregate-flatten   (2F.ds -> 1F.ds)
      -> inference run (1F, 1f template, task_name=init_response) on 1F.ds
      -> eval math
      -> contextual_drag mitigation run
            --variant cm_filter1 --model_config <X> --input_ds <1F.ds>
            --output_dir <mit_dir> --template_path <ctx_manip.json>
      -> contextual_drag analysis mitigation_buckets run
            --direct_jsonl ... --onef_jsonl ... --mit_jsonl ...
            --threshold 0.5 --variant <variant> --out summary.json
      -> load summary; recovery_rate = derived.recovery_rate

Writes results.json:

    {"result": {
        "recovery_rate": 0.40,
        "iatrogenic_rate": 0.05, "new_gain_rate": 0.15,
        "preservation_rate": 0.93, "net_vs_1f": +0.18,
        "net_vs_direct": -0.02, "drag_failed_den": 10,
        "drag_kept_den": 18, "recoverable_drag": 4,
        "persistent_drag": 6, "n_observations": 32,
        "variant": "cm_filter1", ...
    }}

Degenerate branches:
  - data aggregate empty → aggregate_failed=True, recovery_rate=null
  - drag_failed_den == 0 → recovery_rate=null (passes through analysis result)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import scriptconfig as scfg


class RunMitigationExperimentCLI(scfg.DataConfig):
    """Sweep-able pipeline node for the §4 mitigation claim card."""

    model_config = scfg.Value("Qwen3_8B_NoThinking", tags=["algo_param"])
    data_path = scfg.Value("data/gpqa/gpqa.ds", tags=["algo_param"])
    variant = scfg.Value("cm_filter1", choices=["cm_filter1", "cm_revise1"],
                          tags=["algo_param"])
    init_template_path = scfg.Value(
        "prompt_templates/init_response_prompt_templates.json", tags=["algo_param"])
    init_template_key = scfg.Value("qa_mc_prompt", tags=["algo_param"])
    onef_template_path = scfg.Value(
        "prompt_templates/1f_templates.json", tags=["algo_param"])
    onef_template_key = scfg.Value("1f", tags=["algo_param"])
    mit_template_path = scfg.Value(
        "prompt_templates/context_manipulation_templates.json", tags=["algo_param"])
    max_questions = scfg.Value(16, type=int, tags=["algo_param"])
    n = scfg.Value(8, type=int, tags=["algo_param"])
    n_samples_solve = scfg.Value(8, type=int, tags=["algo_param"])
    max_tokens = scfg.Value(2048, type=int, tags=["algo_param"])
    gpu_memory_utilization = scfg.Value(0.85, type=float, tags=["algo_param"])

    results_fpath = scfg.Value("results.json", tags=["out_path", "primary"])

    @classmethod
    def main(cls, argv=None, **kwargs):
        cfg = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)

        results_fpath = Path(cfg.results_fpath).resolve()
        results_fpath.parent.mkdir(parents=True, exist_ok=True)
        work_dir = results_fpath.parent
        direct_dir = work_dir / "direct_inference"
        onef_dir = work_dir / "onef_inference"
        agg_dir = work_dir / "aggregate"
        mit_dir = work_dir / "mitigation"
        for d in (direct_dir, onef_dir, agg_dir, mit_dir):
            d.mkdir(parents=True, exist_ok=True)
        onef_ds_path = agg_dir / "minimal_aggregated_data_T0_F1_flattend_from_F2.ds"

        cdrag = [sys.executable, "-m", "contextual_drag"]

        print(f"[card-node §4] step 1/9: direct inference", flush=True)
        subprocess.run(cdrag + [
            "inference", "run",
            "--model_config", str(cfg.model_config),
            "--data_path", str(cfg.data_path),
            "--prompt_template_path", str(cfg.init_template_path),
            "--prompt_template_key", str(cfg.init_template_key),
            "--output_dir", str(direct_dir),
            "--task_name", "init_response",
            "--max_questions", str(cfg.max_questions),
            "--n", str(cfg.n),
            "--batch_size", str(min(8, int(cfg.max_questions))),
            "--tensor_parallel_size", "1",
            "--gpu_memory_utilization", str(cfg.gpu_memory_utilization),
            "--max_tokens", str(cfg.max_tokens),
        ], check=True)

        print(f"[card-node §4] step 2/9: eval direct (--flatten_dataset)", flush=True)
        subprocess.run(cdrag + [
            "eval", "math",
            "--dataset_dir", str(direct_dir),
            "--single_partition", "--n_jobs", "1",
            "--flatten_dataset",
        ], check=True)

        print(f"[card-node §4] step 3/9: data initial-sampling-postprocess", flush=True)
        subprocess.run(cdrag + [
            "data", "initial-sampling-postprocess",
            "--input_dir", str(work_dir),
            "--input_file_template", "direct_inference/*flattened.jsonl",
        ], check=True)
        processed_ds = work_dir / "processed_flattened_init_responses.ds"
        if not processed_ds.exists():
            raise FileNotFoundError(f"postprocess did not create {processed_ds}")

        print(f"[card-node §4] step 4/9: data aggregate -T 0 -F 2", flush=True)
        agg = subprocess.run(cdrag + [
            "data", "aggregate",
            "--input_dir", str(processed_ds),
            "--num_true", "0",
            "--num_false", "2",
            "--output_dir", str(agg_dir),
            "--init_response_models", str(cfg.model_config),
        ], check=False)
        twof_ds_path = agg_dir / "minimal_aggregated_data_T0_F2.ds"
        if agg.returncode != 0 or not (twof_ds_path / "dataset_info.json").exists():
            print(f"[card-node §4] aggregate produced no usable dataset (exit "
                  f"{agg.returncode}); writing degenerate results.json.", flush=True)
            _write_result(
                results_fpath, cfg, derived=None,
                n_observations=0, aggregate_failed=True,
            )
            return

        print(f"[card-node §4] step 5/9: data minimal-aggregate-flatten "
              f"(2F.ds -> 1F.ds)", flush=True)
        subprocess.run(cdrag + [
            "data", "minimal-aggregate-flatten",
            "--input_ds_path", str(twof_ds_path),
            "--output_ds_path", str(onef_ds_path),
        ], check=True)

        n_onef = _len_dataset(onef_ds_path)
        print(f"[card-node §4] step 6/9: 1F inference on {onef_ds_path} "
              f"({n_onef} rows)", flush=True)
        subprocess.run(cdrag + [
            "inference", "run",
            "--model_config", str(cfg.model_config),
            "--data_path", str(onef_ds_path),
            "--prompt_template_path", str(cfg.onef_template_path),
            "--prompt_template_key", str(cfg.onef_template_key),
            "--output_dir", str(onef_dir),
            "--task_name", "init_response",
            "--max_questions", str(n_onef),
            "--n", str(cfg.n),
            "--batch_size", str(min(8, n_onef)),
            "--tensor_parallel_size", "1",
            "--gpu_memory_utilization", str(cfg.gpu_memory_utilization),
            "--max_tokens", str(cfg.max_tokens),
        ], check=True)

        print(f"[card-node §4] step 7/9: eval 1F", flush=True)
        subprocess.run(cdrag + [
            "eval", "math",
            "--dataset_dir", str(onef_dir),
            "--single_partition", "--n_jobs", "1",
        ], check=True)

        print(f"[card-node §4] step 8/9: mitigation run "
              f"(variant={cfg.variant})", flush=True)
        subprocess.run(cdrag + [
            "mitigation", "run",
            "--variant", str(cfg.variant),
            "--model_config", str(cfg.model_config),
            "--input_ds", str(onef_ds_path),
            "--output_dir", str(mit_dir),
            "--template_path", str(cfg.mit_template_path),
            "--n", str(cfg.n_samples_solve),
            "--tensor_parallel_size", "1",
            "--gpu_memory_utilization", str(cfg.gpu_memory_utilization),
            "--max_tokens", str(cfg.max_tokens),
            "--task_name", "mitigation_cell",
        ], check=True)
        mit_jsonl = mit_dir / "completions.jsonl"
        if not mit_jsonl.is_file():
            raise FileNotFoundError(f"mitigation did not write {mit_jsonl}")

        direct_jsonl = _latest(direct_dir, "evaluated_*.jsonl",
                                exclude_suffix="_flattened.jsonl")
        onef_jsonl = _latest(onef_dir, "evaluated_*.jsonl",
                              exclude_suffix="_flattened.jsonl")
        summary_path = work_dir / "mit_summary.json"

        print(f"[card-node §4] step 9/9: analysis mitigation_buckets run", flush=True)
        subprocess.run(cdrag + [
            "analysis", "mitigation_buckets", "run",
            "--direct_jsonl", str(direct_jsonl),
            "--onef_jsonl", str(onef_jsonl),
            "--mit_jsonl", str(mit_jsonl),
            "--threshold", "0.5",
            "--variant", str(cfg.variant),
            "--out", str(summary_path),
        ], check=True)

        summary = json.loads(summary_path.read_text())
        derived = summary.get("derived") or {}
        n_obs = int(summary.get("n_observations") or 0)
        _write_result(
            results_fpath, cfg, derived=derived,
            n_observations=n_obs, aggregate_failed=False,
        )


def _write_result(results_fpath: Path, cfg, *, derived: dict | None,
                  n_observations: int, aggregate_failed: bool) -> None:
    d = derived or {}
    payload = {
        "result": {
            "recovery_rate":      d.get("recovery_rate"),
            "iatrogenic_rate":    d.get("iatrogenic_rate"),
            "new_gain_rate":      d.get("new_gain_rate"),
            "preservation_rate":  d.get("preservation_rate"),
            "net_vs_1f":          d.get("net_vs_1f"),
            "net_vs_direct":      d.get("net_vs_direct"),
            "drag_failed_den":    int(d.get("drag_failed_den") or 0),
            "drag_kept_den":      int(d.get("drag_kept_den") or 0),
            "recoverable_drag":   _safe_int(d, "recoverable_drag"),
            "persistent_drag":    _safe_int(d, "persistent_drag"),
            "n_observations":     n_observations,
            "variant":            str(cfg.variant),
            "model_config":       str(cfg.model_config),
            "data_path":          str(cfg.data_path),
            "threshold":          0.5,
            "aggregate_failed":   aggregate_failed,
        }
    }
    with open(results_fpath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[card-node §4] wrote {results_fpath}: {payload['result']}", flush=True)


def _safe_int(d: dict, key: str) -> int:
    v = d.get(key)
    try:
        return int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0


def _latest(dir_: Path, pattern: str, *, exclude_suffix: str | None = None) -> Path:
    cands = sorted(dir_.glob(pattern))
    if exclude_suffix is not None:
        cands = [p for p in cands if not str(p).endswith(exclude_suffix)]
    if not cands:
        raise FileNotFoundError(f"No file matching {pattern} (exclude={exclude_suffix}) "
                                f"under {dir_}")
    return cands[-1]


def _len_dataset(path: Path) -> int:
    from datasets import load_from_disk
    return len(load_from_disk(str(path)))


def main(argv=None):
    return RunMitigationExperimentCLI.main(argv=argv)


if __name__ == "__main__":
    main()
