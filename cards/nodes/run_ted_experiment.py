"""
Magnet pipeline node: TED (tree-edit-distance) anchored-vs-direct gap.

Subprocess chain (one long-running Python process on the host's GPU):

    inference run    (direct, clean prompt, task_name=init_response)
      -> eval math --flatten_dataset
      -> data initial-sampling-postprocess
      -> data aggregate -T 0 -F 2
      -> inference run (2F, 2f template, task_name=init_response)
      -> eval math --flatten_dataset
      -> contextual_drag analysis ted build_cache
            --phase 2f --anchored_jsonl <2F flat eval jsonl>
            --init_jsonl <direct flat eval jsonl>
            --metric tree --n_jobs 4 --cache_out cache.json
      -> contextual_drag analysis ted summarize
            --cache_in cache.json --metric tree --reduction min
            --out summary.json
      -> load summary; ted_drag = init_response - anchored_responses

Writes results.json:

    {"result": {
        "ted_drag": 1.83,
        "mean_ted_2f": 2.14, "mean_ted_direct": 3.97,
        "n_kept_problems": 27,
        "ted_metric": "tree", "ted_phase": "2f",
        "ted_reduction": "min", "aggregate_failed": false, ...
    }}

Degenerate branches: aggregate empty OR n_kept_problems==0 → ted_drag=null.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import scriptconfig as scfg


class RunTedExperimentCLI(scfg.DataConfig):
    """Sweep-able pipeline node for the TED claim card."""

    model_config = scfg.Value("Qwen3_8B_NoThinking", tags=["algo_param"])
    data_path = scfg.Value("data/24-game/24-game.ds", tags=["algo_param"])
    init_template_path = scfg.Value(
        "prompt_templates/init_response_prompt_templates.json", tags=["algo_param"])
    init_template_key = scfg.Value("qwen_math_prompt", tags=["algo_param"])
    twof_template_path = scfg.Value(
        "prompt_templates/2f_templates.json", tags=["algo_param"])
    twof_template_key = scfg.Value("2f", tags=["algo_param"])
    ted_metric = scfg.Value("tree", choices=["tree", "levenshtein", "binary"],
                              tags=["algo_param"])
    ted_reduction = scfg.Value("min",
                                 choices=["min", "mean", "max", "median"],
                                 tags=["algo_param"])
    ted_phase = scfg.Value("2f", choices=["1f", "2f"], tags=["algo_param"])
    n_jobs = scfg.Value(4, type=int, tags=["algo_param"])
    max_questions = scfg.Value(32, type=int, tags=["algo_param"])
    n = scfg.Value(8, type=int, tags=["algo_param"])
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
        twof_dir = work_dir / "twof_inference"
        agg_dir = work_dir / "aggregate"
        for d in (direct_dir, twof_dir, agg_dir):
            d.mkdir(parents=True, exist_ok=True)

        phase = str(cfg.ted_phase)
        num_false = 1 if phase == "1f" else 2

        cdrag = [sys.executable, "-m", "contextual_drag"]

        print(f"[card-node TED] step 1/8: direct inference", flush=True)
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

        print(f"[card-node TED] step 2/8: eval direct (--flatten_dataset)", flush=True)
        subprocess.run(cdrag + [
            "eval", "math",
            "--dataset_dir", str(direct_dir),
            "--single_partition", "--n_jobs", "1",
            "--flatten_dataset",
        ], check=True)

        print(f"[card-node TED] step 3/8: data initial-sampling-postprocess", flush=True)
        subprocess.run(cdrag + [
            "data", "initial-sampling-postprocess",
            "--input_dir", str(work_dir),
            "--input_file_template", "direct_inference/*flattened.jsonl",
        ], check=True)
        processed_ds = work_dir / "processed_flattened_init_responses.ds"
        if not processed_ds.exists():
            raise FileNotFoundError(f"postprocess did not create {processed_ds}")

        print(f"[card-node TED] step 4/8: data aggregate -T 0 -F {num_false}", flush=True)
        agg = subprocess.run(cdrag + [
            "data", "aggregate",
            "--input_dir", str(processed_ds),
            "--num_true", "0",
            "--num_false", str(num_false),
            "--output_dir", str(agg_dir),
            "--init_response_models", str(cfg.model_config),
        ], check=False)
        anchored_ds_path = agg_dir / f"minimal_aggregated_data_T0_F{num_false}.ds"
        if agg.returncode != 0 or not (anchored_ds_path / "dataset_info.json").exists():
            print(f"[card-node TED] aggregate produced no usable dataset (exit "
                  f"{agg.returncode}); writing degenerate results.json.", flush=True)
            _write_result(
                results_fpath, cfg, ted_drag=None,
                mean_ted_2f=None, mean_ted_direct=None,
                n_kept_problems=0, aggregate_failed=True,
            )
            return

        print(f"[card-node TED] step 5/8: 2F inference (phase={phase})", flush=True)
        n_kept = _len_dataset(anchored_ds_path)
        subprocess.run(cdrag + [
            "inference", "run",
            "--model_config", str(cfg.model_config),
            "--data_path", str(anchored_ds_path),
            "--prompt_template_path", str(cfg.twof_template_path),
            "--prompt_template_key", str(cfg.twof_template_key),
            "--output_dir", str(twof_dir),
            "--task_name", "init_response",
            "--max_questions", str(n_kept),
            "--n", str(cfg.n),
            "--batch_size", str(min(8, n_kept)),
            "--tensor_parallel_size", "1",
            "--gpu_memory_utilization", str(cfg.gpu_memory_utilization),
            "--max_tokens", str(cfg.max_tokens),
        ], check=True)

        print(f"[card-node TED] step 6/8: eval 2F (--flatten_dataset)", flush=True)
        subprocess.run(cdrag + [
            "eval", "math",
            "--dataset_dir", str(twof_dir),
            "--single_partition", "--n_jobs", "1",
            "--flatten_dataset",
        ], check=True)

        # TED needs the FLAT eval JSONL for init_jsonl (load_init_responses
        # reads init_response_generations_extracted_answer etc.). For
        # anchored, either format works (autodetected); use flat too for
        # symmetry.
        direct_init_jsonl = _latest(direct_dir, "evaluated_*_flattened.jsonl")
        anchored_jsonl = _latest(twof_dir, "evaluated_*_flattened.jsonl")
        cache_path = work_dir / "ted_cache.json"
        summary_path = work_dir / "ted_summary.json"

        print(f"[card-node TED] step 7/8: analysis ted build_cache "
              f"(metric={cfg.ted_metric}, n_jobs={cfg.n_jobs})", flush=True)
        subprocess.run(cdrag + [
            "analysis", "ted", "build_cache",
            "--phase", phase,
            "--anchored_jsonl", str(anchored_jsonl),
            "--init_jsonl", str(direct_init_jsonl),
            "--metric", str(cfg.ted_metric),
            "--n_jobs", str(cfg.n_jobs),
            "--cache_out", str(cache_path),
        ], check=True)

        print(f"[card-node TED] step 8/8: analysis ted summarize "
              f"(reduction={cfg.ted_reduction})", flush=True)
        subprocess.run(cdrag + [
            "analysis", "ted", "summarize",
            "--cache_in", str(cache_path),
            "--metric", str(cfg.ted_metric),
            "--reduction", str(cfg.ted_reduction),
            "--out", str(summary_path),
        ], check=True)

        summary = json.loads(summary_path.read_text())
        mean_ted_2f = summary.get("anchored_responses")
        mean_ted_direct = summary.get("init_response")
        n_both = int(summary.get("n_problems_with_both") or 0)

        if (mean_ted_direct is None or mean_ted_2f is None or n_both == 0):
            ted_drag = None
        else:
            ted_drag = float(mean_ted_direct) - float(mean_ted_2f)

        _write_result(
            results_fpath, cfg, ted_drag=ted_drag,
            mean_ted_2f=mean_ted_2f, mean_ted_direct=mean_ted_direct,
            n_kept_problems=n_both, aggregate_failed=False,
        )


def _write_result(results_fpath: Path, cfg, *, ted_drag, mean_ted_2f,
                  mean_ted_direct, n_kept_problems: int,
                  aggregate_failed: bool) -> None:
    payload = {
        "result": {
            "ted_drag": ted_drag,
            "mean_ted_2f": mean_ted_2f,
            "mean_ted_direct": mean_ted_direct,
            "n_kept_problems": n_kept_problems,
            "ted_metric": str(cfg.ted_metric),
            "ted_phase": str(cfg.ted_phase),
            "ted_reduction": str(cfg.ted_reduction),
            "model_config": str(cfg.model_config),
            "data_path": str(cfg.data_path),
            "aggregate_failed": aggregate_failed,
        }
    }
    with open(results_fpath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[card-node TED] wrote {results_fpath}: {payload['result']}", flush=True)


def _latest(dir_: Path, pattern: str) -> Path:
    cands = sorted(dir_.glob(pattern))
    if not cands:
        raise FileNotFoundError(f"No file matching {pattern} under {dir_}")
    return cands[-1]


def _len_dataset(path: Path) -> int:
    from datasets import load_from_disk
    return len(load_from_disk(str(path)))


def main(argv=None):
    return RunTedExperimentCLI.main(argv=argv)


if __name__ == "__main__":
    main()
