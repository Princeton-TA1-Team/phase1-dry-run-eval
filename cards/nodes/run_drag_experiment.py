"""
Magnet pipeline node: run the full contextual-drag pipeline as a single node.

Internally subprocesses the canonical 6-step chain:
    inference run (clean prompt)
      -> eval math --flatten_dataset
      -> data initial-sampling-postprocess
      -> data aggregate -T num_true -F num_false
      -> inference run (2F prompt)
      -> eval math --response_column twof_generations

Computes:
    acc_clean : init-inference accuracy *restricted to the problems that
                survived the aggregate filter*, so it's apples-to-apples
                with acc_2f.
    acc_2f    : 2F-inference overall accuracy.
    drag      : acc_clean - acc_2f.

Writes a results.json magnet's GenericPipelineProcessor consumes:

    {"result": {
        "acc_clean": 0.40,
        "acc_2f": 0.125,
        "drag": 0.275,
        "n_kept_problems": 5,
        ...
    }}

If `data aggregate` filters out all problems, the wrapper still writes
results.json with `drag: null` and `n_kept_problems: 0` so the claim can
emit a legible message rather than crashing.
"""

from __future__ import annotations

import glob
import json
import subprocess
import sys
from pathlib import Path

import scriptconfig as scfg


class RunDragExperimentCLI(scfg.DataConfig):
    """Sweep-able pipeline node for the contextual-drag claim card."""

    model_config = scfg.Value(
        "Qwen3_8B_NoThinking",
        help="Alias from contextual_drag/resources/inference/eval_models_params.json.",
        tags=["algo_param"],
    )
    data_path = scfg.Value(
        "data/gpqa/gpqa.ds",
        help="HF dataset directory (gpqa works well; math500 is too easy and aime24 too hard for Qwen3_8B).",
        tags=["algo_param"],
    )
    init_template_path = scfg.Value(
        "prompt_templates/init_response_prompt_templates.json",
        help="JSON file with the clean-prompt templates.",
        tags=["algo_param"],
    )
    init_template_key = scfg.Value(
        "qa_mc_prompt",
        help="Template key for the clean-prompt round (qa_mc_prompt for gpqa, qwen_math_prompt for math).",
        tags=["algo_param"],
    )
    twof_template_path = scfg.Value(
        "prompt_templates/2f_templates.json",
        help="JSON file containing the 2F-augmented templates.",
        tags=["algo_param"],
    )
    twof_template_key = scfg.Value(
        "2f",
        help="Template key for the 2F round (consumes {arg_problem}, {arg_traj1}, {arg_traj2}).",
        tags=["algo_param"],
    )
    max_questions = scfg.Value(
        8,
        type=int,
        help="Cap dataset rows for fast cards. 8 was sufficient on gpqa to pass the aggregate filter.",
        tags=["algo_param"],
    )
    n = scfg.Value(
        8,
        type=int,
        help="Samples per question on the clean-prompt round (also used for 2F).",
        tags=["algo_param"],
    )
    max_tokens = scfg.Value(
        2048,
        type=int,
        help="Generation budget per response.",
        tags=["algo_param"],
    )
    gpu_memory_utilization = scfg.Value(
        0.85,
        type=float,
        help="vLLM GPU memory fraction.",
        tags=["algo_param"],
    )
    num_true = scfg.Value(
        0,
        type=int,
        help="`-T` for `data aggregate` (number of correct trajectories injected per 2F prompt).",
        tags=["algo_param"],
    )
    num_false = scfg.Value(
        2,
        type=int,
        help="`-F` for `data aggregate` (number of failed trajectories injected per 2F prompt).",
        tags=["algo_param"],
    )

    results_fpath = scfg.Value(
        "results.json",
        help=(
            "Path the magnet pipeline reads to populate symbols. Magnet "
            "rewrites this to a per-sweep hashed directory."
        ),
        tags=["out_path", "primary"],
    )

    @classmethod
    def main(cls, argv=None, **kwargs):
        cfg = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)

        results_fpath = Path(cfg.results_fpath).resolve()
        results_fpath.parent.mkdir(parents=True, exist_ok=True)
        work_dir = results_fpath.parent
        init_dir = work_dir / "init_inference"
        twof_dir = work_dir / "twof_inference"
        aggregate_dir = work_dir / "aggregate"
        init_dir.mkdir(parents=True, exist_ok=True)
        twof_dir.mkdir(parents=True, exist_ok=True)
        aggregate_dir.mkdir(parents=True, exist_ok=True)

        cdrag = [sys.executable, "-m", "contextual_drag"]

        # 1. Clean-prompt inference
        print("[card-node] step 1/6: clean-prompt inference", flush=True)
        subprocess.run(
            cdrag + [
                "inference", "run",
                "--model_config", str(cfg.model_config),
                "--data_path", str(cfg.data_path),
                "--prompt_template_path", str(cfg.init_template_path),
                "--prompt_template_key", str(cfg.init_template_key),
                "--output_dir", str(init_dir),
                "--task_name", "init_response",
                "--max_questions", str(cfg.max_questions),
                "--n", str(cfg.n),
                "--batch_size", str(min(8, int(cfg.max_questions))),
                "--tensor_parallel_size", "1",
                "--gpu_memory_utilization", str(cfg.gpu_memory_utilization),
                "--max_tokens", str(cfg.max_tokens),
            ],
            check=True,
        )

        # 2. Eval (with --flatten_dataset for the postprocess step)
        print("[card-node] step 2/6: eval clean", flush=True)
        subprocess.run(
            cdrag + [
                "eval", "math",
                "--dataset_dir", str(init_dir),
                "--single_partition", "--n_jobs", "1",
                "--flatten_dataset",
            ],
            check=True,
        )

        # 3. Postprocess. Default glob is */*/*flattened.jsonl (3 deep);
        #    our flattened file lives at init_inference/evaluated_*_flattened.jsonl
        #    so we override the template to match that 1-deep layout.
        print("[card-node] step 3/6: postprocess flattened jsonl", flush=True)
        subprocess.run(
            cdrag + [
                "data", "initial-sampling-postprocess",
                "--input_dir", str(work_dir),
                "--input_file_template", "init_inference/*flattened.jsonl",
            ],
            check=True,
        )
        processed_ds = work_dir / "processed_flattened_init_responses.ds"
        if not processed_ds.exists():
            raise FileNotFoundError(f"postprocess did not create {processed_ds}")

        # 4. Aggregate. May exit 1 if 0 problems pass the filter — capture
        #    that as a legible card outcome rather than letting magnet bomb.
        print("[card-node] step 4/6: aggregate (build 2F dataset)", flush=True)
        agg = subprocess.run(
            cdrag + [
                "data", "aggregate",
                "--input_dir", str(processed_ds),
                "--num_true", str(cfg.num_true),
                "--num_false", str(cfg.num_false),
                "--output_dir", str(aggregate_dir),
                "--init_response_models", str(cfg.model_config),
            ],
            check=False,
        )
        twof_ds_path = (
            aggregate_dir / f"minimal_aggregated_data_T{cfg.num_true}_F{cfg.num_false}.ds"
        )
        if agg.returncode != 0 or not (twof_ds_path / "dataset_info.json").exists():
            print(
                f"[card-node] aggregate produced no usable dataset (exit {agg.returncode}); "
                f"writing degenerate results.json so the claim can fail legibly.",
                flush=True,
            )
            _write_result(
                results_fpath,
                acc_clean=None,
                acc_2f=None,
                drag=None,
                n_kept_problems=0,
                cfg=cfg,
                aggregate_failed=True,
            )
            return

        # 5. 2F inference on the aggregated dataset
        print("[card-node] step 5/6: 2F inference", flush=True)
        n_kept = _len_dataset(twof_ds_path)
        subprocess.run(
            cdrag + [
                "inference", "run",
                "--model_config", str(cfg.model_config),
                "--data_path", str(twof_ds_path),
                "--prompt_template_path", str(cfg.twof_template_path),
                "--prompt_template_key", str(cfg.twof_template_key),
                "--output_dir", str(twof_dir),
                "--task_name", "twof",
                "--max_questions", str(n_kept),
                "--n", str(cfg.n),
                "--batch_size", str(min(8, n_kept)),
                "--tensor_parallel_size", "1",
                "--gpu_memory_utilization", str(cfg.gpu_memory_utilization),
                "--max_tokens", str(cfg.max_tokens),
            ],
            check=True,
        )

        # 6. Eval 2F
        print("[card-node] step 6/6: eval 2F", flush=True)
        subprocess.run(
            cdrag + [
                "eval", "math",
                "--dataset_dir", str(twof_dir),
                "--single_partition", "--n_jobs", "1",
                "--response_column", "twof_generations",
            ],
            check=True,
        )

        # Compute restricted acc_clean + acc_2f + drag
        from datasets import load_from_disk
        from collections import defaultdict

        clean_ds = load_from_disk(str(processed_ds))
        twof_ds = load_from_disk(str(twof_ds_path))
        kept_ids = set(twof_ds["id"])

        per_problem = defaultdict(lambda: {"correct": 0, "total": 0})
        for e in clean_ds:
            if e["id"] not in kept_ids:
                continue
            per_problem[e["id"]]["total"] += 1
            if e["init_response_generations_correctness"]:
                per_problem[e["id"]]["correct"] += 1
        clean_correct = sum(p["correct"] for p in per_problem.values())
        clean_total = sum(p["total"] for p in per_problem.values())
        acc_clean = clean_correct / clean_total if clean_total else None

        analysis_files = sorted(twof_dir.glob("evaluated_*_error_analysis.json"))
        if not analysis_files:
            raise FileNotFoundError(f"No 2F error_analysis under {twof_dir}")
        with open(analysis_files[-1]) as f:
            analysis = json.load(f)
        acc_2f = float(
            analysis["pass_at_k_by_source"]["overall"]["overall_correctness"]
        )

        drag = acc_clean - acc_2f if acc_clean is not None else None
        _write_result(
            results_fpath,
            acc_clean=acc_clean,
            acc_2f=acc_2f,
            drag=drag,
            n_kept_problems=len(kept_ids),
            cfg=cfg,
            aggregate_failed=False,
        )


def _write_result(results_fpath: Path, *, acc_clean, acc_2f, drag,
                  n_kept_problems: int, cfg, aggregate_failed: bool) -> None:
    # magnet's symbol resolver chokes on JSON null values, so use -1.0 as
    # an out-of-range sentinel; the claim text detects aggregate_failed first.
    _sentinel = -1.0
    payload = {
        "result": {
            "acc_clean": _sentinel if acc_clean is None else acc_clean,
            "acc_2f": _sentinel if acc_2f is None else acc_2f,
            "drag": _sentinel if drag is None else drag,
            "n_kept_problems": n_kept_problems,
            "aggregate_failed": aggregate_failed,
            "model_config": str(cfg.model_config),
            "data_path": str(cfg.data_path),
            "init_template_key": str(cfg.init_template_key),
            "twof_template_key": str(cfg.twof_template_key),
            "max_questions": int(cfg.max_questions),
            "n": int(cfg.n),
            "num_true": int(cfg.num_true),
            "num_false": int(cfg.num_false),
        }
    }
    with open(results_fpath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[card-node] wrote {results_fpath}: {payload['result']}", flush=True)


def _len_dataset(path: Path) -> int:
    from datasets import load_from_disk
    return len(load_from_disk(str(path)))


def main(argv=None):
    return RunDragExperimentCLI.main(argv=argv)


if __name__ == "__main__":
    main()
