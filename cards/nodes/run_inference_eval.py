"""
Magnet pipeline node: run `contextual_drag inference run` followed by
`contextual_drag eval math` on a single benchmark + model + prompt template,
then write a results.json that the magnet evaluation card can consume to
populate symbol values.

Output JSON shape (consumed by magnet's GenericPipelineProcessor):

    {"result": {
        "accuracy": 1.0,
        "pass_at_1": 1.0,
        "maj_at_1": 1.0,
        "total_questions": 4,
        "n_trajectories": 1
    }}

The wrapper subprocesses our existing `python -m contextual_drag` CLI rather
than importing it, so heavy deps (vllm) only load when this node actually
runs and never when the magnet card is parsed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import scriptconfig as scfg


class RunInferenceEvalCLI(scfg.DataConfig):
    """Sweep-able pipeline node for the contextual_drag wiring smoke card."""

    model_config = scfg.Value(
        "Qwen3_8B_NoThinking",
        help="Alias from contextual_drag/resources/inference/eval_models_params.json.",
        tags=["algo_param"],
    )
    data_path = scfg.Value(
        "data/math500/math500.ds",
        help="HF dataset directory (DatasetDict; 'test' split is selected automatically).",
        tags=["algo_param"],
    )
    prompt_template_path = scfg.Value(
        "prompt_templates/init_response_prompt_templates.json",
        help="JSON file containing prompt templates.",
        tags=["algo_param"],
    )
    prompt_template_key = scfg.Value(
        "qwen_math_prompt",
        help="Key inside prompt_template_path to use.",
        tags=["algo_param"],
    )
    max_questions = scfg.Value(
        4,
        type=int,
        help="Cap the number of dataset rows for fast smoke runs.",
        tags=["algo_param"],
    )
    n = scfg.Value(
        1,
        type=int,
        help="Samples per question.",
        tags=["algo_param"],
    )
    max_tokens = scfg.Value(
        2048,
        type=int,
        help="Hard cap on generation length per response.",
        tags=["algo_param"],
    )
    gpu_memory_utilization = scfg.Value(
        0.85,
        type=float,
        help="vLLM GPU memory fraction; lower for smaller cards.",
        tags=["algo_param"],
    )

    results_fpath = scfg.Value(
        "results.json",
        help=(
            "Path the magnet pipeline reads to populate symbols. Magnet "
            "rewrites this to a per-sweep hashed directory; we honor that."
        ),
        tags=["out_path", "primary"],
    )

    @classmethod
    def main(cls, argv=None, **kwargs):
        cfg = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)

        results_fpath = Path(cfg.results_fpath).resolve()
        results_fpath.parent.mkdir(parents=True, exist_ok=True)
        inference_output_dir = results_fpath.parent / "inference"
        inference_output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Inference
        inf_cmd = [
            sys.executable, "-m", "contextual_drag", "inference", "run",
            "--model_config", str(cfg.model_config),
            "--data_path", str(cfg.data_path),
            "--prompt_template_path", str(cfg.prompt_template_path),
            "--prompt_template_key", str(cfg.prompt_template_key),
            "--max_questions", str(cfg.max_questions),
            "--n", str(cfg.n),
            "--max_tokens", str(cfg.max_tokens),
            "--output_dir", str(inference_output_dir),
            "--batch_size", "4",
            "--tensor_parallel_size", "1",
            "--gpu_memory_utilization", str(cfg.gpu_memory_utilization),
        ]
        print(f"[card-node] running inference: {' '.join(inf_cmd)}", flush=True)
        subprocess.run(inf_cmd, check=True)

        # 2. Eval (math)
        eval_cmd = [
            sys.executable, "-m", "contextual_drag", "eval", "math",
            "--dataset_dir", str(inference_output_dir),
            "--single_partition",
            "--n_jobs", "1",
        ]
        print(f"[card-node] running eval: {' '.join(eval_cmd)}", flush=True)
        subprocess.run(eval_cmd, check=True)

        # 3. Read the evaluator's error_analysis JSON to extract accuracy.
        analysis_files = sorted(inference_output_dir.glob("evaluated_*_error_analysis.json"))
        if not analysis_files:
            raise FileNotFoundError(
                f"No evaluated_*_error_analysis.json under {inference_output_dir}; eval may have failed."
            )
        with open(analysis_files[-1]) as f:
            analysis = json.load(f)

        overall = analysis.get("pass_at_k_by_source", {}).get("overall", {})
        result = {
            "accuracy": float(overall.get("overall_correctness", 0.0)),
            "pass_at_k": list(overall.get("pass_at_k", []) or []),
            "maj_at_k": list(overall.get("maj_at_k", []) or []),
            "total_questions": int(overall.get("total_questions", 0)),
            "n_trajectories": int(overall.get("n_trajectories", 0)),
            "model_config": str(cfg.model_config),
            "data_path": str(cfg.data_path),
            "prompt_template_key": str(cfg.prompt_template_key),
        }

        with open(results_fpath, "w") as f:
            json.dump({"result": result}, f, indent=2)
        print(f"[card-node] wrote {results_fpath}: {result}", flush=True)


def main(argv=None):
    return RunInferenceEvalCLI.main(argv=argv)


if __name__ == "__main__":
    main()
