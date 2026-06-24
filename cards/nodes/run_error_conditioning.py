"""
Magnet pipeline node: §3 error conditioning.

Subprocess chain (one long-running Python process on the host's GPU;
steps 1-3 are init sampling, auto-reused from the shared init cache):

    inference run  (direct, clean prompt, task_name=init_response)
      -> eval math --flatten_dataset   (eval verb math/crux/game_of_24 per dataset family)
      -> data initial-sampling-postprocess
      -> data aggregate -T 0 -F {2 if regime==2f else 1}   (2f: 2 drafts; 1f/framing: 1)
      -> inference run  (conditioned, regime-specific template,
                         task_name=init_response)
      -> eval math   (eval verb per dataset family)
      -> contextual_drag analysis error_conditioning run
            --setting <regime> --cond_jsonl <cond evaluated_*.jsonl>
            --direct_jsonl <direct evaluated_*.jsonl> --out summary.json
      -> compute delta_acc = acc_direct - acc_conditioned

Three regimes (--regime):
  * 1f / 2f (posthoc): the conditioning prompt asks the model to *judge* one /
    two of its own failed drafts (<overall_verdict*> tags); the metric is
    restricted to the verdict-parseable cohort (the verdict filter).
  * framing (external): the prompt *explicitly states* the single draft is
    incorrect; --setting framing applies NO verdict filter.

Writes a results.json magnet's GenericPipelineProcessor consumes:

    {"result": {
        "delta_acc": 0.22,
        "acc_direct": 0.40, "acc_conditioned": 0.18,
        "n_kept_problems": 9, "regime": "2f",
        "aggregate_failed": false, "filter_dropped_all": false,
        ...
    }}

Degenerate branches that write null delta_acc (so magnet emits a
legible Inconclusive instead of crashing):
  - data aggregate exits non-zero or writes no .ds → aggregate_failed
  - verdict filter empties the 1f/2f cohort → filter_dropped_all (framing: no filter)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import scriptconfig as scfg

from cards.nodes._dataset_registry import aggregate_command_for, eval_verb_for
from cards.nodes._init_cache import ensure_init_sampling


class RunErrorConditioningCLI(scfg.DataConfig):
    """Sweep-able pipeline node for the §3 error-conditioning claim card."""

    model_config = scfg.Value("Qwen3_8B_NoThinking", tags=["algo_param"])
    data_path = scfg.Value("data/aime24/aime24.ds", tags=["algo_param"])
    regime = scfg.Value(
        "2f", choices=["1f", "2f", "framing"], tags=["algo_param"],
        help="1f/2f = posthoc (model judges 1/2 failed drafts; metric uses the "
             "verdict-parseable cohort). framing = external (prompt states the single "
             "draft is incorrect; no verdict filter). Formal cards use 1f and framing.")
    init_template_path = scfg.Value(
        "prompt_templates/init_response_prompt_templates.json", tags=["algo_param"])
    init_template_key = scfg.Value("qwen_math_prompt", tags=["algo_param"])
    cond_template_path_1f = scfg.Value(
        "prompt_templates/1f_templates.json", tags=["algo_param"])
    cond_template_key_1f = scfg.Value("1f", tags=["algo_param"])
    cond_template_path_2f = scfg.Value(
        "prompt_templates/2f_templates.json", tags=["algo_param"])
    cond_template_key_2f = scfg.Value("2f", tags=["algo_param"])
    framing_template_path = scfg.Value(
        "prompt_templates/ablation_templates.json", tags=["algo_param"])
    framing_template_key = scfg.Value("framing", tags=["algo_param"])
    max_questions = scfg.Value(16, type=int, tags=["algo_param"])
    n = scfg.Value(8, type=int, tags=["algo_param"])
    max_tokens = scfg.Value(2048, type=int, tags=["algo_param"])
    gpu_memory_utilization = scfg.Value(0.85, type=float, tags=["algo_param"])
    tensor_parallel_size = scfg.Value(
        1, type=int, tags=["algo_param"],
        help="vLLM tensor-parallel GPUs per model instance (shard one model across "
             "N GPUs). gpt-oss-20B MoE is validated at TP=4.")
    min_num_true_sampling = scfg.Value(
        2, type=int, tags=["algo_param"],
        help="`--min_num_true_sampling` for `data aggregate`: keep a problem only "
             "if it has >= this many correct responses (must be >= num_true=0). "
             "Lower it on hard tasks (e.g. aime24) where few problems reach the "
             "default 2 correct, which otherwise drives n_kept toward 0.")
    min_num_false_sampling = scfg.Value(
        2, type=int, tags=["algo_param"],
        help="`--min_num_false_sampling` for `data aggregate`: keep a problem only "
             "if it has >= this many incorrect responses (must be >= num_false).")

    dataset = scfg.Value("", tags=["algo_param"], help="Benchmark name; selects eval verb + aggregate command per family.")

    init_cache_root = scfg.Value(
        "runs/init_cache",
        help=(
            "Shared init-sampling cache root, keyed by <model>/<dataset>. Cards "
            "with matching init config (model/dataset/template/n/max_tokens/"
            "max_questions) auto-reuse it instead of regenerating. Empty string "
            "= card-local init."
        ),
        tags=["algo_param"],
    )

    results_fpath = scfg.Value("results.json", tags=["out_path", "primary"])

    @classmethod
    def main(cls, argv=None, **kwargs):
        cfg = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)

        results_fpath = Path(cfg.results_fpath).resolve()
        results_fpath.parent.mkdir(parents=True, exist_ok=True)
        work_dir = results_fpath.parent
        cond_dir = work_dir / "cond_inference"
        agg_dir = work_dir / "aggregate"
        for d in (cond_dir, agg_dir):
            d.mkdir(parents=True, exist_ok=True)

        regime = str(cfg.regime)
        num_false = 2 if regime == "2f" else 1
        if regime == "framing":
            cond_tpath, cond_tkey = cfg.framing_template_path, cfg.framing_template_key
        elif regime == "1f":
            cond_tpath, cond_tkey = cfg.cond_template_path_1f, cfg.cond_template_key_1f
        else:
            cond_tpath, cond_tkey = cfg.cond_template_path_2f, cfg.cond_template_key_2f

        cdrag = [sys.executable, "-m", "contextual_drag"]
        _ds = str(cfg.dataset).strip()
        _eval_verb = eval_verb_for(_ds) if _ds else "math"
        _agg_cmd = aggregate_command_for(_ds) if _ds else "aggregate"

        print(f"[card-node §3] steps 1-3/7: init sampling (shared cache)", flush=True)
        init = ensure_init_sampling(
            cdrag=cdrag, work_dir=work_dir, init_cache_root=cfg.init_cache_root,
            model_config=cfg.model_config, data_path=cfg.data_path,
            init_template_path=cfg.init_template_path,
            init_template_key=cfg.init_template_key,
            max_questions=cfg.max_questions, n=cfg.n, max_tokens=cfg.max_tokens,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            tensor_parallel_size=int(cfg.tensor_parallel_size),
            eval_verb=_eval_verb, dataset=_ds,
        )
        processed_ds = init.processed_ds

        print(f"[card-node §3] step 4/7: data aggregate -T 0 -F {num_false}", flush=True)
        agg = subprocess.run(cdrag + [
            "data", _agg_cmd,
            "--input_dir", str(processed_ds),
            "--num_true", "0",
            "--num_false", str(num_false),
            "--min_num_true_sampling", str(cfg.min_num_true_sampling),
            "--min_num_false_sampling", str(cfg.min_num_false_sampling),
            "--output_dir", str(agg_dir),
            "--init_response_models", str(cfg.model_config),
        ], check=False)
        cond_ds_path = agg_dir / f"minimal_aggregated_data_T0_F{num_false}.ds"
        if agg.returncode != 0 or not (cond_ds_path / "dataset_info.json").exists():
            print(f"[card-node §3] aggregate produced no usable dataset (exit "
                  f"{agg.returncode}); writing degenerate results.json.", flush=True)
            _write_result(
                results_fpath, cfg, regime,
                delta_acc=None, acc_direct=None, acc_conditioned=None,
                n_kept_problems=0, aggregate_failed=True, filter_dropped_all=False,
            )
            return

        print(f"[card-node §3] step 5/7: conditioned inference", flush=True)
        n_kept = _len_dataset(cond_ds_path)
        subprocess.run(cdrag + [
            "inference", "run",
            "--model_config", str(cfg.model_config),
            "--data_path", str(cond_ds_path),
            "--prompt_template_path", str(cond_tpath),
            "--prompt_template_key", str(cond_tkey),
            "--output_dir", str(cond_dir),
            "--task_name", "init_response",
            "--max_questions", str(n_kept),
            "--n", str(cfg.n),
            "--batch_size", str(min(8, n_kept)),
            "--tensor_parallel_size", str(cfg.tensor_parallel_size),
            "--gpu_memory_utilization", str(cfg.gpu_memory_utilization),
            "--max_tokens", str(cfg.max_tokens),
        ], check=True)

        print(f"[card-node §3] step 6/7: eval conditioned", flush=True)
        subprocess.run(cdrag + [
            "eval", _eval_verb,
            "--dataset_dir", str(cond_dir),
            "--single_partition", "--n_jobs", "1",
        ], check=True)

        cond_jsonl = _latest(cond_dir, "evaluated_*.jsonl", exclude_suffix="_flattened.jsonl")
        direct_jsonl = _latest(init.init_dir, "evaluated_*.jsonl", exclude_suffix="_flattened.jsonl")
        summary_path = work_dir / "ec_summary.json"

        print(f"[card-node §3] step 7/7: analysis error_conditioning run", flush=True)
        subprocess.run(cdrag + [
            "analysis", "error_conditioning", "run",
            "--setting", regime,
            "--cond_jsonl", str(cond_jsonl),
            "--direct_jsonl", str(direct_jsonl),
            "--out", str(summary_path),
        ], check=True)

        summary = json.loads(summary_path.read_text())
        # 1f/2f use *filtered* metric (verdict filter applied to cond cohort).
        acc_conditioned = summary.get("correctness_filtered")
        acc_direct = summary.get("correctness_filtered_init_sampling")
        n_filtered = int(summary.get("num_problems_filtered") or 0)
        filter_dropped_all = (n_filtered == 0)

        if filter_dropped_all or acc_direct is None or acc_conditioned is None:
            delta_acc = None
        else:
            delta_acc = float(acc_direct) - float(acc_conditioned)

        _write_result(
            results_fpath, cfg, regime,
            delta_acc=delta_acc,
            acc_direct=acc_direct, acc_conditioned=acc_conditioned,
            n_kept_problems=n_filtered,
            aggregate_failed=False,
            filter_dropped_all=filter_dropped_all,
        )


def _write_result(results_fpath: Path, cfg, regime: str, *, delta_acc, acc_direct,
                  acc_conditioned, n_kept_problems: int,
                  aggregate_failed: bool, filter_dropped_all: bool) -> None:
    # magnet's symbol resolver chokes on JSON null values; use -1.0 sentinel.
    # Claim text gates on aggregate_failed / filter_dropped_all first.
    _s = -1.0
    payload = {
        "result": {
            "delta_acc": _s if delta_acc is None else delta_acc,
            "acc_direct": _s if acc_direct is None else acc_direct,
            "acc_conditioned": _s if acc_conditioned is None else acc_conditioned,
            "n_kept_problems": n_kept_problems,
            "regime": regime,
            "aggregate_failed": aggregate_failed,
            "filter_dropped_all": filter_dropped_all,
            "model_config": str(cfg.model_config),
            "data_path": str(cfg.data_path),
        }
    }
    with open(results_fpath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[card-node §3] wrote {results_fpath}: {payload['result']}", flush=True)


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
    return RunErrorConditioningCLI.main(argv=argv)


if __name__ == "__main__":
    main()
