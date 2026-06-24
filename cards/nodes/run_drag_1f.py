"""Magnet node: 1F contextual drag (single incorrect draft, raw/unfiltered).

Pipeline: init sampling (shared init cache) -> aggregate T0 F1 -> 1F conditioned
inference (`1f` template) -> eval -> analysis. The conditioned inference + analysis
are produced into the SHARED conditioned-inference cache
(runs/cond_cache/<model>/<dataset>/1f/) via `ensure_cond_eval`, so the
`error-conditioning-posthoc` analysis card can reuse them without re-running inference.

drag_1f = correctness_raw_init_sampling - correctness_raw   (RAW; over all problems).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import scriptconfig as scfg

from cards.nodes._dataset_registry import aggregate_command_for, eval_verb_for
from cards.nodes._init_cache import ensure_init_sampling
from cards.nodes._cond_cache import ensure_cond_eval, _latest


class RunDrag1fCLI(scfg.DataConfig):
    """Sweep-able node: 1F drag (direct vs single incorrect draft, raw)."""

    model_config = scfg.Value("Qwen3_8B_NoThinking", tags=["algo_param"])
    data_path = scfg.Value("data/smoke/gpqa/gpqa.ds", tags=["algo_param"])
    init_template_path = scfg.Value(
        "prompt_templates/init_response_prompt_templates.json", tags=["algo_param"])
    init_template_key = scfg.Value("qa_mc_prompt", tags=["algo_param"])
    cond_template_path_1f = scfg.Value(
        "prompt_templates/1f_templates.json", tags=["algo_param"])
    cond_template_key_1f = scfg.Value("1f", tags=["algo_param"])
    max_questions = scfg.Value(16, type=int, tags=["algo_param"])
    n = scfg.Value(8, type=int, tags=["algo_param"])
    max_tokens = scfg.Value(2048, type=int, tags=["algo_param"])
    gpu_memory_utilization = scfg.Value(0.85, type=float, tags=["algo_param"])
    tensor_parallel_size = scfg.Value(1, type=int, tags=["algo_param"])
    min_num_true_sampling = scfg.Value(2, type=int, tags=["algo_param"])
    min_num_false_sampling = scfg.Value(2, type=int, tags=["algo_param"])
    dataset = scfg.Value("", tags=["algo_param"], help="Benchmark name; selects family.")
    init_cache_root = scfg.Value("runs/init_cache", tags=["algo_param"])
    cond_cache_root = scfg.Value(
        "runs/cond_cache", tags=["algo_param"],
        help="Shared 1F conditioned-inference cache, keyed by <model>/<dataset>/1f; "
             "the error-conditioning-posthoc analysis card reuses it.")
    results_fpath = scfg.Value("results.json", tags=["out_path", "primary"])

    @classmethod
    def main(cls, argv=None, **kwargs):
        cfg = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)
        results_fpath = Path(cfg.results_fpath).resolve()
        results_fpath.parent.mkdir(parents=True, exist_ok=True)
        work_dir = results_fpath.parent
        cdrag = [sys.executable, "-m", "contextual_drag"]
        _ds = str(cfg.dataset).strip()
        eval_verb = eval_verb_for(_ds) if _ds else "math"
        agg_cmd = aggregate_command_for(_ds) if _ds else "aggregate"

        print("[drag-1f] init sampling (shared cache)", flush=True)
        init = ensure_init_sampling(
            cdrag=cdrag, work_dir=work_dir, init_cache_root=cfg.init_cache_root,
            model_config=cfg.model_config, data_path=cfg.data_path,
            init_template_path=cfg.init_template_path, init_template_key=cfg.init_template_key,
            max_questions=cfg.max_questions, n=cfg.n, max_tokens=cfg.max_tokens,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            tensor_parallel_size=int(cfg.tensor_parallel_size),
            eval_verb=eval_verb, dataset=_ds)

        direct_jsonl = _latest(init.init_dir, "evaluated_*.jsonl", exclude_suffix="_flattened.jsonl")
        print("[drag-1f] ensure 1F conditioned inference + analysis (shared cond cache)", flush=True)
        res = ensure_cond_eval(
            cdrag=cdrag, work_dir=work_dir, cond_cache_root=cfg.cond_cache_root,
            regime="1f", setting="1f", model_config=cfg.model_config, data_path=cfg.data_path,
            dataset=_ds, processed_init_ds=init.processed_ds, direct_eval_jsonl=direct_jsonl,
            agg_cmd=agg_cmd, cond_template_path=cfg.cond_template_path_1f,
            cond_template_key=cfg.cond_template_key_1f, num_false=1,
            min_num_true_sampling=int(cfg.min_num_true_sampling),
            min_num_false_sampling=int(cfg.min_num_false_sampling),
            n=int(cfg.n), max_tokens=int(cfg.max_tokens),
            gpu_memory_utilization=float(cfg.gpu_memory_utilization),
            tensor_parallel_size=int(cfg.tensor_parallel_size), eval_verb=eval_verb)

        ec = res.get("ec_summary")
        if res.get("aggregate_failed") or not ec:
            _write(results_fpath, cfg, drag_1f=None, acc_direct=None, acc_1f=None,
                   n_kept_problems=0, aggregate_failed=True)
            return
        acc_direct = ec.get("correctness_raw_init_sampling")
        acc_1f = ec.get("correctness_raw")
        n_cond = int(ec.get("num_problems_cond") or 0)
        drag_1f = (float(acc_direct) - float(acc_1f)) if (acc_direct is not None and acc_1f is not None) else None
        _write(results_fpath, cfg, drag_1f=drag_1f, acc_direct=acc_direct, acc_1f=acc_1f,
               n_kept_problems=n_cond, aggregate_failed=False)


def _write(results_fpath, cfg, *, drag_1f, acc_direct, acc_1f, n_kept_problems, aggregate_failed):
    s = -1.0
    payload = {"result": {
        "drag_1f": s if drag_1f is None else drag_1f,
        "acc_direct": s if acc_direct is None else acc_direct,
        "acc_1f": s if acc_1f is None else acc_1f,
        "n_kept_problems": int(n_kept_problems),
        "aggregate_failed": bool(aggregate_failed),
        "model_config": str(cfg.model_config), "data_path": str(cfg.data_path),
        "cond_cache_root": str(cfg.cond_cache_root),
    }}
    Path(results_fpath).write_text(json.dumps(payload, indent=2))
    print(f"[drag-1f] wrote {results_fpath}: {payload['result']}", flush=True)


def main(argv=None):
    return RunDrag1fCLI.main(argv=argv)


if __name__ == "__main__":
    main()
