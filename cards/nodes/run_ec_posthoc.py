"""Magnet node: error-conditioning POST-HOC (pure analysis; NO inference).

Reads the verdict-FILTERED metric from the shared 1F conditioned-inference cache that
`drag-1f` produced (runs/cond_cache/<model>/<dataset>/1f/ec_summary.json). This card is
analysis-only: it must NOT be launched on the GPU sweep — run `drag-1f` for the
(model, dataset) first, then this reads its cached ec_summary.

delta_acc = correctness_filtered_init_sampling - correctness_filtered  (verdict-parseable cohort).
"""
from __future__ import annotations

import json
from pathlib import Path

import scriptconfig as scfg

from cards.nodes._cond_cache import read_cond_ec_summary


class RunEcPosthocCLI(scfg.DataConfig):
    """Pure post-hoc analysis node (reuses the drag-1f conditioned-inference cache)."""

    model_config = scfg.Value("Qwen3_8B_NoThinking", tags=["algo_param"])
    data_path = scfg.Value("data/smoke/gpqa/gpqa.ds", tags=["algo_param"])
    dataset = scfg.Value("", tags=["algo_param"], help="Benchmark name (cache key).")
    cond_cache_root = scfg.Value(
        "runs/cond_cache", tags=["algo_param"],
        help="Shared 1F conditioned-inference cache produced by the drag-1f card.")
    results_fpath = scfg.Value("results.json", tags=["out_path", "primary"])

    @classmethod
    def main(cls, argv=None, **kwargs):
        cfg = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)
        results_fpath = Path(cfg.results_fpath).resolve()
        results_fpath.parent.mkdir(parents=True, exist_ok=True)

        ec = read_cond_ec_summary(cfg.cond_cache_root, cfg.model_config,
                                  cfg.dataset, cfg.data_path, regime="1f")
        if ec is None:
            print("[ec-posthoc] no cached 1F conditioned inference for this (model, dataset); "
                  "run the drag-1f card first.", flush=True)
            _write(results_fpath, cfg, delta_acc=None, acc_direct=None, acc_conditioned=None,
                   n_kept_problems=0, filter_dropped_all=False, cache_missing=True)
            return
        acc_direct = ec.get("correctness_filtered_init_sampling")
        acc_cond = ec.get("correctness_filtered")
        n_filt = int(ec.get("num_problems_filtered") or 0)
        dropped = (n_filt == 0)
        delta = (float(acc_direct) - float(acc_cond)) if (not dropped and acc_direct is not None and acc_cond is not None) else None
        _write(results_fpath, cfg, delta_acc=delta, acc_direct=acc_direct, acc_conditioned=acc_cond,
               n_kept_problems=n_filt, filter_dropped_all=dropped, cache_missing=False)


def _write(results_fpath, cfg, *, delta_acc, acc_direct, acc_conditioned,
           n_kept_problems, filter_dropped_all, cache_missing):
    s = -1.0
    payload = {"result": {
        "delta_acc": s if delta_acc is None else delta_acc,
        "acc_direct": s if acc_direct is None else acc_direct,
        "acc_conditioned": s if acc_conditioned is None else acc_conditioned,
        "n_kept_problems": int(n_kept_problems),
        "regime": "1f", "setting": "posthoc",
        "filter_dropped_all": bool(filter_dropped_all),
        "cache_missing": bool(cache_missing),
        "aggregate_failed": bool(cache_missing),
        "model_config": str(cfg.model_config), "data_path": str(cfg.data_path),
    }}
    Path(results_fpath).write_text(json.dumps(payload, indent=2))
    print(f"[ec-posthoc] wrote {results_fpath}: {payload['result']}", flush=True)


def main(argv=None):
    return RunEcPosthocCLI.main(argv=argv)


if __name__ == "__main__":
    main()
