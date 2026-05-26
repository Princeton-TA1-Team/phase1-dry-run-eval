"""CLI surface for the TED (tree-edit-distance) analysis.

Three verbs:

  ``build_cache``  — build a per-model TED cache from local anchored + init
                     jsonls.
  ``summarize``    — read a cache and emit per-problem means JSON.
  ``render``       — render the anchored Direct-vs-Drag figure from a
                     directory of cache files.
"""
from __future__ import annotations

import scriptconfig as scfg


class TedBuildCacheCLI(scfg.DataConfig):
    """Build a per-model TED cache from local anchored + init jsonls.

    Wraps the same
    ``analysis.ted.edit_distance_analysis.{build_processed_anchored_data,
    load_init_responses, batched_compute_edit_distance_parallel}`` chain
    used internally, but takes explicit jsonl paths (no source resolution).
    """
    phase           = scfg.Value("2f", choices=["1f", "2f"], tags=["algo_param"])
    anchored_jsonl  = scfg.Value(None, required=True, tags=["algo_param"])
    init_jsonl      = scfg.Value(None, required=True, tags=["algo_param"])
    metric          = scfg.Value("tree", choices=["tree", "levenshtein", "binary"],
                                 tags=["algo_param"])
    n_jobs          = scfg.Value(4, type=int, tags=["algo_param"])
    cache_out       = scfg.Value(None, required=True, tags=["out_path"])

    @classmethod
    def main(cls, argv=None, **kwargs):
        from contextual_drag.analysis.ted.edit_distance_analysis import (
            build_cache_from_jsonls,
        )

        cfg = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)
        build_cache_from_jsonls(
            phase=cfg.phase,
            anchored_jsonl=cfg.anchored_jsonl,
            init_jsonl=cfg.init_jsonl,
            metric=cfg.metric,
            n_jobs=cfg.n_jobs,
            cache_out=cfg.cache_out,
        )


class TedSummarizeCLI(scfg.DataConfig):
    """Read a TED cache and emit per-problem means JSON.

    Output keys: ``anchored_responses`` (mean across per-problem anchored
    means), ``init_response`` (mean across per-problem init means),
    ``n_problems_with_both`` (+ extras: per-side counts, metric, reduction).
    """
    cache_in   = scfg.Value(None, required=True, tags=["algo_param"])
    metric     = scfg.Value("tree", tags=["algo_param"])
    reduction  = scfg.Value("min", choices=["min", "mean", "max", "median"],
                            tags=["algo_param"])
    out        = scfg.Value(None, required=True, tags=["out_path"])

    @classmethod
    def main(cls, argv=None, **kwargs):
        from contextual_drag.analysis.ted.edit_distance_analysis import summarize_cache
        import json
        from pathlib import Path

        cfg = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)
        Path(cfg.out).write_text(json.dumps(
            summarize_cache(cfg.cache_in, metric=cfg.metric, reduction=cfg.reduction),
            indent=2,
        ))


class TedRenderCLI(scfg.DataConfig):
    """Render the anchored Direct-vs-Drag figure from a directory of caches.

    Walks ``<cache_root>/<phase>/<MODEL>.json``. Restrict with ``--models``
    (comma list); default uses every ``*.json`` in the phase directory.
    """
    phase       = scfg.Value("2f", choices=["1f", "2f"], tags=["algo_param"])
    cache_root  = scfg.Value(None, required=True, tags=["algo_param"])
    models      = scfg.Value(None, tags=["algo_param"])
    metric      = scfg.Value("tree", choices=["tree", "levenshtein", "binary"],
                             tags=["algo_param"])
    out         = scfg.Value(None, required=True, tags=["out_path"])

    @classmethod
    def main(cls, argv=None, **kwargs):
        from contextual_drag.analysis.ted import visualize_anchored_main as _viz
        cfg = cls.cli(argv=argv, data=kwargs, strict=True)
        rebuilt = ["--phase", str(cfg.phase),
                   "--cache-root", str(cfg.cache_root),
                   "--metric", str(cfg.metric),
                   "--out", str(cfg.out)]
        if cfg.models:
            rebuilt += ["--models", str(cfg.models)]
        _viz.main(rebuilt)


class TedCLI(scfg.ModalCLI):
    build_cache = TedBuildCacheCLI
    summarize   = TedSummarizeCLI
    render      = TedRenderCLI
