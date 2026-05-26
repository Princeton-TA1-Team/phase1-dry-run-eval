"""CLI surface for the §4 mitigation outcome-bucket analysis.

Two verbs:

  ``run``     — compute the §4 outcome-bucket decomposition for one
                (model, task) cell from local jsonls.
  ``render``  — render the four-panel mitigation-buckets figure.
"""
from __future__ import annotations

import scriptconfig as scfg


class MitigationBucketsRunCLI(scfg.DataConfig):
    """Compute the §4 outcome-bucket decomposition for one (model, task) cell.

    Used by the §4 magnet card. Takes three local jsonl paths
    (direct, 1f, mitigation), binarizes each row at ``threshold``, and
    accumulates 8-bucket counts. Emits a small summary JSON the card
    wrapper reads to populate symbols.
    """
    direct_jsonl = scfg.Value(None, required=True, tags=["algo_param"])
    onef_jsonl   = scfg.Value(None, required=True, tags=["algo_param"])
    mit_jsonl    = scfg.Value(None, required=True, tags=["algo_param"])
    threshold    = scfg.Value(0.5, type=float, tags=["algo_param"])
    variant      = scfg.Value("cm_filter1", tags=["algo_param"])
    out          = scfg.Value(None, required=True, tags=["out_path"])

    @classmethod
    def main(cls, argv=None, **kwargs):
        from contextual_drag.analysis.mitigation_buckets.buckets import (
            count_observations_from_jsonls,
            derive_metrics,
        )
        import json
        from pathlib import Path

        cfg = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)
        counts, n_obs = count_observations_from_jsonls(
            direct_jsonl=cfg.direct_jsonl,
            onef_jsonl=cfg.onef_jsonl,
            mit_jsonl=cfg.mit_jsonl,
            threshold=cfg.threshold,
        )
        derived = derive_metrics(counts)
        Path(cfg.out).write_text(json.dumps({
            "counts":         counts,
            "derived":        derived,
            "n_observations": n_obs,
            "threshold":      cfg.threshold,
            "variant":        cfg.variant,
        }, indent=2))


class MitigationBucketsRenderCLI(scfg.DataConfig):
    """Render the four-panel mitigation-buckets figure."""

    @classmethod
    def main(cls, argv=None, **kwargs):
        from contextual_drag.analysis.mitigation_buckets import visualize as _viz
        _viz.main()


class MitigationBucketsCLI(scfg.ModalCLI):
    run    = MitigationBucketsRunCLI
    render = MitigationBucketsRenderCLI
