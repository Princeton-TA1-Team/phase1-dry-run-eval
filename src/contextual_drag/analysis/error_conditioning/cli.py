"""CLI surface for the §3 error-conditioning analysis.

Two verbs:

  ``run``        — compute the §3 conditioning metric from local jsonls
                   (alternative to the registry-driven ``run.py`` entrypoint).
  ``visualize``  — render heatmaps + LaTeX tables from a prior run.
"""
from __future__ import annotations

import scriptconfig as scfg


class ErrorConditioningRunCLI(scfg.DataConfig):
    """Compute the §3 conditioning metric from local jsonls.

    Used by the §3 magnet card. Takes explicit ``--cond_jsonl`` +
    ``--direct_jsonl`` paths and emits a small summary JSON the card
    wrapper reads to populate symbols. The existing ``run.py`` registry-
    driven entrypoint is unchanged; this is an alternate local-file
    entrypoint that wraps the same ``metrics.compute_cell`` logic on
    user-supplied paths.
    """
    setting       = scfg.Value("2f", choices=["framing", "1f", "2f"], tags=["algo_param"])
    cond_jsonl    = scfg.Value(None, required=True, tags=["algo_param"])
    direct_jsonl  = scfg.Value(None, required=True, tags=["algo_param"])
    out           = scfg.Value(None, required=True, tags=["out_path"])

    @classmethod
    def main(cls, argv=None, **kwargs):
        from contextual_drag.analysis.error_conditioning.metrics import (
            compute_cell_from_jsonls,
        )
        import json
        from pathlib import Path

        cfg = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)
        result = compute_cell_from_jsonls(
            setting=cfg.setting,
            cond_jsonl=cfg.cond_jsonl,
            direct_jsonl=cfg.direct_jsonl,
        )
        Path(cfg.out).write_text(json.dumps(result, indent=2))


class ErrorConditioningVisualizeCLI(scfg.DataConfig):
    """Render heatmaps + tables from a prior run."""
    setting = scfg.Value(None, tags=["algo_param"])

    @classmethod
    def main(cls, argv=None, **kwargs):
        from contextual_drag.analysis.error_conditioning import visualize as _viz
        cfg = cls.cli(argv=argv, data=kwargs, strict=True)
        _viz.main(setting=cfg.setting)


class ErrorConditioningCLI(scfg.ModalCLI):
    run = ErrorConditioningRunCLI
    visualize = ErrorConditioningVisualizeCLI
