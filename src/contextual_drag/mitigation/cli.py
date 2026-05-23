"""CLI for the contextual_drag mitigation (context-manipulation) pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import scriptconfig as scfg


def _cfg_to_args(cfg) -> argparse.Namespace:
    """Translate scfg cfg → argparse.Namespace the run_pipeline expects.

    For a single-cell CLI invocation we build the TaskConfig directly and
    stash it on `_explicit_cells`, bypassing the multi-cell resolver.
    """
    from contextual_drag.mitigation.pipeline.pipeline import TaskConfig

    modelid = cfg.modelid or cfg.model_config
    input_ds = Path(cfg.input_ds)
    output_dir = Path(cfg.output_dir)
    template_path = Path(cfg.template_path) if cfg.template_path else None
    if template_path is None:
        raise SystemExit("--template_path is required (path to "
                         "context_manipulation_prompt_templates.json)")

    intermediate_dir = output_dir / "intermediate"
    results_dir = output_dir

    cell = TaskConfig(
        task=cfg.task_name or "task",
        modelid=modelid,
        variant=cfg.variant,
        init_input_ds=input_ds,
        intermediate_dir=intermediate_dir,
        results_dir=results_dir,
        template_path=template_path,
        inference_alias=cfg.model_config,
        n_samples_solve=cfg.n,
        max_concurrent=128,
        seed=42,
        verify_workers=8,
    )

    return argparse.Namespace(
        _explicit_cells=[cell],
        variants=cfg.variant,
        model=cfg.model_config,
        modelid=modelid,
        tasks=cfg.task_name or "task",
        n_samples_solve=cfg.n,
        config_path=None,
        template_path=str(template_path),
        init_data_root=str(input_ds.parent.parent),
        intermediate_root=str(intermediate_dir.parent),
        results_root=str(results_dir.parent),
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_concurrent=128,
        max_tokens=cfg.max_tokens,
        max_rows_per_cell=None,
        seed=42,
        verify_workers=8,
        prefix_caching=True,
        gdn_prefill_backend=None,
    )


class MitigationRunCLI(scfg.DataConfig):
    """Run the cm_filter1 / cm_revise1 chain on one (model, task, variant) cell.

    Internally drives the strategy → join → (filter1 | revise1) → solve →
    eval pipeline as one long-running process. Per-stage .ds and per-row
    JSONL resume; kill+restart is safe.
    """
    variant            = scfg.Value("cm_filter1", tags=["algo_param"],
                                    choices=["cm_filter1", "cm_revise1"])
    model_config       = scfg.Value(None, required=True, tags=["algo_param"])
    modelid            = scfg.Value(None, tags=["algo_param"])  # auto-derived if not given
    input_ds           = scfg.Value(None, required=True, tags=["algo_param"])
    output_dir         = scfg.Value(None, required=True, tags=["out_path"])
    template_path      = scfg.Value(None, tags=["algo_param"])
    n                  = scfg.Value(8, type=int, tags=["algo_param"])
    tensor_parallel_size = scfg.Value(1, type=int, tags=["algo_param"])
    gpu_memory_utilization = scfg.Value(0.9, type=float, tags=["algo_param"])
    max_tokens         = scfg.Value(2048, type=int, tags=["algo_param"])
    task_name          = scfg.Value(None, tags=["algo_param"])

    @classmethod
    def main(cls, argv=None, **kwargs):
        from contextual_drag.mitigation.run_pipeline import main as _run
        cfg = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)
        args = _cfg_to_args(cfg)
        _run(args)


class MitigationCLI(scfg.ModalCLI):
    run = MitigationRunCLI
