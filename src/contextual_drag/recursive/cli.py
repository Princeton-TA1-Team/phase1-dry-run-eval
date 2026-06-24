"""CLI for the contextual_drag recursive self-improvement pipeline.

Drives the rf1 (strategy + filter1 + solve loop) and naive (1f-only solve
loop) variants. Single-cell invocation: builds one `TaskConfig`, stashes it
on `_explicit_cells`, and hands it to `run_pipeline.main` to bypass the
multi-cell resolver.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import scriptconfig as scfg


def _cfg_to_args(cfg) -> argparse.Namespace:
    """Translate scfg cfg → argparse.Namespace the run_pipeline expects.

    For a single-cell CLI invocation we build the TaskConfig directly and
    stash it on `_explicit_cells`, bypassing the multi-cell resolver.
    """
    from contextual_drag.recursive.pipeline.pipeline import TaskConfig

    modelid = cfg.modelid or cfg.model_config
    input_ds = Path(cfg.input_ds)
    output_dir = Path(cfg.output_dir)
    template_path = Path(cfg.template_path) if cfg.template_path else None
    if template_path is None:
        raise SystemExit(
            "--template_path is required (path to the prompt-templates JSON; "
            "rf1 expects recursive_filter1-style templates including "
            "`metacognitive_filter_strategy` / `metacognitive_filter_filter1` / "
            "`metacognitive_filter1_solve*`; naive expects 1f-style templates "
            "with `1f` / `1f_qa_mc` / `1f_crux_input`).")

    init_alias = cfg.init_alias or cfg.model_config
    task_name = cfg.task_name or "task"
    # run_id seeds round-0 trajectory selection AND per-round continuation picks,
    # so distinct run_ids give independent trajectories (the multi-run wrapper
    # launches run_id 1..N). seed seeds rollout generation (fixed across runs,
    # matching the old-repo deploy_sweep). Output dir is the cell dir itself.
    run_id = int(cfg.run_id)
    _seed = int(cfg.seed)

    cell = TaskConfig(
        task=task_name,
        init_input_ds=input_ds,
        output_root=output_dir,
        template_path=template_path,
        init_alias=init_alias,
        inference_alias=cfg.model_config,
        run_id=run_id,
        max_recursive_steps=cfg.max_recursive_steps,
        n_samples_solve=cfg.n_samples_solve,
        max_concurrent=128,
        seed=_seed,
        variant=cfg.variant,
        verify_workers=8,
    )

    return argparse.Namespace(
        _explicit_cells=[cell],
        variant=cfg.variant,
        model=cfg.model_config,
        modelid=modelid,
        init_alias=init_alias,
        run_id=run_id,
        tasks=task_name,
        max_recursive_steps=cfg.max_recursive_steps,
        n_samples_solve=cfg.n_samples_solve,
        config_path=None,
        template_path=str(template_path),
        output_root=str(output_dir.parent),
        init_data_root=str(input_ds.parent.parent),
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_concurrent=128,
        max_tokens=cfg.max_tokens,
        max_rows_per_cell=None,
        seed=_seed,
        verify_workers=8,
        makeup_max_attempts=4,
        makeup_batch_size=4,
        prefix_caching=True,
        gdn_prefill_backend=None,
    )


class RecursiveRunCLI(scfg.DataConfig):
    """Run the rf1 / naive recursive self-improvement loop on one
    (model, task, variant) cell.

    Internally drives the Stage 0 → Stage 1 strategy (rf1 only) → Loop
    {2a draft / 2b join (rf1) / 2c filter1 (rf1) / 2d solve / 2e summary}
    pipeline as one long-running process. Per-stage .ds and per-row JSONL
    resume; kill+restart is safe.
    """
    variant            = scfg.Value("rf1", tags=["algo_param"],
                                    choices=["rf1", "naive"])
    model_config       = scfg.Value(None, required=True, tags=["algo_param"])
    modelid            = scfg.Value(None, tags=["algo_param"])  # auto-derived if not given
    init_alias         = scfg.Value(None, tags=["algo_param"])  # round-0 aggregator filter
    input_ds           = scfg.Value(None, required=True, tags=["algo_param"])
    output_dir         = scfg.Value(None, required=True, tags=["out_path"])
    template_path      = scfg.Value(None, tags=["algo_param"])
    max_recursive_steps = scfg.Value(16, type=int, tags=["algo_param"])
    n_samples_solve    = scfg.Value(8, type=int, tags=["algo_param"])
    run_id             = scfg.Value(0, type=int, tags=["algo_param"],
                                    help="Seeds round-0 + per-round continuation picks; vary 1..N for independent trajectories.")
    seed               = scfg.Value(42, type=int, tags=["algo_param"],
                                    help="Rollout-generation seed (fixed across runs, per old-repo).")
    tensor_parallel_size = scfg.Value(1, type=int, tags=["algo_param"])
    gpu_memory_utilization = scfg.Value(0.9, type=float, tags=["algo_param"])
    max_tokens         = scfg.Value(65536, type=int, tags=["algo_param"])
    task_name          = scfg.Value(None, tags=["algo_param"])

    @classmethod
    def main(cls, argv=None, **kwargs):
        from contextual_drag.recursive.run_pipeline import main as _run
        cfg = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)
        args = _cfg_to_args(cfg)
        _run(args)


class RecursiveCLI(scfg.ModalCLI):
    run = RecursiveRunCLI
