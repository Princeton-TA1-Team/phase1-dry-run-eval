#!/usr/bin/env python3
"""CLI entrypoint for the recursive self-improvement (rf1 / naive) pipeline.

One Python process per run. Loads AsyncLLMEngine once, then iterates over
the requested tasks calling pipeline.run_task. Resume is automatic at two
levels (per-row prompt-hash, per-stage artifact check) — kill+restart of the
host process is now the resume trigger (no Slurm afterany chain).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path

# HuggingFace tokenizers warns when a process loads a tokenizer and then
# fork()s child processes (e.g. via ProcessPoolExecutor) — its rust thread
# pool can't safely cross the fork. Stage 2d's `Verifier` does exactly that.
# Disable the rust-side parallelism here, before transformers imports the
# tokenizers package and primes its thread pool. This must happen at module
# top, ahead of `from transformers import ...`.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer  # noqa: E402

from contextual_drag.inference.run_model import (  # noqa: E402
    build_engine, install_signal_handlers, log_engine_info,
)
from contextual_drag.inference.config import load_model_config  # noqa: E402

from contextual_drag.recursive.pipeline.pipeline import (  # noqa: E402
    TaskConfig, VARIANTS, run_task,
)


DEFAULT_TASKS = ["aime24", "aime25", "hmmt24", "hmmt25"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="rf1", choices=list(VARIANTS),
                   help="Pipeline variant. `rf1` runs strategy + filter1 "
                        "context cleaning before each re-solve. `naive` skips "
                        "those stages and feeds the prior incorrect "
                        "trajectory directly to a 1f-style solve prompt.")
    p.add_argument("--model", required=True,
                   help="Inference model alias from eval_models_params.json "
                        "(e.g. GPT_OSS_20B_recursive)")
    p.add_argument("--init-alias", default="GPT_OSS_20B",
                   help="Alias under which the round-0 init responses were "
                        "originally generated. Used as the filter for the "
                        "round-0 aggregator. Default: GPT_OSS_20B.")
    p.add_argument("--modelid", required=True,
                   help="Short id used in run/job naming (e.g. g20r). Not used "
                        "for path layout — the run output tree is keyed by "
                        "run_id, not modelid.")
    p.add_argument("--run-id", type=int, required=True,
                   help="Identifier for this experiment run. Used for the "
                        "output dir (<output_root>/<run_id>/) and as the seed "
                        "for the round-0 trajectory selection.")
    p.add_argument("--tasks", default=",".join(DEFAULT_TASKS),
                   help=f"Comma-separated tasks. Default: {','.join(DEFAULT_TASKS)}")
    p.add_argument("--max-recursive-steps", type=int, default=4,
                   help="Number of filter1+solve rounds to run per task.")
    p.add_argument("--n-samples-solve", type=int, default=16,
                   help="Number of samples per problem in the solve cell.")

    p.add_argument("--config-path", required=True,
                   help="Path to eval_models_params.json.")
    p.add_argument("--template-path", required=True,
                   help="Path to the prompt templates JSON for the chosen variant.")
    p.add_argument("--output-root", required=True,
                   help="Run output directory will be <output_root>/<run_id>/<task>/.")
    p.add_argument("--init-data-root", required=True,
                   help="Round-0 input dir; expects "
                        "<init_data_root>/<task>/processed_flattened_init_responses.ds.")

    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    p.add_argument("--max-concurrent", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Override SamplingParams.max_tokens from the model config.")
    p.add_argument("--max-rows-per-cell", type=int, default=None,
                   help="Cap on per-cell rows after the resume scan. Useful for "
                        "smoke tests; e.g. --max-rows-per-cell 4 limits each "
                        "inference cell to 4 problems.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verify-workers", type=int, default=8,
                   help="ProcessPoolExecutor workers for streaming verification.")
    p.add_argument("--makeup-max-attempts", type=int, default=4,
                   help="Per round-r solve cell, after the main n=N pass, run "
                        "up to this many additional batches for any problem "
                        "that still has 0 (parsable+stop) samples. 0 disables.")
    p.add_argument("--makeup-batch-size", type=int, default=4,
                   help="Number of extra samples per still-unsatisfied id, "
                        "per makeup attempt.")
    p.add_argument("--prefix-caching",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gdn-prefill-backend", choices=["flashinfer", "triton"],
                   default=None)
    return p.parse_args()


async def amain(args=None):
    if args is None:
        args = parse_args()

    # Single-cell invocation (mitigation/recursive cli.py path): caller hands
    # us a pre-built TaskConfig list on `_explicit_cells` so we skip --tasks
    # resolution and trust their shape.
    explicit_cells = getattr(args, "_explicit_cells", None)

    if explicit_cells:
        task_cfgs = list(explicit_cells)
    else:
        tasks = [t for t in args.tasks.split(",") if t.strip()]
        if not tasks:
            raise SystemExit("--tasks resolved to an empty list")

        # Verify all per-task input datasets exist before paying for engine init.
        init_root = Path(args.init_data_root)
        template_path = Path(args.template_path)
        if not template_path.is_file():
            raise SystemExit(f"prompt template not found: {template_path}")

        task_cfgs: list[TaskConfig] = []
        output_run_root = Path(args.output_root) / str(args.run_id)
        for task in tasks:
            init_ds = init_root / task / "processed_flattened_init_responses.ds"
            if not init_ds.is_dir():
                raise SystemExit(
                    f"missing round-0 init dataset for task {task!r}: {init_ds}")
            task_cfgs.append(TaskConfig(
                task=task,
                init_input_ds=init_ds,
                output_root=output_run_root / task,
                template_path=template_path,
                init_alias=args.init_alias,
                inference_alias=args.model,
                run_id=args.run_id,
                max_recursive_steps=args.max_recursive_steps,
                n_samples_solve=args.n_samples_solve,
                max_concurrent=args.max_concurrent,
                seed=args.seed,
                variant=args.variant,
                verify_workers=args.verify_workers,
                makeup_max_attempts=args.makeup_max_attempts,
                makeup_batch_size=args.makeup_batch_size,
            ))

    model_config = load_model_config(args.config_path, args.model)

    print(f"=== recursive ({args.variant}) run ===", flush=True)
    print(f"  model={args.model} ({model_config['model_name']})  "
          f"init_alias={args.init_alias}  run_id={args.run_id}", flush=True)
    print(f"  tasks={[c.task for c in task_cfgs]}  "
          f"steps={args.max_recursive_steps}  "
          f"n_samples_solve={args.n_samples_solve}", flush=True)
    print(f"  tp={args.tensor_parallel_size}  "
          f"max_concurrent={args.max_concurrent}  "
          f"verify_workers={args.verify_workers}", flush=True)
    print(f"  makeup: max_attempts={args.makeup_max_attempts}  "
          f"batch_size={args.makeup_batch_size}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"])

    engine = build_engine(args, model_config)
    log_engine_info(engine, model_config)
    install_signal_handlers(asyncio.get_running_loop())

    t_start = time.time()
    for cfg in task_cfgs:
        try:
            await run_task(engine, tokenizer, cfg, model_config,
                           max_rows_per_cell=args.max_rows_per_cell,
                           max_tokens=args.max_tokens)
        except Exception as e:
            print(f"[task error] {cfg.task}: {e!r}", flush=True)
            raise
    elapsed = time.time() - t_start
    print(f"\n=== done in {elapsed:.1f}s ({elapsed/60:.1f} min) ===", flush=True)


def main(args=None):
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\nInterrupted")


if __name__ == "__main__":
    main()
