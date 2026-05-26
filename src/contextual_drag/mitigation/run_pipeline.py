"""Entrypoint for the context-manipulation (cm_filter1 / cm_revise1) pipeline.

One Python process per run. Builds AsyncLLMEngine once, then iterates over
the requested (variant, task) cells calling pipeline.run_task. Resume is
automatic at two levels (per-row prompt-hash, per-stage artifact check —
same architecture as recursive_filter1).

Faithful to upstream `auto_context_manipulation_lib.sh`: same templates, same
n_samples_solve, same step-prefix JSONL keys (`<step>_response`), same
F2→F1 init data.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path

# HuggingFace tokenizers warns when a process loads a tokenizer and then
# fork()s child processes — the rust thread pool can't safely cross the fork.
# The Verifier in the solve stage forks via ProcessPoolExecutor. Disable rust-
# side parallelism BEFORE transformers imports the tokenizers package.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer  # noqa: E402

from contextual_drag.inference.run_model import (  # noqa: E402
    build_engine, install_signal_handlers, log_engine_info,
)
from contextual_drag.inference.config import load_model_config  # noqa: E402

from contextual_drag.mitigation.pipeline.pipeline import (  # noqa: E402
    TaskConfig, VARIANTS, SOLVE_N_SAMPLES, run_task,
)


DEFAULT_TASKS = ["aime24", "aime25", "hmmt24", "hmmt25",
                 "gpqa", "mmlu", "crux-i", "24-game"]
INIT_DS_BASENAME = "minimal_aggregated_data_T0_F1_flattend_from_F2.ds"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default=",".join(VARIANTS),
                   help=f"Comma-separated variants. Default: {','.join(VARIANTS)}.")
    p.add_argument("--model", required=True,
                   help="Canonical model alias (e.g. Gemma4_E4B).")
    p.add_argument("--modelid", required=True,
                   help="Short id used in cell dir names (e.g. g4e). Must match "
                        "the per-cell layout under <init_data_root>/<task>_<modelid>/.")
    p.add_argument("--tasks", default=",".join(DEFAULT_TASKS),
                   help=f"Comma-separated tasks. Default: all 8.")
    p.add_argument("--n-samples-solve", type=int, default=SOLVE_N_SAMPLES,
                   help=f"Solve cell n_samples. Default: {SOLVE_N_SAMPLES} "
                        f"(matches upstream SOLVE_N_SAMPLES).")

    p.add_argument("--config-path", required=True,
                   help="Path to eval_models_params.json.")
    p.add_argument("--template-path", required=True,
                   help="Path to context_manipulation_prompt_templates.json.")
    p.add_argument("--init-data-root", required=True,
                   help="Per-cell input root; expects "
                        "<init_data_root>/<task>_<modelid>/" + INIT_DS_BASENAME)
    p.add_argument("--intermediate-root", required=True,
                   help="Per-cell intermediate stage outputs go to "
                        "<intermediate_root>/<variant>/<task>_<modelid>/.")
    p.add_argument("--results-root", required=True,
                   help="Solve + eval artifacts go to "
                        "<results_root>/<variant>/<task>_<modelid>/.")

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
    p.add_argument("--verify-workers", type=int, default=8)

    p.add_argument("--prefix-caching",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gdn-prefill-backend", choices=["flashinfer", "triton"],
                   default=None)
    return p.parse_args()


def _resolve_cells(args) -> list[TaskConfig]:
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        raise SystemExit(f"unknown variants: {bad} (expected: {list(VARIANTS)})")
    tasks = [t for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        raise SystemExit("--tasks resolved to an empty list")

    init_root = Path(args.init_data_root)
    intermediate_root = Path(args.intermediate_root)
    results_root = Path(args.results_root)
    template_path = Path(args.template_path)
    if not template_path.is_file():
        raise SystemExit(f"prompt template not found: {template_path}")

    cells: list[TaskConfig] = []
    for variant in variants:
        for task in tasks:
            init_input_ds = init_root / f"{task}_{args.modelid}" / INIT_DS_BASENAME
            intermediate_dir = intermediate_root / variant / f"{task}_{args.modelid}"
            results_dir = results_root / variant / f"{task}_{args.modelid}"
            cells.append(TaskConfig(
                task=task,
                modelid=args.modelid,
                variant=variant,
                init_input_ds=init_input_ds,
                intermediate_dir=intermediate_dir,
                results_dir=results_dir,
                template_path=template_path,
                inference_alias=args.model,
                n_samples_solve=args.n_samples_solve,
                max_concurrent=args.max_concurrent,
                seed=args.seed,
                verify_workers=args.verify_workers,
            ))
    return cells


async def amain(args=None):
    if args is None:
        args = parse_args()
    cells = getattr(args, "_explicit_cells", None) or _resolve_cells(args)

    # Pre-flight: warn (not fail) about cells whose F1 input is missing — so
    # we still build the engine and run the cells that DO have inputs. A
    # missing input usually means the corresponding (task, modelid) cohort
    # filter excluded all problems (rare; e.g. 0 incorrect samples for that
    # model on that task).
    missing = [c for c in cells if not c.init_input_ds.is_dir()]
    if missing:
        print(f"\n[warn] {len(missing)} cell(s) have no F1 init dataset (will be "
              f"skipped at run time):", flush=True)
        for c in missing:
            print(f"   {c.variant}/{c.task}_{c.modelid}: {c.init_input_ds}",
                  flush=True)

    model_config = load_model_config(args.config_path, args.model)

    print(f"\n=== context_manipulation run ===", flush=True)
    print(f"  model={args.model} ({model_config['model_name']})  "
          f"modelid={args.modelid}", flush=True)
    print(f"  variants={[c.variant for c in cells if c.task == cells[0].task]}",
          flush=True)
    print(f"  tasks={list(dict.fromkeys(c.task for c in cells))}", flush=True)
    print(f"  n_samples_solve={args.n_samples_solve}  "
          f"tp={args.tensor_parallel_size}  "
          f"max_concurrent={args.max_concurrent}  "
          f"verify_workers={args.verify_workers}", flush=True)
    print(f"  total cells: {len(cells)} ({len(cells) - len(missing)} runnable)",
          flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"])

    engine = build_engine(args, model_config)
    log_engine_info(engine, model_config)
    install_signal_handlers(asyncio.get_running_loop())

    t_start = time.time()
    n_done = 0
    for cfg in cells:
        if not cfg.init_input_ds.is_dir():
            print(f"\n[skip] {cfg.variant}/{cfg.task}_{cfg.modelid}: "
                  f"missing F1 input {cfg.init_input_ds}", flush=True)
            continue
        try:
            await run_task(engine, tokenizer, cfg, model_config,
                           max_rows_per_cell=args.max_rows_per_cell,
                           max_tokens=args.max_tokens)
            n_done += 1
        except Exception as e:
            print(f"[cell error] {cfg.variant}/{cfg.task}_{cfg.modelid}: {e!r}",
                  flush=True)
            raise
    elapsed = time.time() - t_start
    print(f"\n=== {n_done}/{len(cells)} cell(s) processed in {elapsed:.1f}s "
          f"({elapsed/60:.1f} min) ===", flush=True)


def main(args=None):
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\nInterrupted")


if __name__ == "__main__":
    main()
