#!/usr/bin/env python3
"""
Async vLLM driver for one inference cell.

Long-running Python process. Loads AsyncLLMEngine once, processes the cell
defined by (data_path, prompt_template_path, prompt_template_key, task_name,
n_samples) with per-row JSONL checkpointing keyed by sha256(rendered_prompt).
Rerunning the same command picks up exactly where the previous run left off.

This file is intentionally thin: it owns the async driver + per-row dispatch
logic. Everything else lives in sibling modules:
    config.py       — load_model_config, make_sampling_params
    prompts.py      — render_prompt, prompt_hash, load_completed_hashes
    stats.py        — CellStats, status_reporter
    diagnostics.py  — log_engine_info, prescan_cells, preview_prompts
"""

import asyncio
import dataclasses
import json
import os
import signal
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from contextual_drag.inference.config import load_model_config, make_sampling_params
from contextual_drag.inference.prompts import prompt_hash
from contextual_drag.inference.stats import CellStats, format_secs, status_reporter
from contextual_drag.inference.diagnostics import (
    log_engine_info,
    prescan_cells,
    preview_prompts,
)

SHUTDOWN = asyncio.Event()


# ---------- per-request handler ----------
class _FauxRequestOutput:
    """Minimal stand-in for vllm.RequestOutput, used by build_record + stats
    when we synthesize a per-row result from n separate n=1 sub-requests."""
    __slots__ = ("outputs", "prompt_token_ids", "finished")

    def __init__(self, outputs, prompt_token_ids):
        self.outputs = outputs
        self.prompt_token_ids = prompt_token_ids
        self.finished = True


def build_record(row_dict: dict, prompt: str, output, item: dict,
                 model_alias: str, n_samples_requested: int) -> dict:
    """Build the JSONL record for one row. Shape matches the parent's
    create_generation_result so downstream eval scripts work unchanged."""
    task_name = item["task_name"]
    responses = [
        {"response_id": j,
         "generated_response": o.text,
         "finish_reason": o.finish_reason}
        for j, o in enumerate(output.outputs)
    ]
    return {
        **row_dict,
        f"{task_name}_prompt": prompt,
        f"{task_name}_generations": responses,
        f"{task_name}_generations_metadata": {
            "num_responses": len(responses),
            "n_samples_requested": n_samples_requested,
            "n_samples_variant_default": item["n_samples"],
            "template_path": str(item["template_path"]),
            "template_key": item["template_key"],
            "model_config_alias": model_alias,
            "prompt_hash": prompt_hash(prompt),
        },
    }


async def _stream_one_request(engine, prompt, sampling_params, request_id):
    """Run one engine.generate call to completion. Returns the final
    RequestOutput, or None on shutdown/abort."""
    final = None
    try:
        async for output in engine.generate(prompt, sampling_params,
                                            request_id=request_id):
            final = output
            if SHUTDOWN.is_set():
                await engine.abort(request_id)
                return None
    except asyncio.CancelledError:
        try:
            await engine.abort(request_id)
        except Exception:
            pass
        raise
    if final is None or not final.finished:
        return None
    return final


async def handle_row(engine, sampling_params, prompt, row_dict, item,
                     file_lock, output_fh, model_alias, n_samples_requested,
                     per_sample_requests: bool = True):
    """Drive generation for one row.

    - per_sample_requests=False: one engine.generate call with the configured
      SamplingParams.n. Subject to vLLM's "sometimes returns fewer than n
      outputs" oddity for gpt-oss-family models.
    - per_sample_requests=True (default): dispatch n_samples_requested separate
      n=1 requests in parallel within this row (distinct seeds for diversity),
      gather all, synthesize a faux RequestOutput. Loses 0 sub-samples by
      construction.
    """
    if SHUTDOWN.is_set():
        return None

    if not per_sample_requests:
        request_id = str(uuid.uuid4())
        final = await _stream_one_request(engine, prompt, sampling_params, request_id)
        if final is None:
            return None
    else:
        base_seed = sampling_params.seed if sampling_params.seed is not None else 0
        sub_request_ids = [str(uuid.uuid4()) for _ in range(n_samples_requested)]

        def _make_sub(i: int) -> SamplingParams:
            sp = sampling_params.clone()
            sp.n = 1
            sp.seed = base_seed + i + 1
            return sp

        sub_coros = [
            _stream_one_request(engine, prompt, _make_sub(i), sub_request_ids[i])
            for i in range(n_samples_requested)
        ]
        sub_finals = await asyncio.gather(*sub_coros, return_exceptions=False)
        ok_outputs = []
        for sf in sub_finals:
            if sf is None or not sf.outputs:
                continue
            ok_outputs.append(sf.outputs[0])
        if not ok_outputs:
            return None
        prompt_token_ids = next(
            (sf.prompt_token_ids for sf in sub_finals
             if sf is not None and getattr(sf, "prompt_token_ids", None)),
            [],
        )
        final = _FauxRequestOutput(outputs=ok_outputs,
                                   prompt_token_ids=prompt_token_ids)

    record = build_record(row_dict, prompt, final, item, model_alias,
                          n_samples_requested)
    line = json.dumps(record, ensure_ascii=False)
    async with file_lock:
        output_fh.write(line + "\n")
        output_fh.flush()
        os.fsync(output_fh.fileno())
    return final


# ---------- per-cell processing ----------
async def process_cell(engine, tokenizer, args, item, model_config):
    label = item["label"]
    print(f"\n=== cell {label} ===", flush=True)
    pending: list[tuple[dict, str]] = item["pending"]
    if not pending:
        print(f"  (nothing to do — fully resumed in prescan)", flush=True)
        return 0

    n_samples = args.n_samples if args.n_samples is not None else item["n_samples"]
    sampling_params = make_sampling_params(
        model_config, n_samples, args.seed,
        max_tokens_override=args.max_tokens)
    print(f"  sampling: n={n_samples} max_tokens={sampling_params.max_tokens} "
          f"temperature={sampling_params.temperature} "
          f"top_p={sampling_params.top_p} top_k={sampling_params.top_k}  "
          f"skip_special_tokens={sampling_params.skip_special_tokens}", flush=True)
    mode = ("per-sample (n separate n=1 requests/row)"
            if args.per_sample_requests else "parallel (one n=N request/row)")
    print(f"  concurrency: max_concurrent={args.max_concurrent}  "
          f"prompts_to_dispatch={len(pending)}  "
          f"target_generations={len(pending) * n_samples}  "
          f"mode={mode}", flush=True)

    sem = asyncio.Semaphore(args.max_concurrent)
    file_lock = asyncio.Lock()
    stats = CellStats(total=len(pending), n_samples=n_samples)
    reporter = asyncio.create_task(
        status_reporter(stats, interval=args.status_interval, label=label))

    with open(item["output_path"], "a") as out_fh:
        async def bounded(rd, p):
            async with sem:
                return await handle_row(
                    engine, sampling_params, p, rd, item,
                    file_lock, out_fh, args.model_config, n_samples,
                    per_sample_requests=args.per_sample_requests)

        tasks = [asyncio.create_task(bounded(rd, p)) for rd, p in pending]
        try:
            for fut in asyncio.as_completed(tasks):
                try:
                    final = await fut
                except asyncio.CancelledError:
                    stats.aborted += 1
                    continue
                except Exception as e:
                    # During graceful shutdown the engine subprocess goes away,
                    # so any in-flight engine.generate() raises EngineDeadError
                    # (or similar). Those are shutdown-induced, not real errors.
                    if SHUTDOWN.is_set():
                        stats.aborted += 1
                    else:
                        stats.errors += 1
                        print(f"  [error] {e!r}", flush=True)
                    continue
                if final is not None:
                    stats.record(final)
                    if (stats.completed_rows % 16 == 0
                            or stats.completed_rows == stats.total):
                        print(f"  {stats.snapshot_line()}", flush=True)
                if SHUTDOWN.is_set():
                    print("  [shutdown] cancelling remaining tasks", flush=True)
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                    break
        finally:
            reporter.cancel()
            try:
                await reporter
            except asyncio.CancelledError:
                pass
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=args.shutdown_grace if SHUTDOWN.is_set() else None,
                )
            except asyncio.TimeoutError:
                print(f"  [shutdown] grace exceeded "
                      f"({args.shutdown_grace}s); some requests dropped",
                      flush=True)

    fr_str = "  ".join(f"{r}={n}" for r, n in sorted(stats.finish_reasons.items())) \
             or "(none)"
    elapsed = time.time() - stats.start_time
    print(f"  cell {label} done: +{stats.completed_rows}/{stats.total} rows  "
          f"{stats.errors} err  {stats.aborted} aborted  "
          f"{stats.completion_tokens} compl_tok "
          f"({stats.completion_tokens/max(elapsed,1e-3):.0f} tok/s avg)  "
          f"{format_secs(elapsed)}", flush=True)
    print(f"  finish_reasons: {fr_str}", flush=True)
    return stats.completed_rows


# ---------- signals ----------
def install_signal_handlers(loop):
    def shutdown_handler(signame: str):
        if not SHUTDOWN.is_set():
            print(f"\n[signal {signame}] graceful shutdown initiated", flush=True)
            SHUTDOWN.set()
        else:
            print(f"[signal {signame}] already shutting down", flush=True)
    loop.add_signal_handler(signal.SIGTERM, shutdown_handler, "SIGTERM")
    loop.add_signal_handler(signal.SIGINT, shutdown_handler, "SIGINT")


# ---------- engine init ----------
def build_engine(args, model_config) -> AsyncLLMEngine:
    """Build kwargs against the actual AsyncEngineArgs schema so this script
    stays compatible across vLLM versions (e.g. `disable_log_requests` vs
    `enable_log_requests`)."""
    ea_fields = {f.name for f in dataclasses.fields(AsyncEngineArgs)}
    candidate = dict(
        model=model_config["model_name"],
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=model_config.get("context_length", 32768),
        enable_prefix_caching=args.prefix_caching,
        seed=args.seed,
        # Older vLLM: `disable_log_requests=True`; newer (≥0.10): `enable_log_requests=False`.
        disable_log_requests=True,
        enable_log_requests=False,
    )
    if args.gdn_prefill_backend is not None:
        candidate["gdn_prefill_backend"] = args.gdn_prefill_backend
    ea_kwargs = {k: v for k, v in candidate.items() if k in ea_fields}
    dropped = sorted(set(candidate) - set(ea_kwargs))
    if dropped:
        print(f"[engine] dropping unsupported AsyncEngineArgs kwargs: {dropped}",
              flush=True)
    print(f"[engine] applied AsyncEngineArgs kwargs: {sorted(ea_kwargs)}",
          flush=True)
    engine_args = AsyncEngineArgs(**ea_kwargs)
    return AsyncLLMEngine.from_engine_args(engine_args)


def build_item(args) -> dict:
    """Pack CLI args into the per-cell item dict the driver expects."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "label": output_dir.name or str(output_dir),
        "input_path": Path(args.data_path),
        "output_path": output_dir / "completions.jsonl",
        "template_path": Path(args.prompt_template_path),
        "template_key": args.prompt_template_key,
        "n_samples": args.n,
        "task_name": args.task_name,
    }


# ---------- main ----------
async def amain(args: SimpleNamespace):
    """Drive one inference cell. `args` is a Namespace-like object carrying
    PR-2's per-cell flags plus our async-engine reliability knobs."""
    model_config = load_model_config(args.config_path, args.model_config)

    item = build_item(args)
    items = [item]
    print(f"Running cell {item['label']} with model={args.model_config}",
          flush=True)
    print(f"  input:  {item['input_path']}", flush=True)
    print(f"  output: {item['output_path']}", flush=True)

    # Tokenizer first — needed for prompt rendering during prescan and the
    # --dry_run prompt preview. Loads from the local HF cache (TRANSFORMERS_OFFLINE)
    # in <1s.
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"])
    print(f"[tokenizer] vocab_size={tokenizer.vocab_size} "
          f"eos={tokenizer.eos_token!r} bos={tokenizer.bos_token!r}",
          flush=True)

    # Resume scan: render every row's prompt, drop already-done ones, apply cap.
    # Runs BEFORE engine init so a fully-resumed run exits without paying the
    # model-load cost.
    total_to_run = prescan_cells(items, args, tokenizer)

    if args.dry_run:
        preview_prompts(items, tokenizer, args)
        print("[dry-run] exiting before engine init", flush=True)
        return

    if total_to_run == 0:
        print("Nothing to do — cell is fully resumed. "
              "Exiting before engine init.", flush=True)
        return

    engine = build_engine(args, model_config)
    log_engine_info(engine, model_config)

    install_signal_handlers(asyncio.get_running_loop())

    t_start = time.time()
    total_new = 0
    try:
        total_new = await process_cell(engine, tokenizer, args, item, model_config)
    finally:
        elapsed = time.time() - t_start
        print(f"\nTotal: +{total_new} new generations in "
              f"{elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
