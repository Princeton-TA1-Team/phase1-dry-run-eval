"""Inference cell driver with optional streaming verification.

Bridges the recursive pipeline to the existing async-vLLM machinery under
`contextual_drag.inference`. We build a single work-item dict (same shape that
`work_items.resolve_work_items` produces), run a quiet prompt-hash resume
scan, then drive generation through a near-verbatim copy of
`run_model.process_cell` with one substitution: the per-row handler is
`handle_row_with_verify`, which calls a `Verifier` to attach `correctness` +
`extracted_answer` to each generation BEFORE the JSONL line is flushed.

Why a copy of process_cell instead of an upstream import: process_cell hard-
codes `handle_row` as the per-row coroutine. Rather than monkey-patching the
upstream module at import time (fragile across vLLM versions), we lift the
~50-line cell body into this file. Engine, sampling-params, prompt rendering,
and prescan all come from the upstream modules unchanged.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import types
import uuid
from pathlib import Path
from typing import Optional

from vllm import SamplingParams

from contextual_drag.inference.run_model import (
    SHUTDOWN,
    _FauxRequestOutput,
    _stream_one_request,
    build_record,
)
from contextual_drag.inference.config import make_sampling_params
from contextual_drag.inference.prompts import (
    load_template,
    render_prompt,
    prompt_hash,
    load_completed_hashes,
)
from contextual_drag.inference.stats import CellStats, format_secs, status_reporter
from datasets import load_from_disk

from .verifier import Verifier


def _prescan_cell(item: dict, args, tokenizer) -> None:
    """Quiet replacement for diagnostics.prescan_cells. Renders prompts,
    drops rows whose prompt-hash is already in the on-disk JSONL, applies
    max_rows_per_cell, and stashes `pending` + `template` on the item.
    Prints one summary line."""
    enable_thinking = True if args.enable_thinking is None else args.enable_thinking
    template = load_template(item["template_path"], item["template_key"])
    dataset = load_from_disk(str(item["input_path"]))
    item["output_path"].parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_hashes(item["output_path"], item["task_name"])
    pending: list[tuple[dict, str]] = []
    for row in dataset:
        prompt = render_prompt(template, row, tokenizer, enable_thinking)
        if prompt_hash(prompt) in completed:
            continue
        pending.append((dict(row), prompt))
    n_dataset = len(dataset)
    n_done = n_dataset - len(pending)
    cap_note = ""
    if args.max_rows_per_cell is not None and len(pending) > args.max_rows_per_cell:
        cap_note = f" (capped from {len(pending)})"
        pending = pending[:args.max_rows_per_cell]
    item["template"] = template
    item["pending"] = pending
    label = f"{item['variant']}/{item['task']}"
    print(f"  resume scan {label}: dataset={n_dataset} done={n_done} "
          f"pending={len(pending)}{cap_note}", flush=True)


def make_args_namespace(*,
                        model: str,
                        seed: int = 42,
                        max_concurrent: int = 128,
                        per_sample_requests: bool = True,
                        enable_thinking: bool = True,
                        max_tokens: Optional[int] = None,
                        max_rows_per_cell: Optional[int] = None,
                        status_interval: float = 30.0,
                        shutdown_grace: float = 60.0,
                        n_samples: Optional[int] = None,
                        ) -> types.SimpleNamespace:
    """Mirror the shape of `argparse.Namespace` that `run_model.process_cell`
    and `diagnostics.prescan_cells` read from. We synthesize this at every
    cell call rather than passing the top-level CLI args through, so each
    cell can have its own per_sample_requests / max_tokens / n_samples
    overrides while sharing the same engine."""
    return types.SimpleNamespace(
        model=model,
        seed=seed,
        max_concurrent=max_concurrent,
        per_sample_requests=per_sample_requests,
        enable_thinking=enable_thinking,
        max_tokens=max_tokens,
        max_rows_per_cell=max_rows_per_cell,
        status_interval=status_interval,
        shutdown_grace=shutdown_grace,
        n_samples=n_samples,
    )


def make_work_item(*,
                   variant: str,
                   task: str,
                   input_path: Path,
                   output_path: Path,
                   template_path: Path,
                   template_key: str,
                   task_name: str,
                   n_samples: int) -> dict:
    """Same shape `work_items.resolve_work_items` returns. After we call
    `prescan_cells([item], args, tokenizer)` upstream, the item also gets
    `pending` and `template` keys populated."""
    return {
        "variant": variant,
        "task": task,
        "input_path": Path(input_path),
        "output_path": Path(output_path),
        "template_path": Path(template_path),
        "template_key": template_key,
        "n_samples": n_samples,
        "task_name": task_name,
    }


# ---- per-row handler ----
async def _drive_generation(engine, sampling_params: SamplingParams,
                            prompt: str, n_samples_requested: int,
                            per_sample_requests: bool):
    """Run the engine for one row. Returns the final RequestOutput-like object,
    or None on shutdown / total-failure. Mirrors run_model.handle_row's
    generation block but stops short of writing the JSONL line, so the caller
    can attach verification fields first."""
    if SHUTDOWN.is_set():
        return None

    if not per_sample_requests:
        request_id = str(uuid.uuid4())
        return await _stream_one_request(engine, prompt, sampling_params, request_id)

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
    return _FauxRequestOutput(outputs=ok_outputs, prompt_token_ids=prompt_token_ids)


async def handle_row_with_verify(engine, sampling_params, prompt, row_dict, item,
                                 file_lock, output_fh, model_alias,
                                 n_samples_requested, *,
                                 per_sample_requests: bool,
                                 verifier: Optional[Verifier],
                                 first_sample_holder: Optional[dict] = None):
    """Like run_model.handle_row, but inserts a verifier annotation step
    between record construction and JSONL flush.

    With `verifier=None` (strategy / filter1 cells) this is functionally
    identical to `run_model.handle_row` — we copy the body so we don't have
    to monkey-patch anything upstream.
    """
    del first_sample_holder  # no longer used; kept for signature stability
    final = await _drive_generation(engine, sampling_params, prompt,
                                    n_samples_requested, per_sample_requests)
    if final is None:
        return None

    record = build_record(row_dict, prompt, final, item, model_alias,
                          n_samples_requested)

    if verifier is not None:
        gens_key = f"{item['task_name']}_generations"
        record[gens_key] = await verifier.annotate(
            record[gens_key], ground_truth=row_dict.get("answer"))

    line = json.dumps(record, ensure_ascii=False)
    async with file_lock:
        output_fh.write(line + "\n")
        output_fh.flush()
        os.fsync(output_fh.fileno())
    return final


# ---- cell driver ----
async def process_cell_with_verify(engine, tokenizer, args, item, model_config,
                                   verifier: Optional[Verifier]):
    """Near-verbatim copy of run_model.process_cell with handle_row →
    handle_row_with_verify. See run_model.process_cell for the original."""
    label = f"{item['variant']}/{item['task']}"
    pending: list[tuple[dict, str]] = item["pending"]
    if not pending:
        print(f"  cell {label}: fully resumed (nothing to do)", flush=True)
        return 0

    n_samples = args.n_samples if args.n_samples is not None else item["n_samples"]
    sampling_params = make_sampling_params(
        model_config, n_samples, args.seed,
        max_tokens_override=args.max_tokens)
    print(f"  cell {label}: dispatching {len(pending)} rows × n={n_samples}  "
          f"(max_tok={sampling_params.max_tokens}, "
          f"max_concurrent={args.max_concurrent}, "
          f"verify={'on' if verifier is not None else 'off'})", flush=True)

    sem = asyncio.Semaphore(args.max_concurrent)
    file_lock = asyncio.Lock()
    stats = CellStats(total=len(pending), n_samples=n_samples)
    reporter = asyncio.create_task(
        status_reporter(stats, interval=args.status_interval, label=label))

    with open(item["output_path"], "a") as out_fh:
        async def bounded(rd, p):
            async with sem:
                return await handle_row_with_verify(
                    engine, sampling_params, p, rd, item,
                    file_lock, out_fh, args.model, n_samples,
                    per_sample_requests=args.per_sample_requests,
                    verifier=verifier,
                    first_sample_holder=None)

        tasks = [asyncio.create_task(bounded(rd, p)) for rd, p in pending]
        try:
            for fut in asyncio.as_completed(tasks):
                try:
                    final = await fut
                except asyncio.CancelledError:
                    stats.aborted += 1
                    continue
                except Exception as e:
                    if SHUTDOWN.is_set():
                        stats.aborted += 1
                    else:
                        stats.errors += 1
                        print(f"  [error] {e!r}", flush=True)
                    continue
                if final is not None:
                    stats.record(final)
                    log_every = 1 if stats.total <= 8 else (4 if stats.total <= 32 else 16)
                    if (stats.completed_rows % log_every == 0
                            or stats.completed_rows == stats.total):
                        print(f"  progress: {stats.snapshot_line()}", flush=True)
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

    fr_str = " ".join(f"{r}={n}" for r, n in sorted(stats.finish_reasons.items())) \
             or "(none)"
    elapsed = time.time() - stats.start_time
    err_part = f"  errors={stats.errors}" if stats.errors else ""
    abort_part = f"  aborted={stats.aborted}" if stats.aborted else ""
    print(f"  cell {label} done: {stats.completed_rows}/{stats.total} rows in "
          f"{format_secs(elapsed)} "
          f"({stats.completion_tokens/max(elapsed,1e-3):.0f} tok/s)  "
          f"finish={{{fr_str}}}{err_part}{abort_part}",
          flush=True)
    return stats.completed_rows


# ---- top-level cell entry: targeted makeup ----
async def run_makeup_cell(*,
                          engine,
                          tokenizer,
                          model_alias: str,
                          model_config: dict,
                          rows: list[dict],
                          output_path: Path,
                          template_path: Path,
                          template_key: str,
                          task_name: str,
                          n_samples: int,
                          variant: str,
                          task: str,
                          max_concurrent: int,
                          seed: int,
                          max_tokens: Optional[int] = None,
                          per_sample_requests: bool = True,
                          enable_thinking: bool = True,
                          status_interval: float = 30.0,
                          shutdown_grace: float = 60.0,
                          verifier: Optional[Verifier] = None) -> int:
    """Append n_samples generations for each row in `rows` to `output_path`.

    Bypasses the prompt-hash resume scan — the caller has already determined
    which rows still need more samples. Each call is one batch; the outer loop
    in pipeline.py decides whether to call again. Idempotency comes from the
    outer loop re-reading the JSONL and re-checking parsability after each
    batch, so a crash mid-batch just means the next attempt sees the partial
    progress and continues.
    """
    if not rows:
        return 0
    enable_thinking_eff = True if enable_thinking is None else enable_thinking
    template = load_template(template_path, template_key)
    pending: list[tuple[dict, str]] = []
    for row in rows:
        prompt = render_prompt(template, row, tokenizer, enable_thinking_eff)
        pending.append((dict(row), prompt))
    item = make_work_item(
        variant=variant, task=task,
        input_path=Path("<makeup>"),  # not read; required by item shape
        output_path=Path(output_path),
        template_path=Path(template_path),
        template_key=template_key,
        task_name=task_name, n_samples=n_samples,
    )
    item["template"] = template
    item["pending"] = pending
    args = make_args_namespace(
        model=model_alias, seed=seed,
        max_concurrent=max_concurrent,
        per_sample_requests=per_sample_requests,
        enable_thinking=enable_thinking_eff,
        max_tokens=max_tokens,
        status_interval=status_interval,
        shutdown_grace=shutdown_grace,
        max_rows_per_cell=None,
        n_samples=n_samples,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    return await process_cell_with_verify(engine, tokenizer, args, item,
                                          model_config, verifier=verifier)


# ---- top-level cell entry ----
async def run_inference_cell(*,
                             engine,
                             tokenizer,
                             model_alias: str,
                             model_config: dict,
                             input_path: Path,
                             output_path: Path,
                             template_path: Path,
                             template_key: str,
                             task_name: str,
                             n_samples: int,
                             variant: str,
                             task: str,
                             max_concurrent: int,
                             seed: int,
                             max_tokens: Optional[int] = None,
                             per_sample_requests: bool = True,
                             enable_thinking: bool = True,
                             status_interval: float = 30.0,
                             shutdown_grace: float = 60.0,
                             max_rows_per_cell: Optional[int] = None,
                             verifier: Optional[Verifier] = None) -> int:
    """Build a work item, run prescan, then drive the cell. Returns the number
    of new rows generated this call. Resume is automatic via prescan_cells."""
    item = make_work_item(
        variant=variant, task=task,
        input_path=input_path, output_path=output_path,
        template_path=template_path, template_key=template_key,
        task_name=task_name, n_samples=n_samples,
    )
    args = make_args_namespace(
        model=model_alias, seed=seed,
        max_concurrent=max_concurrent,
        per_sample_requests=per_sample_requests,
        enable_thinking=enable_thinking,
        max_tokens=max_tokens,
        status_interval=status_interval,
        shutdown_grace=shutdown_grace,
        max_rows_per_cell=max_rows_per_cell,
        n_samples=n_samples,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _prescan_cell(item, args, tokenizer)
    return await process_cell_with_verify(engine, tokenizer, args, item,
                                          model_config, verifier=verifier)
