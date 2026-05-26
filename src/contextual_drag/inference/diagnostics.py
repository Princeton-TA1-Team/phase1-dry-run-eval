"""Diagnostic / preview output that's not part of the core inference loop.

- log_engine_info:  one-shot post-init dump of engine config + GPU memory.
- prescan_cells:    load dataset + scan completed JSONL + render prompts +
                    apply --max_rows_per_cell cap, caching `pending` and
                    `template` on each work item. Prints a summary table.
                    Runs BEFORE engine init so a fully-resumed run exits
                    without paying the model-load cost.
- preview_prompts:  print the first row's rendered prompt for each cell.
                    Used by --dry_run to confirm prompt structure pre-launch.
"""

from typing import Any

from datasets import load_from_disk

from contextual_drag.inference.prompts import (
    load_template,
    render_prompt,
    prompt_hash,
    load_completed_hashes,
    template_fields,
)
from contextual_drag.inference.stats import format_gb


def log_engine_info(engine, model_config: dict) -> None:
    """Print engine + GPU state at startup."""
    print(f"[engine] model={model_config['model_name']}", flush=True)

    # vLLM internals are version-fragile; probe defensively.
    inner = getattr(engine, "engine", None)
    parts = []
    for attr_chain, field_label in [
        ("model_config.dtype", "dtype"),
        ("model_config.max_model_len", "max_model_len"),
        ("cache_config.gpu_memory_utilization", "gpu_mem_util"),
        ("cache_config.num_gpu_blocks", "kv_cache_blocks"),
        ("cache_config.block_size", "kv_block_size"),
        ("cache_config.enable_prefix_caching", "prefix_cache"),
        ("parallel_config.tensor_parallel_size", "tp"),
        ("parallel_config.pipeline_parallel_size", "pp"),
    ]:
        obj: Any = inner
        try:
            for a in attr_chain.split("."):
                obj = getattr(obj, a)
            parts.append(f"{field_label}={obj}")
        except (AttributeError, TypeError):
            continue
    if parts:
        print(f"[engine] {'  '.join(parts)}", flush=True)

    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                used = total - free
                name = torch.cuda.get_device_name(i)
                print(f"[engine] cuda:{i} {name}  "
                      f"{format_gb(used)}/{format_gb(total)} used "
                      f"({100*used/total:.0f}%)", flush=True)
    except Exception as e:
        print(f"[engine] cuda mem probe skipped: {e!r}", flush=True)


def prescan_cells(items: list[dict], args, tokenizer) -> int:
    """For every cell, load the dataset, scan the on-disk JSONL for completed
    prompt-hashes, render each row's prompt, drop already-done rows, apply the
    --max_rows_per_cell cap, and cache the result on the work item as
    `item['pending']` and `item['template']`. Returns the total number of new
    rows this run will dispatch."""
    enable_thinking = True if args.enable_thinking is None else args.enable_thinking
    print("\n=== Resume scan (pre-engine) ===", flush=True)
    header = (f"  {'cell':<28} {'dataset':>8} {'done':>6} "
              f"{'remaining':>10} {'this_run':>9}  notes")
    print(header, flush=True)
    print("  " + "-" * (len(header) - 2), flush=True)
    total_this_run = 0
    for item in items:
        label = item["label"]
        template = load_template(item["template_path"], item["template_key"])
        dataset = load_from_disk(str(item["input_path"]))
        n_dataset = len(dataset)

        item["output_path"].parent.mkdir(parents=True, exist_ok=True)
        completed = load_completed_hashes(item["output_path"], item["task_name"])

        pending_all: list[tuple[dict, str]] = []
        for row in dataset:
            prompt = render_prompt(template, row, tokenizer, enable_thinking)
            if prompt_hash(prompt) in completed:
                continue
            pending_all.append((dict(row), prompt))
        n_done = n_dataset - len(pending_all)

        if args.max_rows_per_cell is not None and len(pending_all) > args.max_rows_per_cell:
            pending = pending_all[:args.max_rows_per_cell]
            note = f"cap={args.max_rows_per_cell}"
        else:
            pending = pending_all
            note = "no cap"

        item["template"] = template
        item["pending"] = pending
        total_this_run += len(pending)
        print(f"  {label:<28} {n_dataset:>8} {n_done:>6} "
              f"{len(pending_all):>10} {len(pending):>9}  {note}",
              flush=True)
    print("  " + "-" * (len(header) - 2), flush=True)
    print(f"  TOTAL new generations this run: "
          f"{total_this_run} rows  (× n_samples each)\n", flush=True)
    return total_this_run


def preview_prompts(items: list[dict], tokenizer, args,
                    head_chars: int = 1500, tail_chars: int = 800) -> None:
    """Print the first row's rendered prompt for each cell. Used by --dry_run
    to verify the prompt is what we expect. Long prompts are head+tail elided
    so the framing instruction, problem, draft prefix, draft tail, and
    trailing instruction are all visible."""
    enable_thinking = True if args.enable_thinking is None else args.enable_thinking
    print("=== Prompt preview (first row per cell) ===", flush=True)
    for item in items:
        label = item["label"]
        if item.get("pending"):
            row, prompt = item["pending"][0]
        else:
            template = item.get("template") or load_template(
                item["template_path"], item["template_key"])
            dataset = load_from_disk(str(item["input_path"]))
            if len(dataset) == 0:
                print(f"\n--- {label} ---  (dataset empty, no preview)", flush=True)
                continue
            row = dict(dataset[0])
            prompt = render_prompt(template, row, tokenizer, enable_thinking)

        template = load_template(item["template_path"], item["template_key"])
        arg_fields = template_fields(template)

        try:
            n_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        except Exception:
            n_tokens = "?"

        print(f"\n--- {label} ---", flush=True)
        print(f"  template_key:    {item['template_key']}", flush=True)
        print(f"  template fields: arg_{{{', '.join(arg_fields)}}}", flush=True)
        print(f"  row keys:        {sorted(row)}", flush=True)
        print(f"  prompt size:     {len(prompt)} chars, ~{n_tokens} tokens",
              flush=True)
        print(f"  --- BEGIN PROMPT ---", flush=True)
        if len(prompt) > head_chars + tail_chars + 50:
            elided = len(prompt) - head_chars - tail_chars
            print(prompt[:head_chars], flush=True)
            print(f"\n  [... {elided} chars elided ...]\n", flush=True)
            print(prompt[-tail_chars:], flush=True)
        else:
            print(prompt, flush=True)
        print(f"  --- END PROMPT ---", flush=True)
    print("", flush=True)
