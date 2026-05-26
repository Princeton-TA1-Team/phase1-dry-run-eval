"""Per-task orchestrator for the context-manipulation pipelines.

Two variants, both linear (no recursion, no makeup):

  cm_filter1 (3 inference stages):
      load F1 init  →  strategy (n=1)  →  join strategy  →  filter1 (n=1)  →  solve (n=8)  →  eval

  cm_revise1 (2 inference stages):
      load F1 init  →  revise1 (n=1)  →  solve (n=8)  →  eval

Faithful to upstream `auto_context_manipulation_lib.sh`:
  - Init data: data/self_2f/<task>_<modelid>/minimal_aggregated_data_T0_F1_flattend_from_F2.ds
    (T=0 F=2 sampling, then F2→F1 flatten — 2 rows per problem with one
    incorrect trajectory each).
  - Templates per (variant, step) with task-specific overrides for crux-i and
    gpqa/mmlu — same key set upstream selects.
  - Solve uses `metacognitive_filter1_solve` family for BOTH variants (upstream
    template-family reuse: revise1's "filtered_traj1" column feeds the same
    solve template as filter1's).
  - n_samples_solve = 8 (`SOLVE_N_SAMPLES`).
  - JSONL key prefix `<step>_response` (matches upstream's `--task-name`).

Reuses `mitigation.pipeline.{inference_step.run_inference_cell, flatten,
verifier.Verifier, stage_state}`. cm-specific solve flatten + summary live in
this package because they take `task_name` as an explicit argument (vs hard-
coded `round{r}_response` upstream).

Per-stage artifact-presence resume + async vLLM per-row prompt-hash resume.
"""

from __future__ import annotations

import contextlib
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, load_from_disk

from . import flatten as rf1_flatten
from . import stage_state
from . import summary as cm_summary
from .inference_step import run_inference_cell
from .verifier import Verifier


VARIANTS = ("cm_filter1", "cm_revise1")
SOLVE_TASK_NAME = "solve_response"
SOLVE_N_SAMPLES = 8


@dataclass
class TaskConfig:
    task: str
    modelid: str
    variant: str  # one of VARIANTS
    init_input_ds: Path             # F1 .ds (no preprocessing, just load)
    intermediate_dir: Path          # mitigation/runs/<variant>/<task>_<modelid>/
    results_dir: Path               # data/results/<variant>/<task>_<modelid>/
    template_path: Path
    inference_alias: str            # canonical model name (Gemma4_E4B, …)
    n_samples_solve: int
    max_concurrent: int
    seed: int
    verify_workers: int = 8


def _solve_template_key(task: str) -> str:
    """Solve template family — same selection rule for cm_filter1 and cm_revise1.

    Upstream `auto_context_manipulation_lib.sh:158-167, 205-213`.
    """
    if task == "crux-i":
        return "metacognitive_filter1_solve_crux_input"
    if task in ("gpqa", "mmlu"):
        return "metacognitive_filter1_solve_qa_mc"
    return "metacognitive_filter1_solve"


def _revise1_template_key(task: str) -> str:
    """revise1 stage — qa_mc override for gpqa/mmlu only.

    Upstream `auto_context_manipulation_lib.sh:196-203`. Note: NO crux-i
    override despite the template existing — upstream skips it deliberately,
    so we do too.
    """
    if task in ("gpqa", "mmlu"):
        return "revise_revise1_qa_mc"
    return "revise_revise1"


def _fmt_dur(s: float) -> str:
    if s < 60: return f"{s:.1f}s"
    if s < 3600: return f"{s/60:.1f}m"
    return f"{s/3600:.2f}h"


@contextlib.contextmanager
def _timed(_label: str):
    t0 = time.time()
    yield lambda: time.time() - t0


def _save_ds_atomic(ds: Dataset, path: Path) -> None:
    """save_to_disk, but wipe an incomplete prior write first. Mirrors
    recursive_filter1's helper so the same ds_complete check applies."""
    path = Path(path)
    if path.exists() and not stage_state.ds_complete(path):
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(path))


def _dedup_first_per_id(ds: Dataset, id_column: str = "id") -> Dataset:
    """Keep the first row per id. Used to dedupe the F1 init ds before the
    strategy stage so we don't generate the same strategy twice per problem
    (upstream pays this cost; we save it but the joined filter_input is bit-
    equivalent because the strategy template depends only on `problem`).
    """
    seen: set = set()
    rows = []
    for row in ds:
        pid = row[id_column]
        if pid in seen:
            continue
        seen.add(pid)
        rows.append(dict(row))
    return Dataset.from_list(rows)


def _join_column(left: Dataset, right_path: Path,
                 right_column: str,
                 output_column: str | None = None,
                 id_column: str = "id") -> Dataset:
    """Inner join: copy `right[right_column]` onto each `left` row by id.

    Like `mitigation.pipeline.aggregate.join_strategy` but without the
    duplicate-id rejection (we feed it a deduplicated `right` ds, but `left`
    has 2 rows per id from the F1 flatten — both should get the same column
    value). Errors if any left id is missing on the right side.
    """
    if output_column is None:
        output_column = right_column
    right = load_from_disk(str(right_path))
    if id_column not in right.column_names:
        raise ValueError(f"right ds missing id column {id_column!r}")
    if right_column not in right.column_names:
        raise ValueError(f"right ds missing {right_column!r}")

    right_ids = list(right[id_column])
    dups = [pid for pid, c in Counter(right_ids).items() if c > 1]
    if dups:
        raise ValueError(
            f"right ds has duplicate ids (dedup before join): "
            f"{dups[:10]} (total {len(dups)})")
    id_to_value = {row[id_column]: row[right_column] for row in right}

    missing = sorted({row[id_column] for row in left
                      if row[id_column] not in id_to_value})
    if missing:
        raise ValueError(
            f"join missing for {len(missing)} left ids (first 10): "
            f"{missing[:10]}")

    out_rows = []
    for row in left:
        new_row = dict(row)
        new_row[output_column] = id_to_value[row[id_column]]
        out_rows.append(new_row)
    return Dataset.from_list(out_rows)


async def run_task(engine, tokenizer, cfg: TaskConfig, model_config: dict,
                   max_rows_per_cell=None, max_tokens=None) -> None:
    if cfg.variant not in VARIANTS:
        raise ValueError(f"unknown variant {cfg.variant!r}; expected one of {VARIANTS}")

    cfg.intermediate_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    cap_note = ""
    if max_rows_per_cell is not None:
        cap_note += f"  max_rows_per_cell={max_rows_per_cell}"
    if max_tokens is not None:
        cap_note += f"  max_tokens={max_tokens}"
    print(f"\n========================================================", flush=True)
    print(f"TASK {cfg.task}  variant={cfg.variant}  modelid={cfg.modelid}{cap_note}",
          flush=True)
    print(f"  init: {cfg.init_input_ds}", flush=True)
    print(f"  intermediate: {cfg.intermediate_dir}", flush=True)
    print(f"  results:      {cfg.results_dir}", flush=True)
    print(f"========================================================", flush=True)

    if not cfg.init_input_ds.is_dir():
        raise SystemExit(f"missing init F1 dataset for {cfg.task}/{cfg.modelid}: "
                         f"{cfg.init_input_ds}")
    init_n = len(load_from_disk(str(cfg.init_input_ds)))
    print(f"[init] {init_n} rows loaded from F1 ds", flush=True)
    if init_n == 0:
        print(f"[skip] empty cohort — nothing to run for this cell", flush=True)
        return

    t_task_start = time.time()

    if cfg.variant == "cm_filter1":
        solve_input_ds = await _run_cm_filter1_stages(
            engine, tokenizer, cfg, model_config,
            max_rows_per_cell=max_rows_per_cell, max_tokens=max_tokens,
        )
    else:  # cm_revise1
        solve_input_ds = await _run_cm_revise1_stages(
            engine, tokenizer, cfg, model_config,
            max_rows_per_cell=max_rows_per_cell, max_tokens=max_tokens,
        )

    await _run_solve_stage(
        engine, tokenizer, cfg, model_config,
        solve_input_ds=solve_input_ds,
        max_rows_per_cell=max_rows_per_cell, max_tokens=max_tokens,
    )

    # Always rewrite — single pass over completions.jsonl, idempotent, and
    # ensures the artifacts reflect the latest solve state when a chain
    # restarts mid-solve.
    print(f"\n[eval] writing summary artifacts", flush=True)
    solve_jsonl = cfg.results_dir / "completions.jsonl"
    cm_summary.write_artifacts(
        cell_dir=cfg.results_dir,
        solve_jsonl=solve_jsonl,
        task_name=SOLVE_TASK_NAME,
        model_alias=cfg.inference_alias,
    )

    print(f"\nTASK {cfg.task} ({cfg.variant}) done in "
          f"{_fmt_dur(time.time() - t_task_start)}", flush=True)


async def _run_cm_filter1_stages(engine, tokenizer, cfg: TaskConfig,
                                  model_config: dict, *,
                                  max_rows_per_cell, max_tokens) -> Path:
    """Run strategy + join + filter1. Returns the .ds path that feeds solve."""
    out = cfg.intermediate_dir
    init_dedup_path = out / "init_dedup.ds"
    strategy_dir = out / "strategy"
    strategy_jsonl = strategy_dir / "dataset.jsonl"
    strategy_ds_path = out / "strategy.ds"
    filter_input_path = out / "filter_input.ds"
    filter1_dir = out / "filter1"
    filter1_jsonl = filter1_dir / "dataset.jsonl"
    filter1_ds_path = out / "filter1.ds"

    print(f"\n[1a] dedup init by id", flush=True)
    if not stage_state.ds_complete(init_dedup_path):
        with _timed("1a") as elapsed:
            init_ds = load_from_disk(str(cfg.init_input_ds))
            dedup = _dedup_first_per_id(init_ds)
            _save_ds_atomic(dedup, init_dedup_path)
            print(f"  dedup {len(init_ds)} -> {len(dedup)} unique problems "
                  f"in {_fmt_dur(elapsed())}", flush=True)
    else:
        print(f"  skip ({len(load_from_disk(str(init_dedup_path)))} unique problems "
              f"already on disk)", flush=True)

    print(f"\n[1b] strategy inference", flush=True)
    if not stage_state.ds_complete(strategy_ds_path):
        with _timed("1b") as elapsed:
            await run_inference_cell(
                engine=engine, tokenizer=tokenizer,
                model_alias=cfg.inference_alias, model_config=model_config,
                input_path=init_dedup_path,
                output_path=strategy_jsonl,
                template_path=cfg.template_path,
                template_key="metacognitive_filter_strategy",
                task_name="strategy_response",
                n_samples=1,
                variant=f"{cfg.variant}_strategy", task=cfg.task,
                max_concurrent=cfg.max_concurrent, seed=cfg.seed,
                per_sample_requests=False,
                verifier=None,
                max_rows_per_cell=max_rows_per_cell,
                max_tokens=max_tokens,
            )
            s_ds = rf1_flatten.inference_jsonl_to_ds(
                strategy_jsonl,
                task_name="strategy_response",
                final_column="strategy")
            _save_ds_atomic(s_ds, strategy_ds_path)
            print(f"  strategy ds: {len(s_ds)} rows in {_fmt_dur(elapsed())}",
                  flush=True)
    else:
        print(f"  skip ({len(load_from_disk(str(strategy_ds_path)))} strategies "
              f"already on disk)", flush=True)

    print(f"\n[2] join strategy onto init F1", flush=True)
    if not stage_state.ds_complete(filter_input_path):
        with _timed("2") as elapsed:
            init_ds = load_from_disk(str(cfg.init_input_ds))
            joined = _join_column(init_ds, strategy_ds_path,
                                  right_column="strategy",
                                  output_column="strategy")
            _save_ds_atomic(joined, filter_input_path)
            print(f"  joined {len(joined)} rows in {_fmt_dur(elapsed())}",
                  flush=True)
    else:
        print(f"  skip ({len(load_from_disk(str(filter_input_path)))} already joined)",
              flush=True)

    print(f"\n[3] filter1 inference", flush=True)
    if not stage_state.ds_complete(filter1_ds_path):
        with _timed("3") as elapsed:
            await run_inference_cell(
                engine=engine, tokenizer=tokenizer,
                model_alias=cfg.inference_alias, model_config=model_config,
                input_path=filter_input_path,
                output_path=filter1_jsonl,
                template_path=cfg.template_path,
                template_key="metacognitive_filter_filter1",
                task_name="filter1_response",
                n_samples=1,
                variant=f"{cfg.variant}_filter1", task=cfg.task,
                max_concurrent=cfg.max_concurrent, seed=cfg.seed,
                per_sample_requests=False,
                verifier=None,
                max_rows_per_cell=max_rows_per_cell,
                max_tokens=max_tokens,
            )
            f_ds = rf1_flatten.inference_jsonl_to_ds(
                filter1_jsonl,
                task_name="filter1_response",
                final_column="filtered_traj1")
            _save_ds_atomic(f_ds, filter1_ds_path)
            print(f"  filter1 ds: {len(f_ds)} rows in {_fmt_dur(elapsed())}",
                  flush=True)
    else:
        print(f"  skip ({len(load_from_disk(str(filter1_ds_path)))} filtered drafts "
              f"already on disk)", flush=True)

    return filter1_ds_path


async def _run_cm_revise1_stages(engine, tokenizer, cfg: TaskConfig,
                                  model_config: dict, *,
                                  max_rows_per_cell, max_tokens) -> Path:
    """Run revise1. Returns the .ds path that feeds solve."""
    out = cfg.intermediate_dir
    revise1_dir = out / "revise1"
    revise1_jsonl = revise1_dir / "dataset.jsonl"
    revise1_ds_path = out / "revise1.ds"

    template_key = _revise1_template_key(cfg.task)
    print(f"\n[1] revise1 inference  (template={template_key})", flush=True)
    if not stage_state.ds_complete(revise1_ds_path):
        with _timed("1") as elapsed:
            await run_inference_cell(
                engine=engine, tokenizer=tokenizer,
                model_alias=cfg.inference_alias, model_config=model_config,
                input_path=cfg.init_input_ds,
                output_path=revise1_jsonl,
                template_path=cfg.template_path,
                template_key=template_key,
                task_name="revise1_response",
                n_samples=1,
                variant=f"{cfg.variant}_revise1", task=cfg.task,
                max_concurrent=cfg.max_concurrent, seed=cfg.seed,
                per_sample_requests=False,
                verifier=None,
                max_rows_per_cell=max_rows_per_cell,
                max_tokens=max_tokens,
            )
            r_ds = rf1_flatten.inference_jsonl_to_ds(
                revise1_jsonl,
                task_name="revise1_response",
                final_column="filtered_traj1")
            _save_ds_atomic(r_ds, revise1_ds_path)
            print(f"  revise1 ds: {len(r_ds)} rows in {_fmt_dur(elapsed())}",
                  flush=True)
    else:
        print(f"  skip ({len(load_from_disk(str(revise1_ds_path)))} revised drafts "
              f"already on disk)", flush=True)

    return revise1_ds_path


async def _run_solve_stage(engine, tokenizer, cfg: TaskConfig,
                            model_config: dict, *,
                            solve_input_ds: Path,
                            max_rows_per_cell, max_tokens) -> None:
    """Solve cell with streaming math/crux verification. Writes
    `<results_dir>/completions.jsonl` directly so the registry's xz-first
    rule promotes the cell once eval artifacts land."""
    solve_jsonl = cfg.results_dir / "completions.jsonl"
    template_key = _solve_template_key(cfg.task)

    expected_rows = len(load_from_disk(str(solve_input_ds)))
    if max_rows_per_cell is not None:
        expected_rows = min(expected_rows, max_rows_per_cell)

    print(f"\n[4] solve inference (n={cfg.n_samples_solve}, template={template_key})",
          flush=True)
    if stage_state.solve_raw_complete(solve_jsonl,
                                       task_name=SOLVE_TASK_NAME,
                                       expected_rows=expected_rows,
                                       n_samples=cfg.n_samples_solve):
        print(f"  skip ({expected_rows} solves × {cfg.n_samples_solve} samples "
              f"already on disk)", flush=True)
        return

    verifier = Verifier(max_workers=cfg.verify_workers)
    try:
        with _timed("4") as elapsed:
            await run_inference_cell(
                engine=engine, tokenizer=tokenizer,
                model_alias=cfg.inference_alias, model_config=model_config,
                input_path=solve_input_ds,
                output_path=solve_jsonl,
                template_path=cfg.template_path,
                template_key=template_key,
                task_name=SOLVE_TASK_NAME,
                n_samples=cfg.n_samples_solve,
                variant=f"{cfg.variant}_solve", task=cfg.task,
                max_concurrent=cfg.max_concurrent, seed=cfg.seed,
                per_sample_requests=True,
                verifier=verifier,
                max_rows_per_cell=max_rows_per_cell,
                max_tokens=max_tokens,
            )
            print(f"  solve cell finished in {_fmt_dur(elapsed())}", flush=True)
    finally:
        verifier.close()
