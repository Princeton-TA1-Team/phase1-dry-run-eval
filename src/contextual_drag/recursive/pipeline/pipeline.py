"""Per-task orchestrator for the recursive self-improvement loop.

Two variants share the same engine, resume contracts, and makeup machinery
and differ only in which stages run inside the loop and which solve-prompt
family is used.

  rf1 (full recursive_filter1 pipeline):
      Stage 0  build round-0 draft pool
      Stage 1  per-problem strategy inference (one-time)
      Loop t=1..max_recursive_steps:
        2a   build round-t draft pool (t=1 reuses Stage 0; t>=2 flattens prev solve)
        2b   join the strategy column
        2c   filter1 inference (n=1, "context cleaning" of the prior draft)
        2d   solve inference (n=N_SAMPLES) via `metacognitive_filter1_solve`
             template family, with streaming math verification
        2e   summary writer (3 evaluated_* artifacts)
      + targeted makeup re-generation per round

  naive (recursion without context cleaning):
      Stage 0  build round-0 draft pool
      Loop t=1..max_recursive_steps:
        2a   build round-t draft pool (t=1 reuses Stage 0; t>=2 flattens prev solve)
        2d   solve inference (n=N_SAMPLES) via `1f` template family, fed the
             prior round's incorrect trajectory directly (no strategy, no
             filter1 — the whole point of the baseline)
        2e   summary writer
      + same makeup machinery

Every stage is gated by an artifact-presence check; resumption walks the
checks and skips done work. The engine is shared across all stages and all
tasks (engine init lives in run_pipeline.py).
"""

from __future__ import annotations

import contextlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, load_from_disk

from . import aggregate, flatten, stage_state, summary
from .inference_step import run_inference_cell, run_makeup_cell
from .verifier import Verifier


VARIANTS = ("rf1", "naive")


def _fmt_dur(s: float) -> str:
    if s < 60: return f"{s:.1f}s"
    if s < 3600: return f"{s/60:.1f}m"
    return f"{s/3600:.2f}h"


@contextlib.contextmanager
def _timed(_label: str):
    """Capture the elapsed time of a block. Yields a callable that returns
    elapsed seconds since entry — caller decides how to format it."""
    t0 = time.time()
    yield lambda: time.time() - t0


@dataclass
class TaskConfig:
    task: str
    init_input_ds: Path           # <init_data_root>/<task>/processed_flattened_init_responses.ds
    output_root: Path             # <output_root>/<run_id>/<task>/
    template_path: Path
    init_alias: str               # e.g. "GPT_OSS_20B" — the alias under which
                                  # round-0 init responses were originally generated.
    inference_alias: str          # e.g. "GPT_OSS_20B_recursive" — the alias used
                                  # for strategy/filter1/solve inference in this run.
    run_id: int
    max_recursive_steps: int
    n_samples_solve: int
    max_concurrent: int
    seed: int
    variant: str = "rf1"          # one of VARIANTS
    verify_workers: int = 8
    # Targeted re-generation for the solve cell. After the main solve pass
    # completes, any problem id that has zero samples passing the aggregator
    # filter (model-alias + finish=stop + thinking=parsable) gets up to
    # `makeup_max_attempts` additional batches, each generating
    # `makeup_batch_size` samples per still-unsatisfied id. Without this,
    # such problems would silently drop out of the recursive cohort.
    makeup_max_attempts: int = 4
    makeup_batch_size: int = 4


# ---- small helpers ----
def _save_ds_atomic(ds: Dataset, path: Path) -> None:
    """save_to_disk, but wipe an incomplete prior write first. ds_complete()
    treats a directory missing state.json/dataset_info.json as not-done, so a
    partial dir from a prior interrupted run is the case we need to clean up."""
    path = Path(path)
    if path.exists() and not stage_state.ds_complete(path):
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(path))


def _solve_template_key(task: str, variant: str) -> str:
    """Solve template family per variant.

    rf1   → `metacognitive_filter1_solve*` (consumes `filtered_traj1`)
    naive → `1f*`                          (consumes raw `traj1`)

    Task-specific overrides mirror the upstream context-manipulation rule
    (`gpqa`/`mmlu` → qa_mc; `crux-i` → crux_input).
    """
    if variant == "naive":
        if task == "crux-i":
            return "1f_crux_input"
        if task in ("gpqa", "mmlu"):
            return "1f_qa_mc"
        return "1f"
    # rf1 (and any future filter-style variant)
    if task == "crux-i":
        return "metacognitive_filter1_solve_crux_input"
    if task in ("gpqa", "mmlu"):
        return "metacognitive_filter1_solve_qa_mc"
    return "metacognitive_filter1_solve"


# ---- per-task driver ----
async def run_task(engine, tokenizer, cfg: TaskConfig, model_config: dict,
                   max_rows_per_cell=None, max_tokens=None) -> None:
    if cfg.variant not in VARIANTS:
        raise ValueError(
            f"unknown variant {cfg.variant!r}; expected one of {VARIANTS}")

    out = cfg.output_root
    out.mkdir(parents=True, exist_ok=True)
    t_task_start = time.time()

    cap_note = ""
    if max_rows_per_cell is not None:
        cap_note += f"  max_rows_per_cell={max_rows_per_cell}"
    if max_tokens is not None:
        cap_note += f"  max_tokens={max_tokens}"
    print(f"\n========================================================", flush=True)
    print(f"TASK {cfg.task}  variant={cfg.variant}  run_id={cfg.run_id}  "
          f"steps={cfg.max_recursive_steps}  "
          f"n_samples_solve={cfg.n_samples_solve}{cap_note}", flush=True)
    print(f"  init_alias={cfg.init_alias}  inference_alias={cfg.inference_alias}",
          flush=True)
    print(f"  output: {out}", flush=True)
    print(f"========================================================", flush=True)

    # ---- Stage 0: round-0 draft pool ----
    # max_rows_per_cell, when set, also truncates the Stage 0 draft pool. This
    # is the single knob that caps the universe of problems in test mode —
    # every downstream stage naturally inherits the cap because it consumes
    # this dataset (or its joined/filtered descendants).
    round0_drafts_path = out / "round0_drafts.ds"
    print(f"\n[stage 0] round-0 draft pool", flush=True)
    if not stage_state.ds_complete(round0_drafts_path):
        with _timed("stage 0") as elapsed:
            init_ds = load_from_disk(str(cfg.init_input_ds))
            n_init_problems = len(set(init_ds["id"]))
            ds = aggregate.pick_one_per_problem(
                cfg.init_input_ds, model_alias=cfg.init_alias,
                round_num=0, seed=cfg.run_id)
            if len(ds) < n_init_problems:
                missing = sorted(set(init_ds["id"]) - set(ds["id"]))
                preview = missing[:10]
                print(f"  [warn] round-0 dropouts: {len(missing)}/"
                      f"{n_init_problems} problems have 0 parsable+stop "
                      f"samples for init_alias={cfg.init_alias!r}. "
                      f"first ids: {preview}. these are permanently absent "
                      f"from the cohort (no makeup at round 0 — the init "
                      f"dataset was generated externally and we don't own "
                      f"its prompt template).", flush=True)
            if max_rows_per_cell is not None and len(ds) > max_rows_per_cell:
                print(f"  truncating {len(ds)} -> {max_rows_per_cell} "
                      f"(max_rows_per_cell cap)", flush=True)
                ds = ds.select(range(max_rows_per_cell))
            _save_ds_atomic(ds, round0_drafts_path)
            print(f"  wrote {len(ds)} drafts in {_fmt_dur(elapsed())}",
                  flush=True)
    else:
        existing = load_from_disk(str(round0_drafts_path))
        if (max_rows_per_cell is not None
                and len(existing) > max_rows_per_cell):
            raise SystemExit(
                f"round-0 drafts on disk has {len(existing)} rows but "
                f"max_rows_per_cell={max_rows_per_cell}. Mismatch — wipe "
                f"{round0_drafts_path.parent} and re-run.")
        print(f"  skip ({len(existing)} drafts already on disk)", flush=True)

    # ---- Stage 0v: verify round-0 picked trajectories ----
    # Each draft row has a `traj1` (post-thinking text from the picked init
    # sample) and a `traj1_correctness` field carried over from the init
    # dataset. We re-verify under our own Verifier so the round-0 baseline
    # numbers are computed with the same math_verify config we use for solve
    # rounds, and so log readers see a directly comparable [pass@1] line.
    await _run_stage_0v(cfg, round0_drafts_path)

    # ---- Stage 1: strategy (rf1 only; once per task) ----
    strategy_dir = out / "strategy"
    strategy_jsonl = strategy_dir / "dataset.jsonl"
    strategy_ds_path = strategy_dir / "strategy.ds"
    if cfg.variant == "rf1":
        print(f"\n[stage 1] strategy inference", flush=True)
        if not stage_state.ds_complete(strategy_ds_path):
            with _timed("stage 1") as elapsed:
                await run_inference_cell(
                    engine=engine, tokenizer=tokenizer,
                    model_alias=cfg.inference_alias, model_config=model_config,
                    input_path=round0_drafts_path,
                    output_path=strategy_jsonl,
                    template_path=cfg.template_path,
                    template_key="metacognitive_filter_strategy",
                    task_name="strategy_response",
                    n_samples=1,
                    variant="strategy", task=cfg.task,
                    max_concurrent=cfg.max_concurrent, seed=cfg.seed,
                    per_sample_requests=False,
                    verifier=None,
                    max_rows_per_cell=max_rows_per_cell,
                    max_tokens=max_tokens,
                )
                s_ds = flatten.inference_jsonl_to_ds(
                    strategy_jsonl,
                    task_name="strategy_response",
                    final_column="strategy")
                _save_ds_atomic(s_ds, strategy_ds_path)
                print(f"  strategy ds: {len(s_ds)} rows in {_fmt_dur(elapsed())}",
                      flush=True)
        else:
            existing = load_from_disk(str(strategy_ds_path))
            print(f"  skip ({len(existing)} strategies already on disk)", flush=True)
    else:
        print(f"\n[stage 1] strategy skipped (variant={cfg.variant})", flush=True)

    # ---- Recursive rounds ----
    for step in range(1, cfg.max_recursive_steps + 1):
        await _run_round(engine, tokenizer, cfg, model_config,
                         step=step,
                         round0_drafts_path=round0_drafts_path,
                         strategy_ds_path=strategy_ds_path,
                         out=out,
                         max_rows_per_cell=max_rows_per_cell,
                         max_tokens=max_tokens)

    print(f"\nTASK {cfg.task} done in {_fmt_dur(time.time() - t_task_start)}",
          flush=True)


async def _run_round(engine, tokenizer, cfg: TaskConfig, model_config: dict,
                     *, step: int,
                     round0_drafts_path: Path,
                     strategy_ds_path: Path,
                     out: Path,
                     max_rows_per_cell=None,
                     max_tokens=None) -> None:
    rk = f"round{step}_response"
    round_dir = out / f"round{step}"
    round_dir.mkdir(parents=True, exist_ok=True)
    drafts_ds_path = (round0_drafts_path if step == 1
                      else round_dir / f"round{step}_drafts.ds")
    filter_input_ds_path = round_dir / "filter_input.ds"
    filter1_dir = round_dir / "filter1"
    filter1_jsonl = filter1_dir / "dataset.jsonl"
    filter1_ds_path = filter1_dir / "filter1.ds"
    solve_dir = round_dir / "solve"
    solve_jsonl = solve_dir / "dataset.jsonl"

    t_round_start = time.time()
    print(f"\n--------  ROUND {step} / {cfg.max_recursive_steps}  "
          f"(variant={cfg.variant})  --------", flush=True)

    # 2a. Build round-t draft pool (round 1 reuses Stage 0; t>=2 derives from prev solve)
    print(f"[2a] round-{step} drafts", flush=True)
    if step >= 2 and not stage_state.ds_complete(drafts_ds_path):
        prev_solve_jsonl = out / f"round{step-1}" / "solve" / "dataset.jsonl"
        if not prev_solve_jsonl.is_file():
            raise FileNotFoundError(
                f"round {step} needs round {step-1} solve output, missing: "
                f"{prev_solve_jsonl}")
        flat_path = round_dir / f"round{step-1}_solve_flat.ds"
        with _timed("2a") as elapsed:
            flat = flatten.solve_jsonl_to_flat_ds(prev_solve_jsonl,
                                                  round_num=step - 1)
            _save_ds_atomic(flat, flat_path)
            drafts = aggregate.pick_one_per_problem(
                flat_path, model_alias=cfg.inference_alias,
                round_num=step - 1, seed=cfg.run_id + step)
            _save_ds_atomic(drafts, drafts_ds_path)
            print(f"  flatten {len(flat)} rows → {len(drafts)} drafts in "
                  f"{_fmt_dur(elapsed())}", flush=True)
    elif step == 1:
        existing = load_from_disk(str(drafts_ds_path))
        print(f"  reuse round-0 drafts ({len(existing)} rows)", flush=True)
    else:
        existing = load_from_disk(str(drafts_ds_path))
        print(f"  skip ({len(existing)} drafts already on disk)", flush=True)

    if cfg.variant == "rf1":
        # 2b. Join strategy
        print(f"[2b] join strategy", flush=True)
        if not stage_state.ds_complete(filter_input_ds_path):
            with _timed("2b") as elapsed:
                drafts = load_from_disk(str(drafts_ds_path))
                joined = aggregate.join_strategy(drafts, strategy_ds_path)
                _save_ds_atomic(joined, filter_input_ds_path)
                print(f"  joined {len(joined)} rows in {_fmt_dur(elapsed())}",
                      flush=True)
        else:
            existing = load_from_disk(str(filter_input_ds_path))
            print(f"  skip ({len(existing)} already joined on disk)", flush=True)

        # 2c. Filter1 inference (n=1, no verification)
        print(f"[2c] filter1 inference", flush=True)
        if not stage_state.ds_complete(filter1_ds_path):
            with _timed("2c") as elapsed:
                await run_inference_cell(
                    engine=engine, tokenizer=tokenizer,
                    model_alias=cfg.inference_alias, model_config=model_config,
                    input_path=filter_input_ds_path,
                    output_path=filter1_jsonl,
                    template_path=cfg.template_path,
                    template_key="metacognitive_filter_filter1",
                    task_name=f"round{step}_filter1_response",
                    n_samples=1,
                    variant=f"round{step}_filter1", task=cfg.task,
                    max_concurrent=cfg.max_concurrent, seed=cfg.seed,
                    per_sample_requests=False,
                    verifier=None,
                    max_rows_per_cell=max_rows_per_cell,
                    max_tokens=max_tokens,
                )
                f_ds = flatten.inference_jsonl_to_ds(
                    filter1_jsonl,
                    task_name=f"round{step}_filter1_response",
                    final_column="filtered_traj1")
                _save_ds_atomic(f_ds, filter1_ds_path)
                print(f"  filter1 ds: {len(f_ds)} rows in {_fmt_dur(elapsed())}",
                      flush=True)
        else:
            existing = load_from_disk(str(filter1_ds_path))
            print(f"  skip ({len(existing)} filtered drafts already on disk)",
                  flush=True)

        solve_input_ds_path = filter1_ds_path
    else:
        # naive: skip 2b/2c entirely; the draft pool itself is the solve input.
        # 1f templates consume the raw `traj1` column already on the draft rows.
        print(f"[2b] join strategy skipped (variant={cfg.variant})", flush=True)
        print(f"[2c] filter1 inference skipped (variant={cfg.variant})", flush=True)
        solve_input_ds_path = drafts_ds_path

    # 2d. Solve inference with streaming verification
    template_key = _solve_template_key(cfg.task, cfg.variant)
    print(f"[2d] solve inference (n={cfg.n_samples_solve}, template={template_key})",
          flush=True)
    expected_rows = len(load_from_disk(str(solve_input_ds_path)))
    if max_rows_per_cell is not None:
        expected_rows = min(expected_rows, max_rows_per_cell)
    if not stage_state.solve_raw_complete(
            solve_jsonl, task_name=rk,
            expected_rows=expected_rows, n_samples=cfg.n_samples_solve):
        verifier = Verifier(max_workers=cfg.verify_workers)
        try:
            with _timed("2d") as elapsed:
                await run_inference_cell(
                    engine=engine, tokenizer=tokenizer,
                    model_alias=cfg.inference_alias, model_config=model_config,
                    input_path=solve_input_ds_path,
                    output_path=solve_jsonl,
                    template_path=cfg.template_path,
                    template_key=template_key,
                    task_name=rk,
                    n_samples=cfg.n_samples_solve,
                    variant=f"round{step}_solve", task=cfg.task,
                    max_concurrent=cfg.max_concurrent, seed=cfg.seed,
                    per_sample_requests=True,
                    verifier=verifier,
                    max_rows_per_cell=max_rows_per_cell,
                    max_tokens=max_tokens,
                )
                print(f"  solve cell finished in {_fmt_dur(elapsed())}",
                      flush=True)
        finally:
            verifier.close()
    else:
        print(f"  skip ({expected_rows} solves × {cfg.n_samples_solve} "
              f"samples already on disk)", flush=True)

    # 2d-makeup. Targeted re-generation: keep generating extra samples for
    # any problem that still has zero passing rows (model + finish=stop +
    # parsable thinking). Idempotent — every iteration re-reads the JSONL,
    # so a crash mid-makeup is recoverable. The loop bounds itself by
    # cfg.makeup_max_attempts.
    await _run_solve_makeup(
        engine=engine, tokenizer=tokenizer, cfg=cfg, model_config=model_config,
        step=step, rk=rk, solve_dir=solve_dir, solve_jsonl=solve_jsonl,
        solve_input_ds_path=solve_input_ds_path,
        template_key=template_key,
        max_rows_per_cell=max_rows_per_cell, max_tokens=max_tokens,
    )

    # 2e. Summary writer
    print(f"[2e] summary", flush=True)
    if not stage_state.eval_artifacts_complete(solve_dir, cfg.inference_alias):
        summary.write_artifacts(solve_dir,
                                model_alias=cfg.inference_alias,
                                round_num=step)
    else:
        print(f"  skip (summary already on disk)", flush=True)

    print(f"-- round {step} done in {_fmt_dur(time.time() - t_round_start)} --",
          flush=True)


async def _run_stage_0v(cfg: TaskConfig, drafts_path: Path) -> None:
    """Verify the round-0 picked trajectories against ground truth.

    For each row in `drafts_path`, runs `Verifier._verify_one` on
    (row['traj1'], row['answer']) and updates `traj1_correctness` plus a new
    `traj1_extracted` column with our verifier's result. Idempotent — a
    sentinel `<drafts_dir>/round0_verification.json` records the run; on
    resume we just load and print the cached stats instead of re-verifying.

    The drafts dataset gets re-saved atomically so downstream stages see
    correctness annotated under our verifier (the input init dataset's
    `init_response_generations_correctness` may have come from a different
    verifier config and shouldn't be trusted to match ours).
    """
    sentinel_path = drafts_path.parent / "round0_verification.json"
    print(f"\n[stage 0v] round-0 verification", flush=True)

    if sentinel_path.is_file():
        try:
            with open(sentinel_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            _print_round0_verification_line(stats)
            print(f"  skip (cached in {sentinel_path.name})", flush=True)
            return
        except (json.JSONDecodeError, OSError):
            print(f"  [warn] sentinel {sentinel_path.name} unreadable; "
                  f"re-verifying", flush=True)

    drafts = load_from_disk(str(drafts_path))
    n = len(drafts)
    if n == 0:
        print(f"  no drafts to verify", flush=True)
        return

    pairs = [((row.get("traj1") or ""), row.get("answer")) for row in drafts]
    verifier = Verifier(max_workers=cfg.verify_workers)
    try:
        with _timed("stage 0v") as elapsed:
            results = await verifier.verify_pairs(pairs)
    finally:
        verifier.close()

    correct = incorrect = unparsable = 0
    extracted_col: list = []
    correctness_col: list = []
    for ext, ok in results:
        extracted_col.append(ext)
        correctness_col.append(ok)
        if ok is True:
            correct += 1
        elif ok is False:
            incorrect += 1
        else:
            unparsable += 1
    pass_at_1 = correct / n if n else 0.0

    # Build the updated dataset from a fresh list of dicts. We can't use
    # remove_columns + add_column on `drafts` directly because that keeps a
    # reference to the on-disk arrow files and HF refuses to overwrite a
    # dataset with one of its own backing files.
    new_rows = []
    for row, ext, ok in zip(drafts, extracted_col, correctness_col):
        new_row = {k: v for k, v in row.items()
                   if k not in ("traj1_correctness", "traj1_extracted")}
        new_row["traj1_correctness"] = ok
        new_row["traj1_extracted"] = ext
        new_rows.append(new_row)
    new_drafts = Dataset.from_list(new_rows)
    _save_ds_atomic(new_drafts, drafts_path)

    stats = {
        "n_problems": n,
        "correct": correct,
        "incorrect": incorrect,
        "unparsable": unparsable,
        "pass_at_1": pass_at_1,
    }
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sentinel_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    _print_round0_verification_line(stats)
    print(f"  verified {n} drafts in {_fmt_dur(elapsed())}", flush=True)


def _print_round0_verification_line(stats: dict) -> None:
    n = stats.get("n_problems", 0)
    correct = stats.get("correct", 0)
    incorrect = stats.get("incorrect", 0)
    unparsable = stats.get("unparsable", 0)
    pass_at_1 = stats.get("pass_at_1", 0.0)
    print(f"  problems: {n} cohort  "
          f"correct={correct}  incorrect={incorrect}  "
          f"unparsable={unparsable}  "
          f"pass@1={pass_at_1:.3f}", flush=True)


def _print_solve_stats(flat, *, model_alias: str, round_num: int,
                       expected_ids: set, label: str) -> None:
    """Two-line digest of the solve flat dataset: per-problem coverage and
    per-generation finish/correctness breakdown. Computed against the same
    filters the aggregator uses, so the 'need_makeup' count matches what the
    makeup loop will act on. Cheap — single pass over the flat rows."""
    rk = f"round{round_num}_response" if round_num >= 1 else "init_response"
    meta_col = f"{rk}_generations_metadata"
    finish_col = f"{rk}_generations_finish_reason"
    thinking_col = f"{rk}_thinking_status"
    correctness_col = f"{rk}_generations_correctness"

    finish_counts: dict = {}
    correctness_counts = {"correct": 0, "incorrect": 0, "unparsable": 0}
    satisfied: set = set()

    if (len(flat) > 0
            and meta_col in flat.column_names
            and finish_col in flat.column_names
            and thinking_col in flat.column_names):
        ids = flat["id"]
        metas = flat[meta_col]
        finishes = flat[finish_col]
        thinkings = flat[thinking_col]
        corrects = (flat[correctness_col]
                    if correctness_col in flat.column_names
                    else [None] * len(flat))
        for i in range(len(flat)):
            fr = finishes[i] or "unknown"
            finish_counts[fr] = finish_counts.get(fr, 0) + 1
            c = corrects[i]
            if c is True:
                correctness_counts["correct"] += 1
            elif c is False:
                correctness_counts["incorrect"] += 1
            else:
                correctness_counts["unparsable"] += 1
            m = metas[i] or {}
            if (m.get("model_config_alias") == model_alias
                    and finishes[i] == "stop"
                    and thinkings[i] == "parsable_thinking"):
                satisfied.add(ids[i])

    n_cohort = len(expected_ids)
    n_covered = len(satisfied & expected_ids)
    n_need = n_cohort - n_covered
    total_gens = len(flat)

    fr_str = " ".join(f"{k}={v}" for k, v in sorted(finish_counts.items())) \
             or "(none)"
    cc = correctness_counts
    print(f"[{label}] problems: {n_cohort} cohort  "
          f"{n_covered} covered (≥1 parsable+stop)  "
          f"{n_need} need makeup", flush=True)
    print(f"[{label}] samples:  {total_gens} total  "
          f"finish={{{fr_str}}}  "
          f"correct={cc['correct']}  incorrect={cc['incorrect']}  "
          f"unparsable={cc['unparsable']}", flush=True)


async def _run_solve_makeup(*, engine, tokenizer, cfg: TaskConfig,
                            model_config: dict, step: int, rk: str,
                            solve_dir: Path, solve_jsonl: Path,
                            solve_input_ds_path: Path,
                            template_key: str,
                            max_rows_per_cell, max_tokens) -> None:
    """Iteratively top up the solve JSONL until every problem in the solve
    input has at least one row passing (alias + stop + parsable). Bounded by
    cfg.makeup_max_attempts. Each attempt uses a fresh seed offset so we
    don't redraw the same sample."""
    solve_input_ds = load_from_disk(str(solve_input_ds_path))
    if max_rows_per_cell is not None and len(solve_input_ds) > max_rows_per_cell:
        solve_input_ds = solve_input_ds.select(range(max_rows_per_cell))
    expected_ids = set(solve_input_ds["id"])
    id_to_row: dict = {row["id"]: dict(row) for row in solve_input_ds}

    # Always print stats once after the main solve, even if makeup is disabled.
    initial_flat = flatten.solve_jsonl_to_flat_ds(solve_jsonl, round_num=step)
    _print_solve_stats(initial_flat,
                       model_alias=cfg.inference_alias,
                       round_num=step, expected_ids=expected_ids,
                       label="2d stats")

    if cfg.makeup_max_attempts <= 0:
        return

    print(f"[2d-makeup] checking cohort coverage", flush=True)
    for attempt in range(cfg.makeup_max_attempts + 1):
        flat = (initial_flat if attempt == 0
                else flatten.solve_jsonl_to_flat_ds(solve_jsonl, round_num=step))
        unsatisfied = aggregate.find_unsatisfied_ids(
            flat, model_alias=cfg.inference_alias,
            round_num=step, expected_ids=expected_ids)
        if not unsatisfied:
            if attempt == 0:
                print(f"  all {len(expected_ids)} problems already covered",
                      flush=True)
            else:
                print(f"  satisfied after {attempt} makeup batch(es)",
                      flush=True)
            return
        if attempt == cfg.makeup_max_attempts:
            print(f"  [warn] giving up: {len(unsatisfied)}/{len(expected_ids)} "
                  f"problems still have 0 parsable+stop samples after "
                  f"{cfg.makeup_max_attempts} makeup attempts. first ids: "
                  f"{sorted(unsatisfied)[:10]}. these will drop out of the "
                  f"round-{step+1} cohort.", flush=True)
            return

        rows = [id_to_row[i] for i in sorted(unsatisfied) if i in id_to_row]
        n_batch = cfg.makeup_batch_size
        # Each attempt uses a distinct seed so the sub-request seeds
        # (base_seed+1..base_seed+N inside _drive_generation) don't collide
        # with the main solve pass or with previous makeup attempts.
        makeup_seed = cfg.seed + step * 100003 + (attempt + 1) * 1009
        print(f"  attempt {attempt+1}/{cfg.makeup_max_attempts}: "
              f"{len(rows)} unsatisfied ids × n={n_batch} (seed={makeup_seed})",
              flush=True)
        verifier = Verifier(max_workers=cfg.verify_workers)
        try:
            with _timed("2d-makeup") as elapsed:
                await run_makeup_cell(
                    engine=engine, tokenizer=tokenizer,
                    model_alias=cfg.inference_alias, model_config=model_config,
                    rows=rows,
                    output_path=solve_jsonl,
                    template_path=cfg.template_path,
                    template_key=template_key,
                    task_name=rk,
                    n_samples=n_batch,
                    variant=f"round{step}_solve_makeup", task=cfg.task,
                    max_concurrent=cfg.max_concurrent,
                    seed=makeup_seed,
                    per_sample_requests=True,
                    verifier=verifier,
                    max_tokens=max_tokens,
                )
                print(f"  attempt {attempt+1} done in {_fmt_dur(elapsed())}",
                      flush=True)
        finally:
            verifier.close()
