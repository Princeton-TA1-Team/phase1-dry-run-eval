"""
Magnet pipeline node: recursive self-improvement (rf1 variant).

Subprocess chain (one long-running process on the host's GPU):

    python -m contextual_drag recursive run
        --variant rf1 --model_config <X> --init_alias <Y>
        --input_ds <data/full_data/<task>/<task>.ds>
        --output_dir <out>/recursive --max_recursive_steps 16
        --n_samples_solve 8 --task_name <task>
      -> (the recursive pipeline writes
          <out>/recursive/round0_verification.json with pass@1 for round 0,
          and <out>/recursive/round{step}/solve/dataset.jsonl +
          evaluated_<alias>_error_analysis.json for each step >= 1)
      -> python -m contextual_drag eval math --dataset_dir
             <out>/recursive/round{max_step}/solve
         (rerun as a safety net in case the pipeline crashed before
         summary.write_artifacts ran)
      -> read acc_round_0 from round0_verification.json::pass_at_1
      -> read acc_round_max from the highest-step
         evaluated_*_error_analysis.json::overall_stats.pass_at_1
      -> delta_acc_rf1 = acc_round_max - acc_round_0

Writes results.json:

    {"result": {
        "acc_round_0": 0.42,
        "acc_round_max": 0.55,
        "delta_acc_rf1": 0.13,
        "aggregate_failed": false,
        "makeup_exhausted": false,
        "max_step_reached": 16,
        "n_problems_round_0": 30,
        "variant": "rf1",
        "model_config": "GPT_OSS_20B_recursive",
        "init_alias": "GPT_OSS_20B",
        "task": "aime24",
        "data_path": "data/full_data/aime24/aime24.ds",
        ...
    }}

Degenerate branches (always writes a valid result.json — never leaves
the file missing; uses -1.0 sentinels in place of JSON null since
magnet's Symbol.eval() cannot resolve null):

  - Stage-0 aggregator produced 0 problems -> aggregate_failed=True,
    acc_round_0 = acc_round_max = delta_acc_rf1 = -1.0.
  - Final round never reached / makeup exhausted -> makeup_exhausted=True,
    acc_round_max = delta_acc_rf1 = -1.0 (acc_round_0 may still be valid).
  - Any exception during the subprocess chain -> the wrapper catches,
    sets aggregate_failed=True (best-effort), writes sentinels, re-raises
    only after the result.json has been flushed to disk.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import traceback
from pathlib import Path

import scriptconfig as scfg


VARIANT = "rf1"


class RunRecursiveFilter1CLI(scfg.DataConfig):
    """Sweep-able pipeline node for the recursive (rf1) self-improvement card."""

    model_config = scfg.Value("GPT_OSS_20B_recursive", tags=["algo_param"])
    init_alias = scfg.Value("GPT_OSS_20B", tags=["algo_param"])
    task = scfg.Value("aime24", tags=["algo_param"],
                       choices=["aime24", "aime25", "hmmt24", "hmmt25"])
    data_path = scfg.Value("data/full_data/aime24/aime24.ds", tags=["algo_param"])
    template_path = scfg.Value(
        "prompt_templates/recursive_templates.json", tags=["algo_param"])
    init_template_path = scfg.Value(
        "prompt_templates/init_response_prompt_templates.json", tags=["algo_param"])
    init_n_samples = scfg.Value(8, type=int, tags=["algo_param"])
    max_recursive_steps = scfg.Value(16, type=int, tags=["algo_param"])
    n_samples_solve = scfg.Value(8, type=int, tags=["algo_param"])
    tensor_parallel_size = scfg.Value(1, type=int, tags=["algo_param"])
    gpu_memory_utilization = scfg.Value(0.9, type=float, tags=["algo_param"])
    max_tokens = scfg.Value(65536, type=int, tags=["algo_param"])

    results_fpath = scfg.Value("results.json", tags=["out_path", "primary"])

    @classmethod
    def main(cls, argv=None, **kwargs):
        cfg = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)

        results_fpath = Path(cfg.results_fpath).resolve()
        results_fpath.parent.mkdir(parents=True, exist_ok=True)
        work_dir = results_fpath.parent
        recursive_dir = work_dir / "recursive"
        recursive_dir.mkdir(parents=True, exist_ok=True)

        cdrag = [sys.executable, "-m", "contextual_drag"]

        aggregate_failed = False
        makeup_exhausted = False
        init_sampling_skipped = False
        recursive_input_ds: Path = Path(cfg.data_path)
        acc_round_0: float | None = None
        acc_round_max: float | None = None
        max_step_reached: int = 0
        n_problems_round_0: int = 0

        try:
            print(f"[card-node recursive/{VARIANT}] step 0/3: ensure init responses "
                  f"(input={cfg.data_path}, init_alias={cfg.init_alias}, "
                  f"init_n_samples={cfg.init_n_samples})", flush=True)
            from cards.nodes._recursive_init_sampling import ensure_init_responses
            recursive_input_ds, init_sampling_skipped = ensure_init_responses(
                data_path=Path(cfg.data_path),
                intermediate_dir=work_dir,
                init_alias=str(cfg.init_alias),
                task=str(cfg.task),
                init_template_path=Path(cfg.init_template_path),
                init_n_samples=int(cfg.init_n_samples),
                tensor_parallel_size=int(cfg.tensor_parallel_size),
                gpu_memory_utilization=float(cfg.gpu_memory_utilization),
            )

            print(f"[card-node recursive/{VARIANT}] step 1/3: recursive run "
                  f"(--variant {VARIANT}, max_recursive_steps="
                  f"{cfg.max_recursive_steps}, n_samples_solve={cfg.n_samples_solve})",
                  flush=True)
            subprocess.run(cdrag + [
                "recursive", "run",
                "--variant", VARIANT,
                "--model_config", str(cfg.model_config),
                "--init_alias", str(cfg.init_alias),
                "--input_ds", str(recursive_input_ds),
                "--output_dir", str(recursive_dir),
                "--template_path", str(cfg.template_path),
                "--max_recursive_steps", str(cfg.max_recursive_steps),
                "--n_samples_solve", str(cfg.n_samples_solve),
                "--tensor_parallel_size", str(cfg.tensor_parallel_size),
                "--gpu_memory_utilization", str(cfg.gpu_memory_utilization),
                "--max_tokens", str(cfg.max_tokens),
                "--task_name", str(cfg.task),
            ], check=True)

            # ---- round 0: read pass@1 from the sentinel the pipeline writes ----
            round0_path = _find_round0_verification(recursive_dir)
            if round0_path is None:
                aggregate_failed = True
                print(f"[card-node recursive/{VARIANT}] no round0_verification.json "
                      f"under {recursive_dir} — treating as aggregate_failed.",
                      flush=True)
            else:
                with open(round0_path) as f:
                    r0 = json.load(f)
                n_problems_round_0 = int(r0.get("n_problems") or 0)
                if n_problems_round_0 == 0:
                    aggregate_failed = True
                    acc_round_0 = 0.0
                else:
                    acc_round_0 = float(r0.get("pass_at_1") or 0.0)

            # ---- round N: locate highest-step solve dir and resolve pass@1 ----
            max_step_reached, round_max_solve_dir = _find_max_solve_dir(recursive_dir)
            if round_max_solve_dir is None:
                makeup_exhausted = True
                print(f"[card-node recursive/{VARIANT}] no round*/solve/ dir "
                      f"under {recursive_dir} — treating as makeup_exhausted.",
                      flush=True)
            else:
                print(f"[card-node recursive/{VARIANT}] step 2/3: eval math on "
                      f"{round_max_solve_dir} (safety net)", flush=True)
                # Best-effort eval; the pipeline's summary.write_artifacts may
                # already have produced the error_analysis.json. If eval fails
                # we still try to read whatever's on disk.
                subprocess.run(cdrag + [
                    "eval", "math",
                    "--dataset_dir", str(round_max_solve_dir),
                    "--single_partition", "--n_jobs", "1",
                ], check=False)
                acc_round_max = _read_pass_at_1(round_max_solve_dir, cfg.init_alias)
                if acc_round_max is None:
                    makeup_exhausted = True

            if (acc_round_0 is not None
                    and acc_round_max is not None
                    and not aggregate_failed
                    and not makeup_exhausted):
                delta = float(acc_round_max) - float(acc_round_0)
            else:
                delta = None

            print(f"[card-node recursive/{VARIANT}] step 3/3: writing result.json",
                  flush=True)
            _write_result(
                results_fpath, cfg,
                acc_round_0=acc_round_0,
                acc_round_max=acc_round_max,
                delta=delta,
                aggregate_failed=aggregate_failed,
                makeup_exhausted=makeup_exhausted,
                max_step_reached=max_step_reached,
                n_problems_round_0=n_problems_round_0,
                init_sampling_skipped=init_sampling_skipped,
            )
        except Exception:
            traceback.print_exc()
            _write_result(
                results_fpath, cfg,
                acc_round_0=acc_round_0,
                acc_round_max=acc_round_max,
                delta=None,
                aggregate_failed=True if acc_round_0 is None else aggregate_failed,
                makeup_exhausted=True if acc_round_max is None else makeup_exhausted,
                max_step_reached=max_step_reached,
                n_problems_round_0=n_problems_round_0,
                init_sampling_skipped=init_sampling_skipped,
            )
            raise


def _write_result(results_fpath: Path, cfg, *, acc_round_0, acc_round_max,
                  delta, aggregate_failed: bool, makeup_exhausted: bool,
                  max_step_reached: int, n_problems_round_0: int,
                  init_sampling_skipped: bool) -> None:
    # magnet's symbol resolver chokes on JSON null values; use -1.0 sentinel.
    sentinel = -1.0

    def _f(v):
        return sentinel if v is None else float(v)

    payload = {
        "result": {
            "acc_round_0":          _f(acc_round_0),
            "acc_round_max":        _f(acc_round_max),
            "delta_acc_rf1":        _f(delta),
            "aggregate_failed":     bool(aggregate_failed),
            "makeup_exhausted":     bool(makeup_exhausted),
            "init_sampling_skipped": bool(init_sampling_skipped),
            "max_step_reached":     int(max_step_reached),
            "n_problems_round_0":   int(n_problems_round_0),
            "variant":              VARIANT,
            "model_config":         str(cfg.model_config),
            "init_alias":           str(cfg.init_alias),
            "task":                 str(cfg.task),
            "data_path":            str(cfg.data_path),
            "template_path":        str(cfg.template_path),
            "init_template_path":   str(cfg.init_template_path),
            "init_n_samples":       int(cfg.init_n_samples),
            "max_recursive_steps":  int(cfg.max_recursive_steps),
            "n_samples_solve":      int(cfg.n_samples_solve),
        }
    }
    with open(results_fpath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[card-node recursive/{VARIANT}] wrote {results_fpath}: "
          f"{payload['result']}", flush=True)


def _find_round0_verification(recursive_dir: Path) -> Path | None:
    """Locate the round-0 verification sentinel written by stage 0v."""
    direct = recursive_dir / "round0_verification.json"
    if direct.is_file():
        return direct
    cands = sorted(recursive_dir.rglob("round0_verification.json"))
    return cands[0] if cands else None


_ROUND_DIR_RE = re.compile(r"^round(\d+)$")


def _find_max_solve_dir(recursive_dir: Path) -> tuple[int, Path | None]:
    """Return (max_step, solve_dir) for the highest round{step}/solve/ present.

    Falls back to (0, None) when no round{step >= 1} solve directory exists
    on disk — the loop must have bailed before producing any solve output.
    """
    max_step = 0
    best: Path | None = None
    if not recursive_dir.is_dir():
        return (0, None)
    for child in recursive_dir.iterdir():
        if not child.is_dir():
            continue
        m = _ROUND_DIR_RE.match(child.name)
        if not m:
            continue
        step = int(m.group(1))
        if step < 1:
            continue
        solve_dir = child / "solve"
        if not solve_dir.is_dir():
            continue
        if step > max_step:
            max_step = step
            best = solve_dir
    return (max_step, best)


def _read_pass_at_1(solve_dir: Path, init_alias: str) -> float | None:
    """Pull pass@1 out of evaluated_<alias>_error_analysis.json under *solve_dir*.

    Falls back to any matching evaluated_*_error_analysis.json if the
    alias-specific one is absent (the pipeline's summary.write_artifacts
    writes against `inference_alias` but a re-run of `eval math` keys on
    the dataset's own naming).
    """
    candidates = []
    preferred = solve_dir / f"evaluated_{init_alias}_error_analysis.json"
    if preferred.is_file():
        candidates.append(preferred)
    candidates += sorted(p for p in solve_dir.glob(
        "evaluated_*_error_analysis.json") if p not in candidates)
    for path in candidates:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        overall = data.get("overall_stats") or {}
        if "pass_at_1" in overall:
            return float(overall["pass_at_1"])
        if "pass_at_1" in data:
            return float(data["pass_at_1"])
        # contextual_drag eval math writes a different shape; try that.
        pak = ((data.get("pass_at_k_by_source") or {}).get("overall")
               or {}).get("overall_correctness")
        if pak is not None:
            return float(pak)
    return None


def main(argv=None):
    return RunRecursiveFilter1CLI.main(argv=argv)


if __name__ == "__main__":
    main()
