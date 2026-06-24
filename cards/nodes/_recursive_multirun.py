"""Launch N independent recursive trajectories and aggregate per-step pass@1.

Faithfully mirrors the old-repo `recursive_filter1/deploy_sweep.sh` procedure:
N_RUNS independent runs (run_id 1..N, each seeding a different round-0 trajectory
pick + per-round continuation pick), each sampling `n_samples_solve` rollouts per
step and picking one at random to continue. Per-step accuracy is the average over
(N runs x n_samples_solve rollouts) — e.g. 16 x 16 = 256 generations/step — with a
std taken across the N independent trajectories (the shaded band in the paper's Fig. 3).

Each run is a standalone `contextual_drag recursive run --run_id i` writing to
`<runs_root>/run_<i>/recursive/`. Runs launch in parallel up to
`num_gpus // tensor_parallel_size` at a time, each pinned to its own GPU slice via
CUDA_VISIBLE_DEVICES. Re-invoking resumes (the recursive pipeline is per-stage
resumable), so a walltime-limited job can be resubmitted to completion.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from statistics import fmean, pstdev


# --------------------------------------------------------------------------- GPU pool
def detect_gpu_ids() -> list[str]:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        return [x for x in cvd.split(",") if x != ""]
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True)
        n = len([ln for ln in out.stdout.splitlines() if ln.startswith("GPU ")])
        return [str(i) for i in range(n)] if n else ["0"]
    except Exception:
        return ["0"]


def gpu_pools(gpu_ids: list[str], tp: int) -> list[str]:
    """Partition gpu_ids into CUDA_VISIBLE_DEVICES strings of `tp` GPUs each.
    With tp == len(gpu_ids) there is one pool (runs go sequentially)."""
    tp = max(1, int(tp))
    pools = [",".join(gpu_ids[i:i + tp]) for i in range(0, len(gpu_ids) - tp + 1, tp)]
    return pools or [",".join(gpu_ids)]


# --------------------------------------------------------------------------- launching
def launch_runs(*, cdrag: list[str], variant: str, recursive_input_ds: Path,
                runs_root: Path, base_args: list[str], n_runs: int, seed: int,
                tp: int, gpu_ids: list[str]) -> dict[int, int]:
    """Run `recursive run` for run_id 1..n_runs, parallel across GPU pools.
    Returns {run_id: returncode}."""
    pools = gpu_pools(gpu_ids, tp)
    print(f"[multirun/{variant}] {n_runs} runs over {len(pools)} GPU pool(s) "
          f"(tp={tp}, pools={pools})", flush=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    pending = list(range(1, n_runs + 1))
    active: dict[str, tuple[int, subprocess.Popen]] = {}
    rc: dict[int, int] = {}

    def _spawn(pool: str, rid: int) -> None:
        out_dir = runs_root / f"run_{rid}" / "recursive"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = cdrag + ["recursive", "run",
                       "--variant", variant,
                       "--input_ds", str(recursive_input_ds),
                       "--output_dir", str(out_dir),
                       "--run_id", str(rid),
                       "--seed", str(seed),
                       "--tensor_parallel_size", str(tp)] + base_args
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = pool
        log = (runs_root / f"run_{rid}" / "run.log").open("w")
        print(f"[multirun/{variant}] start run_id={rid} on GPUs[{pool}] -> {out_dir}", flush=True)
        active[pool] = (rid, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))

    for pool in pools:
        if pending:
            _spawn(pool, pending.pop(0))
    while active:
        time.sleep(10)
        for pool, (rid, proc) in list(active.items()):
            ret = proc.poll()
            if ret is None:
                continue
            rc[rid] = ret
            print(f"[multirun/{variant}] run_id={rid} done (exit={ret})", flush=True)
            del active[pool]
            if pending:
                _spawn(pool, pending.pop(0))
    return rc


# --------------------------------------------------------------------------- reading one run
def _read_pass_at_1(solve_dir: Path, init_alias: str) -> float | None:
    cands = []
    pref = solve_dir / f"evaluated_{init_alias}_error_analysis.json"
    if pref.is_file():
        cands.append(pref)
    cands += sorted(p for p in solve_dir.glob("evaluated_*_error_analysis.json") if p not in cands)
    for path in cands:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        ov = data.get("overall_stats") or {}
        if "pass_at_1" in ov:
            return float(ov["pass_at_1"])
        if "pass_at_1" in data:
            return float(data["pass_at_1"])
        pak = ((data.get("pass_at_k_by_source") or {}).get("overall") or {}).get("overall_correctness")
        if pak is not None:
            return float(pak)
    return None


def run_trace(run_recursive_dir: Path, init_alias: str, max_steps: int) -> dict:
    """Per-step pass@1 for one run + its round-0 problem count."""
    trace: dict[int, float] = {}
    n0 = 0
    r0f = run_recursive_dir / "round0_verification.json"
    if not r0f.is_file():
        hits = sorted(run_recursive_dir.rglob("round0_verification.json"))
        r0f = hits[0] if hits else r0f
    if r0f.is_file():
        r0 = json.loads(r0f.read_text())
        n0 = int(r0.get("n_problems") or 0)
        if n0 > 0:
            trace[0] = float(r0.get("pass_at_1") or 0.0)
    for step in range(1, max_steps + 1):
        sd = run_recursive_dir / f"round{step}" / "solve"
        if sd.is_dir():
            a = _read_pass_at_1(sd, init_alias)
            if a is not None:
                trace[step] = a
    return {"trace": trace, "n_problems_round_0": n0}


# --------------------------------------------------------------------------- aggregation
def aggregate(*, runs_root: Path, n_runs: int, max_steps: int, init_alias: str) -> dict:
    """Average per-step pass@1 across the N runs (mean +/- std over runs)."""
    runs, n0s = [], []
    for rid in range(1, n_runs + 1):
        rd = runs_root / f"run_{rid}" / "recursive"
        if not rd.is_dir():
            continue
        rt = run_trace(rd, init_alias, max_steps)
        if rt["trace"]:
            runs.append(rt["trace"])
            n0s.append(rt["n_problems_round_0"])

    if not runs:
        return {"per_step": {}, "acc_round_0": None, "acc_round_max": None,
                "delta": None, "n_runs_completed": 0, "max_step_reached": 0,
                "n_problems_round_0": 0, "aggregate_failed": True, "makeup_exhausted": False}

    steps = sorted({s for t in runs for s in t})
    per_step = {}
    for s in steps:
        vals = [t[s] for t in runs if s in t]
        per_step[s] = {"mean": fmean(vals), "std": (pstdev(vals) if len(vals) > 1 else 0.0),
                       "n_runs": len(vals), "values": [round(v, 4) for v in vals]}
    have0 = 0 in per_step
    acc_round_0 = per_step[0]["mean"] if have0 else None
    common = [s for s in steps if s >= 1 and per_step[s]["n_runs"] == len(runs)]
    max_common = max(common) if common else 0
    acc_round_max = per_step[max_common]["mean"] if max_common >= 1 else None
    delta = (acc_round_max - acc_round_0) if (acc_round_0 is not None and acc_round_max is not None) else None
    return {"per_step": per_step, "acc_round_0": acc_round_0, "acc_round_max": acc_round_max,
            "delta": delta, "n_runs_completed": len(runs), "max_step_reached": max_common,
            "n_problems_round_0": (max(n0s) if n0s else 0),
            "aggregate_failed": not have0, "makeup_exhausted": (acc_round_max is None)}


# --------------------------------------------------------------------------- node body (shared rf1 / naive)
def _write_result(results_fpath: Path, cfg, *, variant: str, delta_key: str, agg: dict,
                  init_sampling_skipped: bool, n_runs: int) -> None:
    sentinel = -1.0

    def _f(v):
        return sentinel if v is None else float(v)

    # compact per-step trace (mean/std) keyed by str(step) for JSON
    trace = {str(s): {"mean": round(d["mean"], 4), "std": round(d["std"], 4),
                      "n_runs": d["n_runs"]} for s, d in (agg.get("per_step") or {}).items()}
    payload = {"result": {
        "acc_round_0":           _f(agg.get("acc_round_0")),
        "acc_round_max":         _f(agg.get("acc_round_max")),
        delta_key:               _f(agg.get("delta")),
        "aggregate_failed":      bool(agg.get("aggregate_failed")),
        "makeup_exhausted":      bool(agg.get("makeup_exhausted")),
        "init_sampling_skipped": bool(init_sampling_skipped),
        "max_step_reached":      int(agg.get("max_step_reached") or 0),
        "n_problems_round_0":    int(agg.get("n_problems_round_0") or 0),
        "n_runs":                int(n_runs),
        "n_runs_completed":      int(agg.get("n_runs_completed") or 0),
        "per_step_pass_at_1":    trace,
        "variant":               variant,
        "model_config":          str(cfg.model_config),
        "init_alias":            str(cfg.init_alias),
        "task":                  str(cfg.task),
        "data_path":             str(cfg.data_path),
        "template_path":         str(cfg.template_path),
        "init_template_path":    str(cfg.init_template_path),
        "init_n_samples":        int(cfg.init_n_samples),
        "max_recursive_steps":   int(cfg.max_recursive_steps),
        "n_samples_solve":       int(cfg.n_samples_solve),
    }}
    results_fpath.write_text(json.dumps(payload, indent=2))
    print(f"[multirun/{variant}] wrote {results_fpath}: "
          f"acc0={payload['result']['acc_round_0']} accmax={payload['result']['acc_round_max']} "
          f"{delta_key}={payload['result'][delta_key]} "
          f"n_runs={agg.get('n_runs_completed')}/{n_runs}", flush=True)


def run_node(cfg, *, variant: str, delta_key: str) -> None:
    """Shared body for the rf1 / naive recursive nodes: build init once, launch
    cfg.n_runs independent trajectories, aggregate per-step pass@1, write result."""
    results_fpath = Path(cfg.results_fpath).resolve()
    results_fpath.parent.mkdir(parents=True, exist_ok=True)
    work_dir = results_fpath.parent
    runs_root = work_dir / "runs"
    cdrag = [sys.executable, "-m", "contextual_drag"]
    n_runs = int(cfg.n_runs)
    init_sampling_skipped = False
    agg = {"aggregate_failed": True, "makeup_exhausted": False, "acc_round_0": None,
           "acc_round_max": None, "delta": None, "max_step_reached": 0,
           "n_problems_round_0": 0, "n_runs_completed": 0, "per_step": {}}
    try:
        print(f"[multirun/{variant}] step 0: ensure init responses "
              f"(init_alias={cfg.init_alias}, init_n_samples={cfg.init_n_samples})", flush=True)
        from cards.nodes._recursive_init_sampling import ensure_init_responses
        recursive_input_ds, init_sampling_skipped = ensure_init_responses(
            data_path=Path(cfg.data_path), intermediate_dir=work_dir,
            init_alias=str(cfg.init_alias), task=str(cfg.task),
            init_template_path=Path(cfg.init_template_path),
            init_n_samples=int(cfg.init_n_samples),
            tensor_parallel_size=int(cfg.tensor_parallel_size),
            gpu_memory_utilization=float(cfg.gpu_memory_utilization))

        base_args = [
            "--model_config", str(cfg.model_config),
            "--init_alias", str(cfg.init_alias),
            "--template_path", str(cfg.template_path),
            "--max_recursive_steps", str(cfg.max_recursive_steps),
            "--n_samples_solve", str(cfg.n_samples_solve),
            "--gpu_memory_utilization", str(cfg.gpu_memory_utilization),
            "--max_tokens", str(cfg.max_tokens),
            "--task_name", str(cfg.task)]
        gpu_ids = detect_gpu_ids()
        print(f"[multirun/{variant}] step 1: launch {n_runs} runs "
              f"(n_samples_solve={cfg.n_samples_solve}, steps={cfg.max_recursive_steps}, "
              f"gpus={gpu_ids}, tp={cfg.tensor_parallel_size})", flush=True)
        rc = launch_runs(cdrag=cdrag, variant=variant, recursive_input_ds=recursive_input_ds,
                         runs_root=runs_root, base_args=base_args, n_runs=n_runs,
                         seed=int(cfg.seed), tp=int(cfg.tensor_parallel_size), gpu_ids=gpu_ids)
        nonzero = [r for r, c in rc.items() if c != 0]
        if nonzero:
            print(f"[multirun/{variant}] WARN: runs with nonzero exit: {nonzero}", flush=True)

        print(f"[multirun/{variant}] step 2: aggregate per-step pass@1 across runs", flush=True)
        agg = aggregate(runs_root=runs_root, n_runs=n_runs,
                        max_steps=int(cfg.max_recursive_steps), init_alias=str(cfg.init_alias))
        _write_result(results_fpath, cfg, variant=variant, delta_key=delta_key,
                      agg=agg, init_sampling_skipped=init_sampling_skipped, n_runs=n_runs)
    except Exception:
        traceback.print_exc()
        _write_result(results_fpath, cfg, variant=variant, delta_key=delta_key,
                      agg=agg, init_sampling_skipped=init_sampling_skipped, n_runs=n_runs)
        raise
