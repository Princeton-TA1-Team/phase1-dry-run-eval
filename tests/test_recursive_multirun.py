"""Unit tests for cards/nodes/_recursive_multirun.py (GPU-free).

Covers GPU-pool partitioning and the per-step averaging across N runs
(mean = average over n_runs x n_samples_solve generations; std across runs).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cards.nodes._recursive_multirun import aggregate, gpu_pools, run_trace  # noqa: E402


def test_gpu_pools():
    assert gpu_pools(["0", "1", "2", "3"], 4) == ["0,1,2,3"]      # tp=4 -> sequential
    assert gpu_pools(["0", "1", "2", "3"], 1) == ["0", "1", "2", "3"]  # tp=1 -> 4 concurrent
    assert gpu_pools(["0", "1", "2", "3"], 2) == ["0,1", "2,3"]   # tp=2 -> 2 concurrent
    assert gpu_pools(["0"], 4) == ["0"]                           # fewer gpus than tp


def _seed_run(root: Path, rid: int, vals: dict[int, float], n0: int = 30):
    rd = root / f"run_{rid}" / "recursive"
    rd.mkdir(parents=True)
    (rd / "round0_verification.json").write_text(
        json.dumps({"n_problems": n0, "pass_at_1": vals[0]}))
    for step, v in vals.items():
        if step == 0:
            continue
        sd = rd / f"round{step}" / "solve"
        sd.mkdir(parents=True)
        (sd / "evaluated_GPT_OSS_20B_error_analysis.json").write_text(
            json.dumps({"overall_stats": {"pass_at_1": v}}))


def test_aggregate_mean_over_runs(tmp_path):
    _seed_run(tmp_path, 1, {0: 0.50, 1: 0.60, 2: 0.70})
    _seed_run(tmp_path, 2, {0: 0.40, 1: 0.50, 2: 0.80})
    _seed_run(tmp_path, 3, {0: 0.60, 1: 0.70, 2: 0.60})
    agg = aggregate(runs_root=tmp_path, n_runs=3, max_steps=2, init_alias="GPT_OSS_20B")
    assert abs(agg["acc_round_0"] - 0.50) < 1e-9       # mean(0.5,0.4,0.6)
    assert abs(agg["acc_round_max"] - 0.70) < 1e-9     # step2 mean(0.7,0.8,0.6)
    assert abs(agg["delta"] - 0.20) < 1e-9
    assert agg["n_runs_completed"] == 3
    assert agg["max_step_reached"] == 2
    assert not agg["aggregate_failed"] and not agg["makeup_exhausted"]
    assert agg["per_step"][2]["n_runs"] == 3
    assert agg["per_step"][2]["std"] > 0


def test_aggregate_uses_deepest_common_step(tmp_path):
    # run 2 stalls at step 1; final must use the deepest step ALL runs reached (1).
    _seed_run(tmp_path, 1, {0: 0.5, 1: 0.6, 2: 0.7})
    _seed_run(tmp_path, 2, {0: 0.5, 1: 0.4})
    agg = aggregate(runs_root=tmp_path, n_runs=2, max_steps=2, init_alias="GPT_OSS_20B")
    assert agg["max_step_reached"] == 1
    assert abs(agg["acc_round_max"] - 0.50) < 1e-9     # step1 mean(0.6,0.4)


def test_aggregate_no_runs_is_failed(tmp_path):
    agg = aggregate(runs_root=tmp_path, n_runs=4, max_steps=2, init_alias="X")
    assert agg["aggregate_failed"] and agg["delta"] is None and agg["n_runs_completed"] == 0


def test_run_trace_reads_round0_and_steps(tmp_path):
    _seed_run(tmp_path, 7, {0: 0.55, 1: 0.66})
    rt = run_trace(tmp_path / "run_7" / "recursive", "GPT_OSS_20B", max_steps=2)
    assert rt["n_problems_round_0"] == 30
    assert abs(rt["trace"][0] - 0.55) < 1e-9 and abs(rt["trace"][1] - 0.66) < 1e-9
    assert 2 not in rt["trace"]
