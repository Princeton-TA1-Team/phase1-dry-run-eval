"""Visualize GPT-OSS-20B contextual-drag results, averaged across datasets per task.

Error bars are 95% confidence intervals that account for sample size, following the
old repo's `final_draft/tables/lib.py::bernoulli_ci_pp`:
  - Each dataset's pass@1 p_i is a Bernoulli proportion over n_i = n_problems × n_samples
    trials, with variance p_i(1-p_i)/n_i (normal approximation).
  - A bar's height is the unweighted mean over datasets; its 95% CI half-width is
    1.96 * sqrt( (1/k^2) * sum_i p_i(1-p_i)/n_i )  (CI of that mean), in pp.
Recursive band = 95% CI of the per-step mean across pooled trajectories (1.96 * SEM).

Panels: Drag(1F,raw) | Drag(2F) | Error-cond post-hoc(1F,verdict-filtered) |
Error-cond external(1F,framing) | Mitigation(dragged vs mitigated) ; + recursive trend.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from statistics import fmean, pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATASETS = ["aime24", "aime25", "hmmt24", "hmmt25", "gpqa", "mmlu", "crux-i", "24-game"]
EC_N_SAMPLES = 16   # formal EC cards use n=16 (not stored in results.json)
REC_N_SOLVE = 16    # recursive cards' n_samples_solve (rollouts/step/problem)
Z = 1.96
SENT = -1.0


def _ok(x) -> bool:
    return isinstance(x, (int, float)) and x != SENT and x >= 0


def _result(eval_root: Path, test: str, ds: str) -> dict | None:
    hits = list((eval_root / f"GPT_OSS_20B__{test}__{ds}").rglob("results.json"))
    if not hits:
        return None
    try:
        return json.loads(hits[0].read_text()).get("result", {})
    except Exception:
        return None


def _ec_summary(eval_root: Path, ds: str) -> dict | None:
    hits = list((eval_root / f"GPT_OSS_20B__error-conditioning-posthoc__{ds}").rglob("ec_summary.json"))
    try:
        return json.loads(hits[0].read_text()) if hits else None
    except Exception:
        return None


def _mit_derived(eval_root: Path, ds: str) -> dict | None:
    hits = list((eval_root / f"GPT_OSS_20B__mitigation__{ds}").rglob("mit_summary.json"))
    try:
        return (json.loads(hits[0].read_text()).get("derived") or {}) if hits else None
    except Exception:
        return None


def ci95_of_mean(ps: list[float], ns: list[int]) -> float:
    """95% CI half-width (fraction) of the unweighted mean of Bernoulli proportions,
    propagated from each dataset's p(1-p)/n (old repo's normal-approx basis)."""
    k = len(ps)
    if k == 0:
        return 0.0
    var = sum(p * (1.0 - p) / n for p, n in zip(ps, ns) if n and n > 0) / (k * k)
    return Z * math.sqrt(var)


# ---- per-test (direct, dragged, n_trials) rows across datasets ----
def rows_drag(eval_root, test):
    """drag (2F) / ec-external / ec-posthoc: from results.json. n_trials = n_kept * n_samples."""
    out = []
    for ds in DATASETS:
        r = _result(eval_root, test, ds)
        if not r or r.get("aggregate_failed") or r.get("filter_dropped_all"):
            continue
        di = r.get("acc_clean", r.get("acc_direct"))
        dr = r.get("acc_2f", r.get("acc_conditioned"))
        npr = r.get("n_kept_problems")
        nsamp = r.get("n", EC_N_SAMPLES)        # drag stores n; EC falls back to 16
        if _ok(di) and _ok(dr) and npr:
            out.append((di, dr, int(npr) * int(nsamp)))
    return out


def rows_drag_1f(eval_root):
    """raw/unfiltered 1F from EC post-hoc. n_trials = num_problems_cond * EC_N_SAMPLES."""
    out = []
    for ds in DATASETS:
        s = _ec_summary(eval_root, ds)
        if not s:
            continue
        di, dr, npc = s.get("correctness_raw_init_sampling"), s.get("correctness_raw"), s.get("num_problems_cond")
        if _ok(di) and _ok(dr) and npc:
            out.append((di, dr, int(npc) * EC_N_SAMPLES))
    return out


def rows_mit(eval_root):
    """mitigation: (direct, dragged_1f, mitigated, n_observations). n_trials = n_obs (thresholded per obs)."""
    out = []
    for ds in DATASETS:
        s = _mit_derived(eval_root, ds)
        if not s or not s.get("n"):
            continue
        if all(_ok(s.get(k)) for k in ("direct_acc", "f1_acc", "mit_acc")):
            out.append((s["direct_acc"], s["f1_acc"], s["mit_acc"], int(s["n"])))
    return out


def _p1(sd: Path):
    for p in sorted(sd.glob("evaluated_*_error_analysis.json")):
        try:
            ov = json.loads(p.read_text()).get("overall_stats") or {}
        except Exception:
            continue
        if "pass_at_1" in ov:
            return float(ov["pass_at_1"])
    return None


def recursive_pools(recursive_glob, max_steps=16):
    """var -> step -> list of (pass@1, n_trials). n_trials = n_problems * n_samples
    (round 0: 1 picked draft/problem; rounds >=1: REC_N_SOLVE rollouts/problem)."""
    pools = {"rf1": {}, "naive": {}}
    for runs_root in glob.glob(recursive_glob.rstrip("/") + "/**/runs", recursive=True):
        rr = Path(runs_root)
        var = "naive" if "recursive_naive" in str(rr) else ("rf1" if "recursive_filter1" in str(rr) else None)
        if var is None:
            continue
        for rdir in sorted(rr.glob("run_*/recursive")):
            r0 = rdir / "round0_verification.json"
            if not r0.is_file():
                continue
            try:
                j0 = json.loads(r0.read_text())
            except Exception:
                continue
            nprob = int(j0.get("n_problems") or 0)
            if nprob <= 0:
                continue
            pools[var].setdefault(0, []).append((float(j0.get("pass_at_1") or 0.0), nprob))
            for s in range(1, max_steps + 1):
                sd = rdir / f"round{s}" / "solve"
                if sd.is_dir():
                    a = _p1(sd)
                    if a is not None:
                        pools[var].setdefault(s, []).append((a, nprob * REC_N_SOLVE))
    return pools


def _bar(ax, labels, means, cis, colors, title, ns, hline=None):
    x = range(len(labels))
    bars = ax.bar(x, [m * 100 for m in means], yerr=[c * 100 for c in cis],
                  color=colors, capsize=5, width=0.6, error_kw=dict(elinewidth=1.5, ecolor="#333"))
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylim(0, 100); ax.set_ylabel("pass@1 (%)")
    ax.set_title(f"{title}\n(mean ± 95% CI over {ns} datasets)", fontsize=9.5)
    for b, m, c in zip(bars, means, cis):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + c * 100 + 1.5,
                f"{m*100:.1f}±{c*100:.1f}", ha="center", va="bottom", fontsize=8)
    if hline is not None:
        ax.axhline(hline * 100, ls="--", color="#4C72B0", lw=1.2, label=f"DIRECT ({hline*100:.1f})")
        ax.legend(fontsize=8, loc="upper right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", default=None,
                    help="drag/EC/mitigation eval root; default = newest runs/full_oss20b_*/eval")
    ap.add_argument("--recursive_glob", default=str(REPO / "runs/recursive_full_*"))
    ap.add_argument("--out", default=str(REPO / "runs/gpt_oss_20b_summary.png"))
    args = ap.parse_args()
    if args.eval_root:
        eval_root = Path(args.eval_root)
    else:
        # newest runs/*/eval that actually holds GPT_OSS_20B results (any run-dir naming)
        cands = [p for p in glob.glob(str(REPO / "runs/*/eval"))
                 if glob.glob(p + "/GPT_OSS_20B__*")]
        eval_root = (Path(max(cands, key=lambda p: Path(p).stat().st_mtime))
                     if cands else (REPO / "runs"))
    print(f"[viz] eval_root={eval_root}")

    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.1], hspace=0.45, wspace=0.38)

    panels = [
        ("Drag (1F, raw)", rows_drag_1f(eval_root)),
        ("Drag (2F)", rows_drag(eval_root, "drag")),
        ("Error-cond post-hoc\n(1F, verdict-filtered)", rows_drag(eval_root, "error-conditioning-posthoc")),
        ("Error-cond external\n(1F, framing)", rows_drag(eval_root, "error-conditioning-external")),
    ]
    summary = []
    for i, (title, rows) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        if not rows:
            ax.text(0.5, 0.5, f"{title}\n(no data)", ha="center", va="center"); ax.axis("off"); continue
        d = [r[0] for r in rows]; g = [r[1] for r in rows]; ns = [r[2] for r in rows]
        cd, cg = ci95_of_mean(d, ns), ci95_of_mean(g, ns)
        _bar(ax, ["DIRECT", "DRAGGED"], [fmean(d), fmean(g)], [cd, cg],
             ["#4C72B0", "#C44E52"], title, len(d))
        summary.append((title.replace("\n", " "), fmean(d), cd, fmean(g), cg, len(d)))

    # mitigation
    ax = fig.add_subplot(gs[0, 4])
    mrows = rows_mit(eval_root)
    if mrows:
        direct = [r[0] for r in mrows]; f1 = [r[1] for r in mrows]; mit = [r[2] for r in mrows]; ns = [r[3] for r in mrows]
        cf, cm = ci95_of_mean(f1, ns), ci95_of_mean(mit, ns)
        _bar(ax, ["DRAGGED (1F)", "MITIGATED"], [fmean(f1), fmean(mit)], [cf, cm],
             ["#C44E52", "#55A868"], "Mitigation (Filter)", len(f1), hline=fmean(direct))
        summary.append(("Mitigation: dragged->mitigated", fmean(f1), cf, fmean(mit), cm, len(f1)))
    else:
        ax.text(0.5, 0.5, "Mitigation\n(no data)", ha="center", va="center"); ax.axis("off")

    # recursive trend: 95% CI of per-step mean (1.96 * SEM across pooled trajectories)
    axr = fig.add_subplot(gs[1, :])
    pools = recursive_pools(args.recursive_glob)
    colors = {"naive": "#C44E52", "rf1": "#55A868"}
    labels = {"naive": "naive (unfiltered)", "rf1": "filtered (rf1)"}
    for var in ("naive", "rf1"):
        steps = sorted(pools[var])
        if not steps:
            continue
        means = [fmean([p for p, _ in pools[var][s]]) * 100 for s in steps]
        ci = [ci95_of_mean([p for p, _ in pools[var][s]], [n for _, n in pools[var][s]]) * 100
              for s in steps]
        nmax = max(len(v) for v in pools[var].values())
        axr.plot(steps, means, "-o", color=colors[var], ms=4,
                 label=f"{labels[var]} (n≈{nmax} traj×ds)")
        axr.fill_between(steps, [m - c for m, c in zip(means, ci)], [m + c for m, c in zip(means, ci)],
                         color=colors[var], alpha=0.18)
    axr.set_xlabel("recursive step"); axr.set_ylabel("pass@1 (%)")
    axr.set_title("Recursive self-improvement: naive vs filtered "
                  "(mean ± 95% Bernoulli CI over pooled trajectories × 4 math datasets)", fontsize=11)
    axr.set_xticks(range(0, 17)); axr.grid(alpha=0.3); axr.legend(fontsize=10)

    fig.suptitle("GPT-OSS-20B — contextual drag summary (mean ± 95% CI, averaged across datasets per task)",
                 fontsize=13, y=0.98)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}\n=== numeric summary (mean ± 95% CI, pp) ===")
    for name, a, ca, b, cb, n in summary:
        print(f"  {name:42s}: {a*100:5.1f}±{ca*100:.1f}  ->  {b*100:5.1f}±{cb*100:.1f}  (n={n} ds)")
    for var in ("rf1", "naive"):
        if pools[var]:
            ss = sorted(pools[var]); last = ss[-1]
            def _m(step): return fmean([p for p, _ in pools[var][step]]) * 100
            def _c(step): return ci95_of_mean([p for p, _ in pools[var][step]], [n for _, n in pools[var][step]]) * 100
            print(f"  Recursive {var:5s}: r0={_m(0):.1f}±{_c(0):.1f}  "
                  f"r{last}={_m(last):.1f}±{_c(last):.1f}  (n≈{max(len(v) for v in pools[var].values())})")


if __name__ == "__main__":
    main()
