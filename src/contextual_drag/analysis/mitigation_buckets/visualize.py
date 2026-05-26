"""Unified figure summarizing the mitigation outcome-bucket decomposition.

Reads ``<results-dir>/stats.json`` and writes
``<results-dir>/fig_mitigation_buckets.{pdf,png}``.

Four panels in a 2×2 grid (with width_ratios=[1.55, 1.0]):

  (A) 8-bucket stacked bars per (model × mitigation).
  (B) Recovery vs Iatrogenic scatter, equal aspect.
  (C) Filter vs Revise complementarity on drag-induced failures.
  (D) Per-task recovery vs iatrogenic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from contextual_drag.analysis.mitigation_buckets import buckets, cross

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42

MODEL_DISPLAY = {
    "GPT_OSS_120B": "G-120B", "GPT_OSS_20B":  "G-20B",
    "Nemotron_32B": "N-32B",  "Nemotron_7B":  "N-7B",
    "Qwen3_32B":    "Q3-32B", "Qwen3_8B":     "Q3-8B",
    "Gemma4_31B":   "G4-31B", "Gemma4_E4B":   "G4-E4B",
    "LlamaR1_8B":   "R1-L8B", "QwenR1_7B":    "R1-Q7B",
}
TASK_DISPLAY = {t: t for t in ["aime24", "aime25", "hmmt24", "hmmt25",
                               "gpqa", "mmlu", "crux-i", "24-game"]}

BUCKET_COLOR = {
    "unaffected":       "#BDBDBD",
    "iatrogenic_harm":  "#B91C1C",
    "recoverable_drag": "#22C55E",
    "persistent_drag":  "#FCA5A5",
    "drag_helps_kept":  "#86EFAC",
    "drag_helps_lost":  "#F59E0B",
    "new_gain":         "#0EA5E9",
    "robust_failure":   "#525252",
}
BUCKET_LABEL_SHORT = {
    "unaffected":       "Unaffected (D✓ 1F✓ M✓)",
    "iatrogenic_harm":  "Iatrogenic harm (D✓ 1F✓ M✗)",
    "recoverable_drag": "Recoverable drag (D✓ 1F✗ M✓)",
    "persistent_drag":  "Persistent drag (D✓ 1F✗ M✗)",
    "drag_helps_kept":  "Drag helps, kept (D✗ 1F✓ M✓)",
    "drag_helps_lost":  "Drag helps, lost (D✗ 1F✓ M✗)",
    "new_gain":         "New gain (D✗ 1F✗ M✓)",
    "robust_failure":   "Robust failure (D✗ 1F✗ M✗)",
}
BUCKET_ORDER = [
    "unaffected", "drag_helps_kept", "recoverable_drag", "new_gain",
    "iatrogenic_harm", "drag_helps_lost", "persistent_drag", "robust_failure",
]

MIT_KEYS  = ["cm_filter1", "cm_revise1"]
MIT_LABEL = {"cm_filter1": "Filter", "cm_revise1": "Revise"}
MIT_MARKER = {"cm_filter1": "o", "cm_revise1": "^"}


def _agg_counts(cells, filter_fn):
    out = buckets.empty_counts()
    for cell in cells:
        if cell["status"] != "ok" or not filter_fn(cell):
            continue
        for b in buckets.BUCKET_NAMES:
            out[b] += cell["counts"].get(b, 0)
    return out


def _per_model_mit_counts(cells, models):
    return {(m, mk): _agg_counts(cells, lambda x, m=m, mk=mk: x["model"] == m and x["mitigation"] == mk)
            for m in models for mk in MIT_KEYS}


def _per_task_mit_counts(cells, tasks):
    return {(t, mk): _agg_counts(cells, lambda x, t=t, mk=mk: x["task"] == t and x["mitigation"] == mk)
            for t in tasks for mk in MIT_KEYS}


def _per_model_drag_failed_counts(cross_cells, models):
    by_model = {m: {"FF": 0, "FT": 0, "TF": 0, "TT": 0} for m in models}
    for cc in cross_cells:
        if cc.get("status") != "ok":
            continue
        for k, v in cc["regimes"].get("drag_failed", {}).items():
            by_model[cc["model"]][k] += v
    return by_model


def _fmt_n(n: int) -> str:
    if n >= 1000:
        return f"{n/1000:.1f}k".replace(".0k", "k")
    return str(n)


def _style_axes(ax) -> None:
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5, color="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(0.8)


def _panel_buckets_by_model(ax, counts_by_mm, models) -> None:
    n_models = len(models)
    bar_w = 0.40
    inner_gap = 0.04
    x = np.arange(n_models)
    pos = {(m, mk): x[i] + (-bar_w / 2 - inner_gap / 2 if mk == "cm_filter1" else bar_w / 2 + inner_gap / 2)
           for i, m in enumerate(models) for mk in MIT_KEYS}
    hatch_for = {"cm_filter1": "", "cm_revise1": "////"}

    fracs = {}
    n_obs = {}
    for (m, mk), counts in counts_by_mm.items():
        n = sum(counts.values())
        n_obs[(m, mk)] = n
        fracs[(m, mk)] = {b: (counts[b] / n if n else 0.0) for b in buckets.BUCKET_NAMES}

    for mk in MIT_KEYS:
        bottom = np.zeros(n_models)
        for b in BUCKET_ORDER:
            heights = np.array([fracs[(m, mk)][b] for m in models])
            xs = np.array([pos[(m, mk)] for m in models])
            ax.bar(xs, heights, bar_w, bottom=bottom,
                   color=BUCKET_COLOR[b], edgecolor="white", linewidth=0.3,
                   hatch=hatch_for[mk])
            bottom += heights

    for i, m in enumerate(models):
        n = n_obs[(m, "cm_filter1")]
        ax.text(x[i], 1.025, f"n={_fmt_n(n)}", ha="center", va="bottom",
                fontsize=7.0, color="#444")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.10)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(v*100)}%" for v in np.linspace(0, 1, 6)])
    ax.set_ylabel("Share of (problem, draft) observations", fontsize=10)
    ax.set_title("(A) Outcome buckets per model × mitigation", fontsize=11, pad=8, loc="left")

    h_solid = mpatches.Patch(facecolor="#9CA3AF", edgecolor="white", label="Filter (solid)")
    h_hatch = mpatches.Patch(facecolor="#9CA3AF", edgecolor="white", hatch="////", label="Revise (hatched)")
    ax.legend(handles=[h_solid, h_hatch], loc="upper right",
              bbox_to_anchor=(1.0, 1.0), fontsize=7.5, framealpha=0.95,
              ncol=2, handlelength=1.3, columnspacing=1.0, handletextpad=0.4)
    _style_axes(ax)


def _panel_recovery_iatrogenic(ax, counts_by_mm, models) -> None:
    cmap = plt.get_cmap("tab10")
    color_for_model = {m: cmap(i % 10) for i, m in enumerate(models)}

    AX_MAX = 60.0
    for (m, mk), counts in counts_by_mm.items():
        met = buckets.derive_metrics(counts)
        if met.get("recovery_rate") is None or met.get("iatrogenic_rate") is None:
            continue
        x = met["iatrogenic_rate"] * 100
        y = met["recovery_rate"]   * 100
        ax.scatter(x, y, marker=MIT_MARKER[mk], s=110,
                   facecolor=color_for_model[m], edgecolor="black", linewidth=0.9,
                   alpha=0.92, zorder=3)
        ax.annotate(MODEL_DISPLAY.get(m, m), (x, y),
                    xytext=(5, 4), textcoords="offset points",
                    fontsize=6.8, color="#222", zorder=4)

    ax.plot([0, AX_MAX], [0, AX_MAX], color="gray", linestyle="--", linewidth=0.7, alpha=0.6, zorder=1)
    ax.text(AX_MAX * 0.55, AX_MAX * 0.50, "y = x\n(rescue = harm)",
            fontsize=7, color="gray", rotation=45, alpha=0.75, ha="center")
    ax.text(2.5, AX_MAX - 3, "Ideal\n(low harm,\nhigh recovery)",
            fontsize=7.2, color="#22C55E", weight="bold", va="top")
    ax.text(AX_MAX - 3, 3, "Worst\n(high harm,\nlow recovery)",
            fontsize=7.2, color="#B91C1C", weight="bold", ha="right", va="bottom")

    ax.set_xlim(0, AX_MAX)
    ax.set_ylim(0, AX_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Iatrogenic rate %  (D ✓ ∧ 1F ✓ → Mit ✗)", fontsize=10)
    ax.set_ylabel("Recovery rate %  (D ✓ ∧ 1F ✗ → Mit ✓)", fontsize=10)
    ax.set_title("(B) Recovery vs Iatrogenic tradeoff", fontsize=11, pad=8, loc="left")
    _style_axes(ax)
    ax.grid(True, alpha=0.25, linewidth=0.5, color="gray")

    handles = [plt.Line2D([0], [0], marker=MIT_MARKER[mk], color="black",
                          markerfacecolor="white", markersize=8, linestyle="none",
                          label=MIT_LABEL[mk])
               for mk in MIT_KEYS]
    ax.legend(handles=handles, loc="lower right",
              fontsize=8, framealpha=0.95, handletextpad=0.4, borderpad=0.5)


def _panel_cross_drag_failed(ax, by_model, models) -> None:
    n = len(models)
    x = np.arange(n)
    bar_w = 0.65

    frac_TT = np.zeros(n); frac_TF = np.zeros(n)
    frac_FT = np.zeros(n); frac_FF = np.zeros(n)
    for i, m in enumerate(models):
        c = by_model[m]
        rn = sum(c.values())
        if rn:
            frac_TT[i] = c["TT"] / rn
            frac_TF[i] = c["TF"] / rn
            frac_FT[i] = c["FT"] / rn
            frac_FF[i] = c["FF"] / rn

    ax.bar(x, frac_TT, bar_w, color="#22C55E", edgecolor="white", linewidth=0.3, label="Both ✓")
    ax.bar(x, frac_TF, bar_w, bottom=frac_TT, color="#3B82F6",
           edgecolor="white", linewidth=0.3, label="Filter only ✓")
    ax.bar(x, frac_FT, bar_w, bottom=frac_TT + frac_TF, color="#A855F7",
           edgecolor="white", linewidth=0.3, label="Revise only ✓")
    ax.bar(x, frac_FF, bar_w, bottom=frac_TT + frac_TF + frac_FT, color="#9CA3AF",
           edgecolor="white", linewidth=0.3, label="Neither ✓")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(v*100)}%" for v in np.linspace(0, 1, 6)])
    ax.set_ylabel("Share within drag_failed regime\n(D ✓, 1F ✗)", fontsize=10)
    ax.set_title("(C) Filter vs Revise complementarity on drag-induced failures",
                 fontsize=11, pad=22, loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005),
              fontsize=8.0, frameon=False, ncol=4,
              handlelength=1.2, columnspacing=1.6, handletextpad=0.4)
    _style_axes(ax)


def _panel_per_task(ax, counts_by_tm, tasks) -> None:
    n_tasks = len(tasks)
    x = np.arange(n_tasks)
    bar_w = 0.20
    offsets = [-1.5 * bar_w, -0.5 * bar_w, 0.5 * bar_w, 1.5 * bar_w]

    rec_f = np.zeros(n_tasks); iat_f = np.zeros(n_tasks)
    rec_r = np.zeros(n_tasks); iat_r = np.zeros(n_tasks)
    for i, t in enumerate(tasks):
        for mk, rec_arr, iat_arr in (("cm_filter1", rec_f, iat_f), ("cm_revise1", rec_r, iat_r)):
            met = buckets.derive_metrics(counts_by_tm[(t, mk)])
            rec_arr[i] = (met.get("recovery_rate")   or 0) * 100
            iat_arr[i] = (met.get("iatrogenic_rate") or 0) * 100

    ax.bar(x + offsets[0], rec_f, bar_w, color="#22C55E", edgecolor="black", linewidth=0.3)
    ax.bar(x + offsets[1], iat_f, bar_w, color="#B91C1C", edgecolor="black", linewidth=0.3)
    ax.bar(x + offsets[2], rec_r, bar_w, color="#22C55E", edgecolor="black", linewidth=0.3,
           hatch="////", alpha=0.85)
    ax.bar(x + offsets[3], iat_r, bar_w, color="#B91C1C", edgecolor="black", linewidth=0.3,
           hatch="////", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_DISPLAY.get(t, t) for t in tasks], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Rate %", fontsize=10)
    ax.set_ylim(0, 90)
    ax.set_title("(D) Per-task recovery vs iatrogenic harm", fontsize=11, pad=22, loc="left")

    handles = [
        mpatches.Patch(facecolor="#22C55E", edgecolor="black", label="Recovery"),
        mpatches.Patch(facecolor="#B91C1C", edgecolor="black", label="Iatrogenic"),
        mpatches.Patch(facecolor="#9CA3AF", edgecolor="black", label="Filter (solid)"),
        mpatches.Patch(facecolor="#9CA3AF", edgecolor="black", hatch="////", label="Revise (hatch)"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.005),
              fontsize=8.0, frameon=False, ncol=4,
              handlelength=1.2, columnspacing=1.6, handletextpad=0.4)
    _style_axes(ax)


def _bucket_legend(fig) -> None:
    handles = [mpatches.Patch(color=BUCKET_COLOR[b], label=BUCKET_LABEL_SHORT[b])
               for b in BUCKET_ORDER]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.005),
               ncol=4, fontsize=8.5, frameon=True, framealpha=0.95,
               handlelength=1.6, columnspacing=1.6, handletextpad=0.5,
               title="Outcome buckets   (D = Direct • 1F = drag context • M = mitigation)",
               title_fontsize=9)


def _summary_text(data: dict):
    cells       = data["cells"]
    cross_cells = data.get("cross_cells", [])
    f_counts = _agg_counts(cells, lambda x: x["mitigation"] == "cm_filter1")
    r_counts = _agg_counts(cells, lambda x: x["mitigation"] == "cm_revise1")
    f_m = buckets.derive_metrics(f_counts)
    r_m = buckets.derive_metrics(r_counts)

    aT = sum(sum(cc["regimes"][r]["TT"] for r in cross.REGIMES) for cc in cross_cells if cc.get("status") == "ok")
    bT = sum(sum(cc["regimes"][r]["TF"] for r in cross.REGIMES) for cc in cross_cells if cc.get("status") == "ok")
    cT = sum(sum(cc["regimes"][r]["FT"] for r in cross.REGIMES) for cc in cross_cells if cc.get("status") == "ok")
    dT = sum(sum(cc["regimes"][r]["FF"] for r in cross.REGIMES) for cc in cross_cells if cc.get("status") == "ok")
    denom_sq = (aT + bT) * (cT + dT) * (aT + cT) * (bT + dT)
    phi = (aT * dT - bT * cT) / (denom_sq ** 0.5) if denom_sq > 0 else float("nan")

    a_dr = sum(cc["regimes"]["drag_failed"]["TT"] for cc in cross_cells if cc.get("status") == "ok")
    b_dr = sum(cc["regimes"]["drag_failed"]["TF"] for cc in cross_cells if cc.get("status") == "ok")
    c_dr = sum(cc["regimes"]["drag_failed"]["FT"] for cc in cross_cells if cc.get("status") == "ok")
    d_dr = sum(cc["regimes"]["drag_failed"]["FF"] for cc in cross_cells if cc.get("status") == "ok")
    n_dr = a_dr + b_dr + c_dr + d_dr

    line1 = (
        f"Filter recovery {f_m['recovery_rate']*100:.1f}%   vs   Revise {r_m['recovery_rate']*100:.1f}%  "
        f"   |    Iatrogenic harm: F {f_m['iatrogenic_rate']*100:.1f}%, R {r_m['iatrogenic_rate']*100:.1f}%  "
        f"   |    Net vs 1F: F {f_m['net_vs_1f']*100:+.1f}pp, R {r_m['net_vs_1f']*100:+.1f}pp"
    )
    line2 = (
        f"Filter–Revise φ = {phi:.2f}   "
        f"Drag-failed split: filter-only {b_dr/n_dr*100:.1f}%, revise-only {c_dr/n_dr*100:.1f}%, "
        f"both {a_dr/n_dr*100:.1f}%, neither {d_dr/n_dr*100:.1f}%   "
        f"(N: F {sum(f_counts.values()):,}; R {sum(r_counts.values()):,}; paired drag-failed {n_dr:,})"
    )
    return line1, line2


def render(data: dict):
    cfg         = data["config"]
    cells       = data["cells"]
    cross_cells = data.get("cross_cells", [])
    models      = cfg["models"]
    tasks       = cfg["tasks"]

    counts_by_mm   = _per_model_mit_counts(cells, models)
    counts_by_tm   = _per_task_mit_counts(cells, tasks)
    cross_by_model = _per_model_drag_failed_counts(cross_cells, models)

    fig = plt.figure(figsize=(13.5, 10.0), dpi=300, constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.55, 1.0],
        wspace=0.22, hspace=0.55,
        left=0.06, right=0.985, top=0.84, bottom=0.13,
    )
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    _panel_buckets_by_model(axA, counts_by_mm, models)
    _panel_recovery_iatrogenic(axB, counts_by_mm, models)
    _panel_cross_drag_failed(axC, cross_by_model, models)
    _panel_per_task(axD, counts_by_tm, tasks)

    fig.suptitle("What do the mitigations actually do?  Outcome decomposition of cm_filter1 / cm_revise1",
                 fontsize=13, fontweight="bold", y=0.965)
    line1, line2 = _summary_text(data)
    fig.text(0.5, 0.918, line1, ha="center", va="top", fontsize=9.0, color="#222")
    fig.text(0.5, 0.892, line2, ha="center", va="top", fontsize=8.5, color="#444")

    _bucket_legend(fig)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--stats", type=Path, default=_RESULTS / "stats.json")
    ap.add_argument("--out",   type=Path, default=_RESULTS / "fig_mitigation_buckets.pdf")
    args = ap.parse_args()

    if not args.stats.exists():
        print(f"stats.json not found at {args.stats}; run "
              f"`python -m contextual_drag.analysis.mitigation_buckets.run` first.",
              file=sys.stderr)
        sys.exit(2)
    data = json.loads(args.stats.read_text())
    fig = render(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
