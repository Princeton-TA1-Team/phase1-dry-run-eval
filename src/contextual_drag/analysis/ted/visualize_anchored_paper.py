"""Paper-quality 1F-vs-Direct TED bar chart, rendered from a cache directory.

Like ``visualize_anchored_main`` but tighter typography, narrowed y-range,
and per-pair Δ annotations. Walks ``<cache-root>/<phase>/<MODEL>.json``;
restrict with ``--models``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from contextual_drag.analysis.ted.edit_distance_analysis import compute_stats_for_dataset

PHASE_LABEL = {"1f": "1F", "2f": "2F"}
COLOR_BASELINE = "#519ABA"
COLOR_BAD = "#FF5A78"


def style_spines(ax) -> None:
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)


def _discover_models(phase_dir: Path) -> list[str]:
    return sorted(p.stem for p in phase_dir.glob("*.json"))


def _load_per_model(phase_dir: Path, models: list[str], *, metric: str,
                    filter_verification: bool):
    rows = []
    for model in models:
        cache_path = phase_dir / f"{model}.json"
        if not cache_path.exists():
            print(f"[warn] missing {cache_path}; skipping {model}", file=sys.stderr)
            continue
        with open(cache_path) as f:
            processed = json.load(f)
        stats = compute_stats_for_dataset(
            processed, metric=metric, filter_verification=filter_verification,
        )
        anchored = [s["anchored_responses"] for s in stats if s["anchored_responses"] is not None]
        direct   = [s["init_response"]      for s in stats if s["init_response"]      is not None]
        if not anchored or not direct:
            print(f"[warn] empty stats for {model}", file=sys.stderr)
            continue
        rows.append({
            "model":      model,
            "short":      model,
            "n_problems": len(anchored),
            "direct":     float(np.mean(direct)),
            "direct_sem": float(np.std(direct, ddof=1) / np.sqrt(len(direct))) if len(direct) > 1 else 0.0,
            "anchored":   float(np.mean(anchored)),
            "anchored_sem": float(np.std(anchored, ddof=1) / np.sqrt(len(anchored))) if len(anchored) > 1 else 0.0,
        })
    return rows


def render(rows: list[dict], output_stem: Path, *, phase_dir: str, metric: str):
    if not rows:
        raise SystemExit("no data to render")

    short_labels = [r["short"] for r in rows]
    direct  = np.array([r["direct"]      for r in rows])
    drag    = np.array([r["anchored"]    for r in rows])
    d_err   = np.array([r["direct_sem"]  for r in rows])
    a_err   = np.array([r["anchored_sem"] for r in rows])
    deltas  = drag - direct

    x = np.arange(len(rows))
    width = 0.36
    phase_label = PHASE_LABEL.get(phase_dir, phase_dir.upper())

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.bar(
        x - width / 2, direct, width,
        label="Direct (no draft)",
        color=COLOR_BASELINE, alpha=0.92,
        yerr=d_err, capsize=3,
        error_kw={"elinewidth": 1.1, "capthick": 1.1, "ecolor": "black"},
        edgecolor="black", linewidth=0.7,
    )
    ax.bar(
        x + width / 2, drag, width,
        label=f"Contextual Drag ({phase_label})",
        color=COLOR_BAD, alpha=0.92,
        yerr=a_err, capsize=3,
        error_kw={"elinewidth": 1.1, "capthick": 1.1, "ecolor": "black"},
        edgecolor="black", linewidth=0.7,
    )

    y_min_data = float(min((drag - a_err).min(), (direct - d_err).min()))
    y_max_data = float(max((drag + a_err).max(), (direct + d_err).max()))
    y_lo = max(0.0, y_min_data - 0.20 * (y_max_data - y_min_data))
    y_hi = y_max_data + 0.22 * (y_max_data - y_min_data)
    ax.set_ylim(y_lo, y_hi)
    rng = y_hi - y_lo
    label_pad = 0.012 * rng
    delta_pad = 0.055 * rng

    for i, (d, a, de, ae) in enumerate(zip(direct, drag, d_err, a_err)):
        ax.text(i - width / 2, d + de + label_pad, f"{d:.2f}",
                ha="center", va="bottom", fontsize=9.5, color="black")
        ax.text(i + width / 2, a + ae + label_pad, f"{a:.2f}",
                ha="center", va="bottom", fontsize=9.5, color="black")

    top_per_pair = np.maximum(direct + d_err, drag + a_err)
    for i, (d, top) in enumerate(zip(deltas, top_per_pair)):
        ax.text(i, top + delta_pad, f"Δ {d:+.2f}",
                ha="center", va="bottom", fontsize=9, color=COLOR_BAD,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(f"Mean {metric.title()} Edit Distance\nw.r.t. Draft in Context", fontsize=11)

    style_spines(ax)
    ax.grid(True, axis="y", alpha=0.30, linestyle="-", linewidth=0.7, color="gray")
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.04),
        ncol=2, fontsize=11, frameon=True, fancybox=False,
        edgecolor="black", framealpha=1.0, facecolor="white",
        handlelength=1.8, columnspacing=2.0,
    )

    plt.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    plt.savefig(str(pdf_path), bbox_inches="tight")
    plt.savefig(str(png_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[wrote] {pdf_path}")
    print(f"[wrote] {png_path}")
    print()
    print(f"Per-model ({phase_label} vs Direct, {metric}):")
    print(f"  {'model':<14s}  {'n':>4s}  {'direct':>7s}  {phase_label:>7s}  {'Δ':>7s}")
    for r, dlt in zip(rows, deltas):
        print(f"  {r['short']:<14s}  {r['n_problems']:>4d}  "
              f"{r['direct']:>7.2f}  {r['anchored']:>7.2f}  {dlt:>+7.2f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--phase", default="1f", choices=("1f", "2f"))
    p.add_argument("--metric", default="tree", choices=("tree", "levenshtein", "binary"))
    p.add_argument("--filter-verification", action="store_true")
    p.add_argument("--cache-root", type=Path, required=True,
                   help="Directory containing <phase>/<MODEL>.json caches.")
    p.add_argument("--models", default=None,
                   help="Comma list. Default: every *.json in <cache-root>/<phase>/.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output stem (without extension); writes .pdf and .png.")
    args = p.parse_args(argv)

    phase_dir = args.cache_root / args.phase
    if not phase_dir.is_dir():
        print(f"cache dir not found: {phase_dir}", file=sys.stderr)
        return 1
    models = args.models.split(",") if args.models else _discover_models(phase_dir)
    if not models:
        print(f"no cache files found in {phase_dir}", file=sys.stderr)
        return 1

    rows = _load_per_model(phase_dir, models, metric=args.metric,
                           filter_verification=args.filter_verification)
    render(rows, args.out, phase_dir=args.phase, metric=args.metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
