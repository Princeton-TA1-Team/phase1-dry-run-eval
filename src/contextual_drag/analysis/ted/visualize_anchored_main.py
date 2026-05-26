"""Render a per-model TED bar chart from a directory of cache files.

Walks ``<cache-root>/<phase>/<MODEL>.json`` (each built by
:func:`contextual_drag.analysis.ted.edit_distance_analysis.build_cache_from_jsonls`)
and plots two bars per model: Direct (init responses) vs. Anchored,
with SEM error bars across problems.

Models can be restricted with ``--models``; default is every ``<MODEL>.json``
in the phase directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from contextual_drag.analysis.ted.edit_distance_analysis import compute_stats_for_dataset

DIRECT_COLOR = "#519ABA"
ANCHORED_COLOR = "#FF5A78"


def _discover_models(phase_dir: Path) -> list[str]:
    return sorted(p.stem for p in phase_dir.glob("*.json"))


def _load_per_model_means(phase_dir: Path, models: list[str], *, metric: str,
                          filter_verification: bool):
    out: dict[str, dict] = {}
    for model in models:
        cache_path = phase_dir / f"{model}.json"
        if not cache_path.exists():
            print(f"[warn] missing {cache_path}; skipping {model}")
            continue
        with open(cache_path) as f:
            processed_data = json.load(f)
        stats = compute_stats_for_dataset(
            processed_data, metric=metric,
            filter_verification=filter_verification,
        )
        anchored = [s["anchored_responses"] for s in stats if s["anchored_responses"] is not None]
        direct   = [s["init_response"]      for s in stats if s["init_response"]      is not None]
        out[model] = {
            "short":                  model,
            "anchored_problem_means": anchored,
            "direct_problem_means":   direct,
            "anchored_mean":          float(np.mean(anchored)) if anchored else None,
            "direct_mean":            float(np.mean(direct))   if direct   else None,
        }
        print(
            f"{model:<14s}  n_problems(anchored)={len(anchored):>4d}  "
            f"n_problems(direct)={len(direct):>4d}  "
            f"anchored_mean={out[model]['anchored_mean']!s:<6}  "
            f"direct_mean={out[model]['direct_mean']!s:<6}"
        )
    return out


def plot_mean_comparison(all_model_data: dict, output_path: Path, *,
                         metric: str, phase_label: str):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    models = list(all_model_data.keys())
    short_labels = [all_model_data[m]["short"] for m in models]

    direct_means, anchored_means = [], []
    direct_errors, anchored_errors = [], []
    for model in models:
        data = all_model_data[model]
        d_mean = data["direct_mean"]
        a_mean = data["anchored_mean"]
        d_problem_means = data["direct_problem_means"]
        a_problem_means = data["anchored_problem_means"]

        if d_mean is not None and len(d_problem_means) > 0:
            d_err = float(np.std(d_problem_means, ddof=1) / np.sqrt(len(d_problem_means))) if len(d_problem_means) > 1 else 0.0
        else:
            d_err = 0.0
        if a_mean is not None and len(a_problem_means) > 0:
            a_err = float(np.std(a_problem_means, ddof=1) / np.sqrt(len(a_problem_means))) if len(a_problem_means) > 1 else 0.0
        else:
            a_err = 0.0

        direct_means.append(d_mean)
        anchored_means.append(a_mean)
        direct_errors.append(d_err)
        anchored_errors.append(a_err)

    x = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.bar(
        x - width / 2, direct_means, width,
        label="Direct", color=DIRECT_COLOR, alpha=0.88,
        yerr=direct_errors, capsize=4,
        error_kw={"elinewidth": 1.3, "capthick": 1.3},
        edgecolor="black", linewidth=0.6,
    )
    ax.bar(
        x + width / 2, anchored_means, width,
        label=f"Contextual Drag ({phase_label.upper()})", color=ANCHORED_COLOR, alpha=0.88,
        yerr=anchored_errors, capsize=4,
        error_kw={"elinewidth": 1.3, "capthick": 1.3},
        edgecolor="black", linewidth=0.6,
    )

    all_top = []
    for m, e in zip(direct_means, direct_errors):
        if m is not None:
            all_top.extend([m + e, m - e])
    for m, e in zip(anchored_means, anchored_errors):
        if m is not None:
            all_top.extend([m + e, m - e])
    if all_top:
        y_min = min(all_top); y_max = max(all_top)
        rng = y_max - y_min if y_max > y_min else max(1.0, abs(y_max))
        ax.set_ylim(max(0, y_min - 0.15 * rng), y_max + 0.20 * rng)
        text_offset = 0.015 * rng
    else:
        text_offset = 0.0

    ax.set_xlabel("Model", fontsize=13)
    ax.set_ylabel(f"Mean {metric.title()} Edit Distance\nw.r.t. Draft in Context",
                  fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=11)
    ax.tick_params(axis="y", which="major", labelsize=11)

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.8, color="gray", axis="y")
    ax.set_axisbelow(True)

    for i, (a, d) in enumerate(zip(anchored_means, direct_means)):
        if d is not None:
            ax.text(i - width / 2, d + direct_errors[i] + text_offset, f"{d:.2f}",
                    ha="center", va="bottom", fontsize=9)
        if a is not None:
            ax.text(i + width / 2, a + anchored_errors[i] + text_offset, f"{a:.2f}",
                    ha="center", va="bottom", fontsize=9)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.06),
        ncol=2, fontsize=11, frameon=True, fancybox=False,
        edgecolor="black", framealpha=1, facecolor="white",
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path.with_suffix(".pdf")), bbox_inches="tight")
    plt.savefig(str(output_path.with_suffix(".png")), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[list[str]] = None, *, phase: Optional[str] = None,
         cache_root: Optional[Path] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", default="2f", help="Subdir under --cache-root.")
    p.add_argument("--metric", default="tree",
                   choices=("tree", "levenshtein", "binary"))
    p.add_argument("--filter-verification", action="store_true")
    p.add_argument("--cache-root", type=Path, required=cache_root is None,
                   default=cache_root,
                   help="Directory containing <phase>/<MODEL>.json caches.")
    p.add_argument("--models", default=None,
                   help="Comma list of models. Default: every *.json in <cache-root>/<phase>/.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output stem (without extension); writes .pdf and .png.")
    args = p.parse_args(argv)
    if phase is not None:
        args.phase = phase

    phase_dir = args.cache_root / args.phase
    if not phase_dir.is_dir():
        print(f"cache dir not found: {phase_dir}", file=sys.stderr)
        return 1

    models = args.models.split(",") if args.models else _discover_models(phase_dir)
    if not models:
        print(f"no cache files found in {phase_dir}", file=sys.stderr)
        return 1

    means = _load_per_model_means(phase_dir, models, metric=args.metric,
                                  filter_verification=args.filter_verification)
    if not means:
        print("no models loaded successfully.", file=sys.stderr)
        return 1

    plot_mean_comparison(means, args.out, metric=args.metric, phase_label=args.phase)
    print(f"wrote {args.out.with_suffix('.pdf')}")
    print(f"wrote {args.out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
