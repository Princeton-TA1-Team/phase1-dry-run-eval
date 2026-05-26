"""Heatmap + LaTeX-table renderer for §3 error-conditioning results.

Reads ``<results-dir>/{1f,2f,framing}.json`` (each a
``{task: {model: cell}}`` mapping written by an upstream sweep) and emits:

  heatmap_{1f,2f,framing}.pdf   — task × model heatmap of Δacc(%)
  heatmap_combined.pdf           — three side-by-side panels
  table_{1f,2f,framing}.tex      — LaTeX table

Tasks and models are determined from the results dict by default; pass
``--tasks`` / ``--models`` to override the ordering or restrict to a subset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

_HERE = Path(__file__).resolve().parent

SETTING_TITLE = {
    "framing": "External (Prompted)",
    "1f": "Self-Detected (1F)",
    "2f": "Self-Detected (2F)",
}

SETTING_TITLE_RAW = {
    "framing": "External (Prompted)",
    "1f": "1F (no oracle filter)",
    "2f": "2F (no oracle filter)",
}

_CMAP = LinearSegmentedColormap.from_list(
    "delta_acc", ["#FF5A78", "white", "#519ABA"]
)


def _delta_for_setting(cell: dict, setting: str, mode: str = "filtered") -> Optional[float]:
    if cell.get("status") != "ok":
        return None
    if mode == "raw" or setting == "framing":
        a, b = cell.get("correctness_raw"), cell.get("correctness_raw_init_sampling")
    elif mode == "filtered":
        a, b = cell.get("correctness_filtered"), cell.get("correctness_filtered_init_sampling")
    else:
        raise ValueError(f"unknown mode {mode!r}")
    if a is None or b is None:
        return None
    return a - b


def load_results(setting: str, results_dir: Path) -> dict:
    p = results_dir / f"{setting}.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found.")
    return json.loads(p.read_text())


def infer_axes(data: dict) -> tuple[list[str], list[str]]:
    """Pull the task and model orderings out of a results dict."""
    tasks = list(data.keys())
    models: list[str] = []
    seen: set[str] = set()
    for t in tasks:
        for m in data[t].keys():
            if m not in seen:
                seen.add(m)
                models.append(m)
    return tasks, models


def build_matrix(data, setting, tasks, models, mode="filtered") -> np.ndarray:
    M = np.full((len(tasks), len(models)), np.nan)
    for i, t in enumerate(tasks):
        for j, m in enumerate(models):
            cell = data.get(t, {}).get(m, {})
            d = _delta_for_setting(cell, setting, mode=mode)
            if d is not None:
                M[i, j] = d
    return M


def plot_single_heatmap(ax, matrix, tasks, models, title, vmin, vmax):
    im = ax.imshow(matrix, cmap=_CMAP, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                txt = "—"
            else:
                pct = v * 100
                txt = f"+{pct:.1f}%" if pct > 0 else f"{pct:.1f}%"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=7)
    return im


def render_combined(matrices, tasks, models, out_path, title_map=None) -> None:
    title_map = title_map or SETTING_TITLE
    settings = ["framing", "1f", "2f"]
    all_vals = np.concatenate([matrices[s].ravel() for s in settings])
    all_vals = all_vals[~np.isnan(all_vals)]
    if all_vals.size == 0:
        print("[combined] no valid cells; skipping")
        return
    vmax = float(np.max(np.abs(all_vals))) + 0.01
    vmin = -vmax

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor("white")
    im = None
    for ax, s in zip(axes, settings):
        im = plot_single_heatmap(ax, matrices[s], tasks, models, title_map[s], vmin, vmax)

    plt.subplots_adjust(top=0.85, wspace=0.45, bottom=0.18, right=0.92)
    cbar_ax = fig.add_axes((0.93, 0.18, 0.012, 0.67))
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar.ax.tick_params(labelsize=9)
    ticks = np.linspace(vmin, vmax, 7)
    cbar.set_ticks(list(ticks))
    cbar.set_ticklabels([f"{t*100:+.0f}%" for t in ticks])
    fig.suptitle("Δ Accuracy vs. Direct Baseline", fontsize=13, y=0.97)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def render_single(matrix, tasks, models, setting, out_path, title_map=None) -> None:
    title_map = title_map or SETTING_TITLE
    vals = matrix[~np.isnan(matrix)]
    if vals.size == 0:
        print(f"[{setting}] no valid cells; skipping")
        return
    vmax = float(np.max(np.abs(vals))) + 0.01
    vmin = -vmax

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fig.patch.set_facecolor("white")
    im = plot_single_heatmap(
        ax, matrix, tasks, models, title_map.get(setting, setting), vmin, vmax
    )
    plt.subplots_adjust(top=0.88, right=0.85, bottom=0.22)
    cbar_ax = fig.add_axes((0.87, 0.22, 0.025, 0.66))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=9)
    ticks = np.linspace(vmin, vmax, 7)
    cbar.set_ticks(list(ticks))
    cbar.set_ticklabels([f"{t*100:+.0f}%" for t in ticks])
    fig.suptitle("Δ Accuracy vs. Direct Baseline", fontsize=12, y=0.97)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def render_latex_table(matrix, tasks, models, setting, out_path, title_map=None) -> None:
    title_map = title_map or SETTING_TITLE
    rows = []
    rows.append("% " + f"§3 error-conditioning ({title_map.get(setting, setting)}): Δ accuracy (%)")
    align = "l" + "r" * len(models)
    rows.append(r"\begin{tabular}{" + align + r"}")
    rows.append(r"\toprule")
    header = ["Task"] + list(models)
    rows.append(" & ".join(header) + r" \\")
    rows.append(r"\midrule")
    for i, t in enumerate(tasks):
        cells: list[str] = [t]
        for j in range(len(models)):
            v = matrix[i, j]
            if np.isnan(v):
                cells.append("---")
            else:
                pct = v * 100
                cells.append(f"{pct:+.1f}")
        rows.append(" & ".join(cells) + r" \\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n")
    print(f"  wrote {out_path}")


def _render_one_mode(settings, mode, tasks, models, results_dir):
    suffix = "_raw" if mode == "raw" else ""
    title_map = SETTING_TITLE_RAW if mode == "raw" else SETTING_TITLE
    matrices: dict[str, np.ndarray] = {}
    for s in settings:
        data = load_results(s, results_dir)
        # If caller didn't pass --tasks/--models, derive from the data.
        local_tasks, local_models = tasks, models
        if local_tasks is None or local_models is None:
            it, im_ = infer_axes(data)
            local_tasks = local_tasks or it
            local_models = local_models or im_
        matrices[s] = build_matrix(data, s, local_tasks, local_models, mode=mode)
        is_framing_raw = mode == "raw" and s == "framing"
        out_pdf = (
            results_dir / f"heatmap_{s}.pdf"
            if is_framing_raw
            else results_dir / f"heatmap_{s}{suffix}.pdf"
        )
        out_tex = (
            results_dir / f"table_{s}.tex"
            if is_framing_raw
            else results_dir / f"table_{s}{suffix}.tex"
        )
        render_single(matrices[s], local_tasks, local_models, s, out_pdf, title_map=title_map)
        render_latex_table(matrices[s], local_tasks, local_models, s, out_tex, title_map=title_map)
    return matrices


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--setting", choices=["1f", "2f", "framing"])
    p.add_argument("--mode", choices=["filtered", "raw", "both"], default="both")
    p.add_argument("--results-dir", type=Path, default=_HERE / "results")
    p.add_argument("--tasks", default=None,
                   help="Comma-separated task order. Default: order in results JSON.")
    p.add_argument("--models", default=None,
                   help="Comma-separated model order. Default: order in results JSON.")
    return p.parse_args()


def main(setting: Optional[str] = None) -> None:
    args = parse_args()
    if setting is not None:
        args.setting = setting

    tasks = args.tasks.split(",") if args.tasks else None
    models = args.models.split(",") if args.models else None

    settings = [args.setting] if args.setting else ["framing", "1f", "2f"]

    if args.mode in ("filtered", "both"):
        # Need a consistent task/model axis across the combined panel —
        # infer from the framing file if not supplied.
        if tasks is None or models is None:
            any_data = load_results(settings[0], args.results_dir)
            it, im_ = infer_axes(any_data)
            tasks = tasks or it
            models = models or im_
        matrices = _render_one_mode(settings, "filtered", tasks, models, args.results_dir)
        if not args.setting and len(matrices) == 3:
            render_combined(matrices, tasks, models, args.results_dir / "heatmap_combined.pdf")
    if args.mode in ("raw", "both"):
        if tasks is None or models is None:
            any_data = load_results(settings[0], args.results_dir)
            it, im_ = infer_axes(any_data)
            tasks = tasks or it
            models = models or im_
        matrices_raw = _render_one_mode(settings, "raw", tasks, models, args.results_dir)
        if not args.setting and len(matrices_raw) == 3:
            render_combined(
                matrices_raw,
                tasks,
                models,
                args.results_dir / "heatmap_combined_raw.pdf",
                title_map=SETTING_TITLE_RAW,
            )


if __name__ == "__main__":
    main()
