"""Generate a markdown report from ``stats.json``.

Sections: summary, aggregate buckets, per-model rollup, per-task rollup,
8-bucket breakdown, Filter vs Revise cross contingency, auto-generated
findings.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from contextual_drag.analysis.mitigation_buckets import buckets, cross

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results"

MITIGATION_LABEL = {"cm_filter1": "Filter", "cm_revise1": "Revise"}
BUCKET_PRETTY = {
    "unaffected":       "Unaffected (D✓ 1F✓ M✓)",
    "iatrogenic_harm":  "Iatrogenic harm (D✓ 1F✓ M✗)",
    "recoverable_drag": "Recoverable drag (D✓ 1F✗ M✓)",
    "persistent_drag":  "Persistent drag (D✓ 1F✗ M✗)",
    "drag_helps_kept":  "Drag helps, kept (D✗ 1F✓ M✓)",
    "drag_helps_lost":  "Drag helps, lost (D✗ 1F✓ M✗)",
    "new_gain":         "New gain (D✗ 1F✗ M✓)",
    "robust_failure":   "Robust failure (D✗ 1F✗ M✗)",
}


def _fmt_pct(x: Optional[float], digits: int = 1) -> str:
    if x is None:
        return "—"
    return f"{x*100:.{digits}f}%"


def _fmt_signed_pct(x: Optional[float], digits: int = 1) -> str:
    if x is None:
        return "—"
    return f"{x*100:+.{digits}f}%"


def _fmt_int(x) -> str:
    return "—" if x is None else f"{x:,}"


def _agg_counts(cells, filter_fn):
    out = buckets.empty_counts()
    for cell in cells:
        if not filter_fn(cell):
            continue
        if cell["status"] != "ok":
            continue
        for b in buckets.BUCKET_NAMES:
            out[b] += cell["counts"].get(b, 0)
    return out


def _section_header(text: str, level: int = 2) -> str:
    return f"{'#' * level} {text}\n"


def _render_summary(data: dict) -> str:
    cfg = data["config"]
    cells = data["cells"]
    cross_cells = data.get("cross_cells", [])
    n_total = len(cells)
    n_ok = sum(1 for c in cells if c["status"] == "ok")
    n_missing = n_total - n_ok
    by_status = defaultdict(int)
    for c in cells:
        by_status[c["status"]] += 1

    total_obs_filter = sum(c["n_observations"] for c in cells if c["status"] == "ok" and c["mitigation"] == "cm_filter1")
    total_obs_revise = sum(c["n_observations"] for c in cells if c["status"] == "ok" and c["mitigation"] == "cm_revise1")
    total_paired = sum(c["n_observations"] for c in cross_cells if c.get("status") == "ok")

    lines = [
        "# Mitigation Outcome-Bucket Decomposition",
        "",
        f"- **Threshold**: `{cfg['threshold']}`",
        f"- **Models**: {len(cfg['models'])} ({', '.join(cfg['models'])})",
        f"- **Tasks**: {len(cfg['tasks'])} ({', '.join(cfg['tasks'])})",
        f"- **Cells**: {n_total}   (ok: {n_ok}, missing: {n_missing})",
        f"- **Filter observations**: {total_obs_filter:,}",
        f"- **Revise observations**: {total_obs_revise:,}",
        f"- **Paired observations**: {total_paired:,}",
        "",
    ]
    return "\n".join(lines)


def _render_aggregate(cells) -> str:
    lines = [_section_header("1. Aggregate bucket distribution"), ""]
    header = ["Mitigation", "N obs"] + [BUCKET_PRETTY[b] for b in buckets.BUCKET_NAMES]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for mit_key, mit_label in MITIGATION_LABEL.items():
        c = _agg_counts(cells, lambda x, mk=mit_key: x["mitigation"] == mk)
        n = sum(c.values())
        row = [mit_label, _fmt_int(n)] + [
            _fmt_pct(c[b] / n if n else None) for b in buckets.BUCKET_NAMES
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def _render_per_axis(cells, axis: str, axis_values: list, section_num: int) -> str:
    lines = [_section_header(f"{section_num}. Per-{axis} derived metrics"), ""]
    head = [axis.capitalize(), "N", "Direct", "1F", "Mit", "Recovery", "Iatrogenic",
            "Preservation", "New-gain", "Net vs 1F", "Net vs Direct"]

    for mit_key, mit_label in MITIGATION_LABEL.items():
        lines.append(_section_header(f"{mit_label}", 3))
        lines.append("")
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "|".join(["---"] * len(head)) + "|")
        for v in axis_values:
            c = _agg_counts(cells, lambda x, val=v, mk=mit_key: x[axis] == val and x["mitigation"] == mk)
            m = buckets.derive_metrics(c)
            if m["n"] == 0:
                lines.append(f"| {v} | 0 | — | — | — | — | — | — | — | — | — |")
                continue
            row = [v, _fmt_int(m["n"]),
                   _fmt_pct(m["direct_acc"]),
                   _fmt_pct(m["f1_acc"]),
                   _fmt_pct(m["mit_acc"]),
                   _fmt_pct(m["recovery_rate"]),
                   _fmt_pct(m["iatrogenic_rate"]),
                   _fmt_pct(m["preservation_rate"]),
                   _fmt_pct(m["new_gain_rate"]),
                   _fmt_signed_pct(m["net_vs_1f"]),
                   _fmt_signed_pct(m["net_vs_direct"])]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def render(data: dict) -> str:
    cfg = data["config"]
    cells = data["cells"]
    parts = [
        _render_summary(data),
        _render_aggregate(cells),
        _render_per_axis(cells, "model", cfg["models"], section_num=2),
        _render_per_axis(cells, "task",  cfg["tasks"],  section_num=3),
    ]
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--stats", type=Path, default=_RESULTS / "stats.json")
    ap.add_argument("--out",   type=Path, default=_RESULTS / "REPORT.md")
    args = ap.parse_args()

    if not args.stats.exists():
        print(f"stats.json not found at {args.stats}.", file=sys.stderr)
        sys.exit(2)
    data = json.loads(args.stats.read_text())
    md = render(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(f"Wrote → {args.out}  ({len(md):,} chars)")


if __name__ == "__main__":
    main()
