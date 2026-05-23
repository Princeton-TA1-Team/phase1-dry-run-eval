"""Per-cell stats tracking + periodic status reporter + small formatters.

CellStats accumulates row/error/abort counts and token totals as a cell runs,
and renders a one-line snapshot used by both the periodic background reporter
and the per-16-rows progress prints in process_cell.
"""

import asyncio
import time


def format_secs(s: float) -> str:
    if s == float("inf") or s != s:  # inf or NaN
        return "?"
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"


def format_gb(bytes_: int) -> str:
    return f"{bytes_/2**30:.1f}GB"


class CellStats:
    """Per-cell counters: rows / errors / aborted, token totals, finish reasons."""

    def __init__(self, total: int, n_samples: int):
        self.total = total
        self.n_samples = n_samples
        self.completed_rows = 0
        self.errors = 0
        self.aborted = 0
        self.prompt_tokens_seen = 0   # one prompt counted once even with n>1
        self.completion_tokens = 0    # summed across all n samples
        self.finish_reasons: dict[str, int] = {}
        self.start_time = time.time()

    def record(self, output) -> None:
        """Update counters from a completed RequestOutput-like object."""
        self.completed_rows += 1
        self.prompt_tokens_seen += len(output.prompt_token_ids or [])
        for o in output.outputs:
            self.completion_tokens += len(o.token_ids or [])
            r = o.finish_reason or "unknown"
            self.finish_reasons[r] = self.finish_reasons.get(r, 0) + 1

    def snapshot_line(self) -> str:
        elapsed = time.time() - self.start_time
        rows_per_s = self.completed_rows / max(elapsed, 1e-3)
        tok_per_s = self.completion_tokens / max(elapsed, 1e-3)
        remaining = max(0, self.total - self.completed_rows)
        eta = remaining / rows_per_s if rows_per_s > 0 else float("inf")
        return (
            f"rows={self.completed_rows}/{self.total} err={self.errors} "
            f"abort={self.aborted}  "
            f"compl_tok={self.completion_tokens} "
            f"({rows_per_s:.2f} rows/s, {tok_per_s:.0f} tok/s)  "
            f"ETA {format_secs(eta)}"
        )


async def status_reporter(stats: CellStats, interval: float, label: str):
    """Background task — print stats.snapshot_line() every `interval` seconds.
    Cancelled by process_cell when the cell finishes."""
    try:
        while True:
            await asyncio.sleep(interval)
            print(f"  [status {label}] {stats.snapshot_line()}", flush=True)
    except asyncio.CancelledError:
        return
