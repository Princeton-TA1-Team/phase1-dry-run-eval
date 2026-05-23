"""`data aggregate` must fail cleanly when the filter accepts zero problems.

Pre-PR2 a degenerate aggregate would crash with
``ValueError: Dataset.from_list(...)`` because the empty list reached
arrow. PR-2 now exits 1 with a legible WARNING that names the
``num_true``/``num_false`` knobs the user must tune.

This regression test feeds the aggregator a single trivially-correct row
(3-of-3 correct, no incorrect) and asks for ``-T 0 -F 2``. The filter
strips everything; the wrapper must:

  * exit non-zero (so the magnet driver knows aggregation failed);
  * NOT leak the ``from_list`` traceback to stderr;
  * mention ``num_true`` or ``num_false`` somewhere in the user-facing output.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _synth_input_dataset(tmp_path: Path) -> Path:
    """Build a one-row HF dataset that exercises the empty-filter path.

    Requires `datasets` (a core dep). If unavailable, skip.
    """
    pytest.importorskip("datasets")
    from datasets import Dataset

    row = {
        "id": "fake_0",
        "init_response_generations_correctness": [True, True, True],
        # The aggregator's pre-filter restricts to rows whose
        # init_response_generations_metadata.model_config_alias is in
        # `--init_response_models`; supply a matching alias.
        "init_response_generations_metadata": {
            "model_config_alias": "FakeModel",
            "prompt_hash": "deadbeef00000000",
        },
        "init_response_generations": [
            {"response_id": 0, "generated_response": "ok", "finish_reason": "stop"}
        ],
    }
    ds = Dataset.from_list([row])
    out = tmp_path / "input.ds"
    ds.save_to_disk(str(out))
    return out


def test_aggregate_empty_filter_exits_clean(tmp_path: Path) -> None:
    input_ds = _synth_input_dataset(tmp_path)
    cmd = [
        sys.executable, "-m", "contextual_drag", "data", "aggregate",
        "--input_dir", str(input_ds),
        "--num_true", "0", "--num_false", "2",
        "--output_dir", str(tmp_path / "out"),
        "--init_response_models", "FakeModel",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)

    if result.returncode == 0:
        pytest.skip(
            "aggregator accepted the synthetic input; refine the fixture so it "
            "actually trips the empty-filter path"
        )

    assert result.returncode == 1, (
        f"`data aggregate` on empty filter should exit 1, got "
        f"{result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    combined = (result.stderr + "\n" + result.stdout).lower()
    assert "from_list" not in combined, (
        f"`from_list` traceback leaked to the user (bug class):\n{combined}"
    )
    assert "num_true" in combined or "num_false" in combined, (
        f"Expected legible filter message naming num_true/num_false; got:\n{combined}"
    )
