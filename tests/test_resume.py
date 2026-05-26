"""Resume-checkpoint sanity tests for the inference JSONL scanner.

The contract: ``load_completed_hashes(path, task_name)`` returns the set
of ``<task_name>_generations_metadata.prompt_hash`` values found in a
JSONL file, tolerant of partial trailing lines (kill-restart case).

These are the only safety net between "kill mid-run" and "duplicate
generation cost" — worth pinning behaviour-level rather than letting the
shape drift silently.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _row(task: str, prompt_hash: str, **extra) -> dict:
    """Build a minimal JSONL row matching the shape build_record emits."""
    return {
        "id": extra.get("id", f"problem_{prompt_hash[:4]}"),
        f"{task}_prompt": "irrelevant",
        f"{task}_generations": [],
        f"{task}_generations_metadata": {
            "prompt_hash": prompt_hash,
            "num_responses": 0,
            "n_samples_requested": 1,
            "n_samples_variant_default": 1,
            "template_path": "irrelevant",
            "template_key": "irrelevant",
            "model_config_alias": "irrelevant",
        },
    }


def test_load_completed_hashes_skips_existing_rows(tmp_path: Path) -> None:
    """Two well-formed rows → scanner returns those two hashes."""
    from contextual_drag.inference.prompts import load_completed_hashes

    jsonl = tmp_path / "completions.jsonl"
    rows = [_row("init_response", "deadbeef" + "00" * 4),
            _row("init_response", "cafebabe" + "11" * 4)]
    with open(jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    done = load_completed_hashes(jsonl, task_name="init_response")
    assert done == {"deadbeef00000000", "cafebabe11111111"}


def test_load_completed_hashes_tolerates_partial_lines(
    tmp_path: Path, capfd: pytest.CaptureFixture
) -> None:
    """3 valid + 1 truncated trailing line: returns 3 hashes; does not raise."""
    from contextual_drag.inference.prompts import load_completed_hashes

    jsonl = tmp_path / "completions.jsonl"
    rows = [_row("init_response", f"hash{i:012x}") for i in range(3)]
    with open(jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        # Truncated trailing line (kill-mid-write simulation): NO newline,
        # JSON cut in half.
        f.write('{"id": "problem_x", "init_response_prompt": "trunc')

    done = load_completed_hashes(jsonl, task_name="init_response")
    assert done == {f"hash{i:012x}" for i in range(3)}

    captured = capfd.readouterr()
    # Warn line is informational, not a hard failure.
    assert "[warn]" in captured.out or "skipped" in captured.out, captured.out


def test_no_duplicate_hashes_after_resume_extension(tmp_path: Path) -> None:
    """Simulate a 4-row run: rows 1-2 pre-resumed, then driver appends 3-4.

    Re-scanning afterwards must return 4 *unique* hashes — no duplicates
    even though the file grew under us.
    """
    from contextual_drag.inference.prompts import load_completed_hashes

    jsonl = tmp_path / "completions.jsonl"
    initial = [_row("init_response", f"resumed_{i:08x}") for i in range(2)]
    with open(jsonl, "w") as f:
        for r in initial:
            f.write(json.dumps(r) + "\n")

    pre = load_completed_hashes(jsonl, task_name="init_response")
    assert len(pre) == 2

    # "Driver" appends two fresh rows.
    appended = [_row("init_response", f"fresh___{i:08x}") for i in range(2)]
    with open(jsonl, "a") as f:
        for r in appended:
            f.write(json.dumps(r) + "\n")

    post = load_completed_hashes(jsonl, task_name="init_response")
    assert len(post) == 4
    assert post == pre | {f"fresh___{i:08x}" for i in range(2)}
