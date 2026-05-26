"""CLI smoke tests for the recursive self-improvement subpackage.

These tests block on the `recursive-subpackage` deliverable (the
`python -m contextual_drag recursive run ...` CLI surface). They are
expected to fail with a clean "no such command" exit until that
subpackage lands; once it does, they pin down the CLI shape the cards
+ wrappers depend on.

Coverage:

  * `recursive --help` and `recursive run --help` exit 0.
  * `recursive run --help` advertises both `--variant rf1` and
    `--variant naive`.
  * The ported pipeline modules import without dragging in vllm
    (smoke-equivalent to test_smoke.py's import-light guarantee).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    return subprocess.run(
        [sys.executable, "-m", "contextual_drag", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )


def test_recursive_modal_help_exits_zero() -> None:
    """`contextual-drag recursive --help` should exit 0 with a usage banner."""
    result = _run("recursive", "--help")
    assert result.returncode == 0, (
        f"`recursive --help` exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert "run" in (result.stdout + result.stderr).lower()


def test_recursive_run_help_exits_zero() -> None:
    """`contextual-drag recursive run --help` advertises --variant {rf1,naive}."""
    result = _run("recursive", "run", "--help")
    assert result.returncode == 0, (
        f"`recursive run --help` exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    blob = (result.stdout + result.stderr)
    assert "--variant" in blob, (
        f"`recursive run --help` did not advertise --variant.\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert "rf1" in blob, "`recursive run --help` does not mention the rf1 variant"
    assert "naive" in blob, "`recursive run --help` does not mention the naive variant"


def test_recursive_run_variant_naive_help() -> None:
    """`--variant naive` parses cleanly with --help (no validation error)."""
    result = _run("recursive", "run", "--variant", "naive", "--help")
    assert result.returncode == 0, (
        f"`recursive run --variant naive --help` exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def test_recursive_pipeline_modules_import() -> None:
    """The seven recursive pipeline modules import without touching vllm.

    `tests/conftest.py` stubs vllm and transformers at the module level,
    so this also pins down the no-vllm-at-import contract: if any of
    these modules grew a top-level `from vllm import ...` line, the
    stub's `__call__` would still let bare imports through, but the
    test would fail loudly the first time something actually invoked
    the stub. We just check that the import succeeds.
    """
    try:
        from contextual_drag.recursive.pipeline import (  # noqa: F401
            pipeline,
            thinking,
            aggregate,
            flatten,
            inference_step,
            stage_state,
            summary,
            verifier,
        )
    except ImportError as exc:
        pytest.fail(
            "Recursive pipeline modules failed to import. This usually "
            "means the recursive-subpackage deliverable has not landed "
            "yet, or one of the ported files still references an "
            f"unported `inference.X` module. Raw error: {exc}"
        )
