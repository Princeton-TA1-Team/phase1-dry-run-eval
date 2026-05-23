"""Every CLI verb in the tree must answer ``--help`` cleanly.

Spawning a fresh subprocess is the point: we want to catch any verb that
accidentally triggers a heavy import (vllm/transformers) just to print its
help. The smoke test in ``test_smoke.py`` covers the root command; here we
parameterise over every concrete leaf verb.

Adding a new verb only requires updating ``VERBS`` below.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


# Full verb tree as agreed with team-lead (see brief: no registry, no eval-run).
VERBS: list[list[str]] = [
    # Root + groups
    [],
    ["inference"],
    ["eval"],
    ["data"],
    ["mitigation"],
    ["analysis"],
    ["analysis", "error_conditioning"],
    ["analysis", "ted"],
    ["analysis", "mitigation_buckets"],
    # inference leaves (underscored, matches __command__ in inference/cli.py)
    ["inference", "run"],
    ["inference", "dry_run"],
    ["inference", "list_models"],
    # eval leaves
    ["eval", "math"],
    ["eval", "crux"],
    # data leaves (hyphenated, matches __command__ in data/*_cli.py)
    ["data", "aggregate"],
    ["data", "aggregate-crux"],
    ["data", "initial-sampling-postprocess"],
    ["data", "minimal-aggregate-flatten"],
    ["data", "aggregate-iterative"],
    ["data", "stage1-postprocess-iterative"],
    # mitigation leaf
    ["mitigation", "run"],
    # analysis sub-subs
    ["analysis", "error_conditioning", "run"],
    ["analysis", "error_conditioning", "visualize"],
    ["analysis", "ted", "build_cache"],
    ["analysis", "ted", "summarize"],
    ["analysis", "ted", "render"],
    ["analysis", "mitigation_buckets", "run"],
    ["analysis", "mitigation_buckets", "render"],
]


def _verb_id(verb: list[str]) -> str:
    return " ".join(verb) or "<root>"


@pytest.mark.parametrize("verb", VERBS, ids=_verb_id)
def test_verb_help_runs_clean(verb: list[str]) -> None:
    """`--help` exits 0 and never leaks the stub message into stderr."""
    cmd = [sys.executable, "-m", "contextual_drag", *verb, "--help"]
    env = os.environ.copy()
    # Force CPU-only resolution; ban any accidental device touch.
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    assert result.returncode == 0, (
        f"`{' '.join(cmd)}` exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    # If `--help` triggered vllm import, our conftest stub would surface the
    # canonical message. Either no vllm import (preferred) or never *called*
    # the stub.
    assert "vllm import attempted" not in result.stderr, (
        f"--help triggered a vllm call: {result.stderr}"
    )
    assert "non-GPU test" not in result.stderr, (
        f"--help leaked the stub RuntimeError: {result.stderr}"
    )
