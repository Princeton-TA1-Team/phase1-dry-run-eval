from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from contextual_drag.config.paths import ExecutionMode, get_repo_root, resolve_execution_mode
from contextual_drag.config.resources import crux_dataset_resource_path, inference_model_config_resource


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cmd(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        env=merged_env,
        text=True,
        capture_output=True,
    )


def test_package_imports() -> None:
    import contextual_drag

    assert contextual_drag.__version__


def test_cli_help_runs() -> None:
    result = run_cmd("-m", "contextual_drag", "--help")
    assert result.returncode == 0, result.stderr
    assert "inference" in result.stdout
    assert "eval" in result.stdout


def test_group_help_runs_without_heavy_imports() -> None:
    code = """
import builtins
orig_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'vllm', 'transformers'}:
        raise AssertionError(f'unexpected import: {name}')
    return orig_import(name, *args, **kwargs)
builtins.__import__ = guarded
import contextual_drag.cli
raise SystemExit(contextual_drag.cli.main(['inference', '--help']))
"""
    result = run_cmd("-c", code)
    assert result.returncode == 0, result.stderr


def test_packaged_resources_load() -> None:
    configs = inference_model_config_resource()
    assert isinstance(configs, dict)
    assert "Qwen3_8B_Thinking" in configs
    assert crux_dataset_resource_path().exists()


def test_execution_mode_override() -> None:
    assert resolve_execution_mode("installed") == ExecutionMode.INSTALLED
    assert resolve_execution_mode("workspace") == ExecutionMode.WORKSPACE
    assert get_repo_root(execution_mode="workspace", start=REPO_ROOT) == REPO_ROOT
