from __future__ import annotations

import os
from enum import Enum
from importlib import resources
from pathlib import Path


class ExecutionMode(str, Enum):
    WORKSPACE = "workspace"
    INSTALLED = "installed"


# Primary env vars (AIQ-Contextual-Drag) and PR-2 fallbacks (CONTEXTUAL_DRAG_*)
# are read in that order so cards from PR-2 stay drop-in compatible.
_ENV_PAIRS = {
    "EXECUTION_MODE": ("AIQ_CDRAG_EXECUTION_MODE", "CONTEXTUAL_DRAG_EXECUTION_MODE"),
    "REPO_ROOT": ("AIQ_CDRAG_REPO_ROOT", "CONTEXTUAL_DRAG_REPO_ROOT"),
    "DATA_ROOT": ("AIQ_CDRAG_DATA_ROOT", "CONTEXTUAL_DRAG_DATA_ROOT"),
    "OUTPUT_ROOT": ("AIQ_CDRAG_OUTPUT_ROOT", "CONTEXTUAL_DRAG_OUTPUT_ROOT"),
    "PROMPT_TEMPLATE_ROOT": ("AIQ_CDRAG_PROMPT_TEMPLATE_ROOT", "CONTEXTUAL_DRAG_PROMPT_TEMPLATE_ROOT"),
    "SPLIT_ROOT": ("AIQ_CDRAG_SPLIT_ROOT", "CONTEXTUAL_DRAG_SPLIT_ROOT"),
    "RESULTS_ROOT": ("AIQ_CDRAG_RESULTS_ROOT",),
    "REGISTRY_PATH": ("AIQ_CDRAG_REGISTRY_PATH",),
    "CACHE_ROOT": ("AIQ_CDRAG_CACHE_ROOT",),
}

ENV_EXECUTION_MODE = _ENV_PAIRS["EXECUTION_MODE"][0]
ENV_REPO_ROOT = _ENV_PAIRS["REPO_ROOT"][0]
ENV_DATA_ROOT = _ENV_PAIRS["DATA_ROOT"][0]
ENV_OUTPUT_ROOT = _ENV_PAIRS["OUTPUT_ROOT"][0]
ENV_PROMPT_TEMPLATE_ROOT = _ENV_PAIRS["PROMPT_TEMPLATE_ROOT"][0]
ENV_SPLIT_ROOT = _ENV_PAIRS["SPLIT_ROOT"][0]
ENV_RESULTS_ROOT = _ENV_PAIRS["RESULTS_ROOT"][0]
ENV_REGISTRY_PATH = _ENV_PAIRS["REGISTRY_PATH"][0]
ENV_CACHE_ROOT = _ENV_PAIRS["CACHE_ROOT"][0]
ENV_TIKTOKEN_ENCODINGS_BASE = "TIKTOKEN_ENCODINGS_BASE"


def _read_env(key: str) -> str | None:
    for name in _ENV_PAIRS[key]:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _is_repo_root(path: Path) -> bool:
    markers = [
        path / "pyproject.toml",
        path / "src" / "contextual_drag",
        path / "cards",
        path / "submodules" / "aiq-magnet",
    ]
    return all(marker.exists() for marker in markers)


def find_repo_root(start: Path | None = None) -> Path | None:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if _is_repo_root(candidate):
            return candidate
    return None


def resolve_execution_mode(explicit: str | None = None, start: Path | None = None) -> ExecutionMode:
    if explicit:
        return ExecutionMode(explicit)
    env_value = _read_env("EXECUTION_MODE")
    if env_value:
        return ExecutionMode(env_value)
    return ExecutionMode.WORKSPACE if find_repo_root(start=start) else ExecutionMode.INSTALLED


def get_repo_root(execution_mode: str | None = None, start: Path | None = None, required: bool = False) -> Path | None:
    env_root = _read_env("REPO_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    repo_root = find_repo_root(start=start)
    mode = resolve_execution_mode(execution_mode, start=start)
    if repo_root is None and required and mode == ExecutionMode.WORKSPACE:
        raise RuntimeError("Workspace mode requires a checked-out repository root, but none was detected.")
    return repo_root


def _resolve_workspace_or_env(
    env_key: str,
    repo_relative: str,
    execution_mode: str | None = None,
    start: Path | None = None,
    required: bool = False,
    description: str = "path",
) -> Path | None:
    env_value = _read_env(env_key)
    if env_value:
        return Path(env_value).expanduser().resolve()

    mode = resolve_execution_mode(execution_mode, start=start)
    repo_root = get_repo_root(execution_mode=execution_mode, start=start)
    if mode == ExecutionMode.WORKSPACE and repo_root is not None:
        return (repo_root / repo_relative).resolve()

    if required:
        names = _ENV_PAIRS[env_key]
        raise RuntimeError(
            f"{description} is not available in installed mode unless you pass it explicitly "
            f"or set one of {names}."
        )
    return None


def get_data_root(execution_mode: str | None = None, start: Path | None = None, required: bool = False) -> Path | None:
    return _resolve_workspace_or_env(
        "DATA_ROOT", "data", execution_mode=execution_mode, start=start, required=required, description="data root"
    )


def get_output_root(execution_mode: str | None = None, start: Path | None = None, required: bool = False) -> Path | None:
    return _resolve_workspace_or_env(
        "OUTPUT_ROOT",
        "outputs",
        execution_mode=execution_mode,
        start=start,
        required=required,
        description="output root",
    )


def get_prompt_template_root(
    execution_mode: str | None = None, start: Path | None = None, required: bool = False
) -> Path | None:
    return _resolve_workspace_or_env(
        "PROMPT_TEMPLATE_ROOT",
        "prompt_templates",
        execution_mode=execution_mode,
        start=start,
        required=required,
        description="prompt template root",
    )


def get_default_split_root(
    execution_mode: str | None = None, start: Path | None = None, required: bool = False
) -> Path | None:
    return _resolve_workspace_or_env(
        "SPLIT_ROOT",
        "data/big_math_rl_verified/train_split/detailed_splits",
        execution_mode=execution_mode,
        start=start,
        required=required,
        description="split root",
    )


def get_results_root(execution_mode: str | None = None, start: Path | None = None, required: bool = False) -> Path | None:
    return _resolve_workspace_or_env(
        "RESULTS_ROOT",
        "evaluation_runs",
        execution_mode=execution_mode,
        start=start,
        required=required,
        description="results root",
    )


def get_registry_path(execution_mode: str | None = None, start: Path | None = None, required: bool = False) -> Path | None:
    return _resolve_workspace_or_env(
        "REGISTRY_PATH",
        "data_registry/registry.json",
        execution_mode=execution_mode,
        start=start,
        required=required,
        description="registry path",
    )


def get_cache_root(execution_mode: str | None = None, start: Path | None = None, required: bool = False) -> Path | None:
    return _resolve_workspace_or_env(
        "CACHE_ROOT",
        ".cache",
        execution_mode=execution_mode,
        start=start,
        required=required,
        description="cache root",
    )


def resolve_split_file(
    split: str,
    split_root: str | Path | None = None,
    execution_mode: str | None = None,
    start: Path | None = None,
) -> list[Path]:
    split_names = split.split("+")
    for item in split_names:
        if item not in {"sft", "rl", "val"}:
            raise ValueError("Data split must be one of sft, rl, val or a '+' combination of them.")

    if split_root is not None:
        root = Path(split_root).expanduser().resolve()
    else:
        root = get_default_split_root(execution_mode=execution_mode, start=start, required=True)
        assert root is not None

    return [root / f"{item}_ids.json" for item in split_names]


def get_tiktoken_encodings_base(
    execution_mode: str | None = None, start: Path | None = None, required: bool = False
) -> Path | None:
    env_value = os.environ.get(ENV_TIKTOKEN_ENCODINGS_BASE)
    if env_value:
        return Path(env_value).expanduser().resolve()

    mode = resolve_execution_mode(execution_mode, start=start)
    repo_root = get_repo_root(execution_mode=execution_mode, start=start)
    if mode == ExecutionMode.WORKSPACE and repo_root is not None:
        candidate = repo_root / "src" / "contextual_drag" / "resources" / "encodings"
        if candidate.exists():
            return candidate

    # Fall back to the packaged copy when shipped via importlib.resources.
    try:
        packaged = Path(str(resources.files("contextual_drag.resources").joinpath("encodings")))
        if packaged.exists():
            return packaged
    except (ModuleNotFoundError, FileNotFoundError):
        pass

    if required:
        raise RuntimeError(
            "TIKTOKEN encodings path is not available unless you ship "
            "src/contextual_drag/resources/encodings/ or set "
            f"{ENV_TIKTOKEN_ENCODINGS_BASE}."
        )
    return None
