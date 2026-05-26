"""Packaged-resource accessors must keep working after refactors.

Each accessor reaches into ``importlib.resources``; the failure mode if
``package-data`` is misconfigured in ``pyproject.toml`` is a silent
``FileNotFoundError`` at runtime. Pin that down here.
"""
from __future__ import annotations


def test_eval_models_params_loads() -> None:
    from contextual_drag.config.resources import inference_model_config_resource

    cfg = inference_model_config_resource()
    assert isinstance(cfg, dict)
    assert "Qwen3_8B_Thinking" in cfg or "Qwen3_8B_NoThinking" in cfg, (
        f"expected at least one Qwen3 alias in packaged config, got {sorted(cfg)[:5]}..."
    )


def test_crux_dataset_loads() -> None:
    from contextual_drag.config.resources import crux_dataset_resource_path

    path = crux_dataset_resource_path()
    assert path.exists(), f"packaged crux dataset missing at {path}"
    # File must be readable as JSONL (cheap structural check).
    first = path.read_text().splitlines()[0]
    assert first.startswith("{"), f"first line of crux dataset isn't JSON: {first[:80]!r}"


def test_tiktoken_encodings_path_resolves() -> None:
    from contextual_drag.config.paths import get_tiktoken_encodings_base

    p = get_tiktoken_encodings_base()
    # In CI there's no checked-in encodings directory: None is acceptable.
    # If present, it must point to a real directory.
    assert p is None or p.exists(), f"encodings base set to a non-existent path: {p}"
