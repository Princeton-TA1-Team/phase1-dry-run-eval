"""Unit tests for cards/nodes/_init_cache.py — the shared init-sampling cache.

Pure logic only (keying, manifest, reuse/fallback/generate decision); no GPU,
no subprocess, no vLLM.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cards.nodes._init_cache import (  # noqa: E402
    MANIFEST_NAME,
    PROCESSED_DS_NAME,
    build_manifest,
    cache_action,
    dataset_key,
    shared_init_dir,
)

_WANTED = dict(
    model_config="Qwen3_8B_NoThinking",
    data_path="data/full_data/aime24/aime24.ds",
    init_template_path="prompt_templates/init_response_prompt_templates.json",
    init_template_key="qwen_math_prompt",
    max_questions=30, n=16, max_tokens=8192,
)


def test_dataset_key_prefers_dataset_then_stem():
    assert dataset_key("aime24", "data/full_data/aime24/aime24.ds") == "aime24"
    assert dataset_key("", "data/full_data/aime24/aime24.ds") == "aime24"
    assert dataset_key(None, "data/smoke/gpqa/gpqa.ds") == "gpqa"


def test_shared_dir_keyed_by_model_and_dataset():
    d = shared_init_dir("runs/init_cache", "Nemotron_7B", "crux-i", "x/crux-i.ds")
    assert d == Path("runs/init_cache/Nemotron_7B/crux-i")


def test_manifest_is_order_independent_and_typed():
    m = build_manifest(**_WANTED)
    assert m["max_questions"] == 30 and isinstance(m["n"], int)
    # same content, different insertion order -> equal dict (used for ==)
    assert m == build_manifest(**{k: _WANTED[k] for k in reversed(list(_WANTED))})


def _seed_cache(base: Path, manifest: dict | None):
    ds = base / PROCESSED_DS_NAME
    (ds).mkdir(parents=True, exist_ok=True)
    (ds / "dataset_info.json").write_text("{}")
    if manifest is not None:
        (base / MANIFEST_NAME).write_text(json.dumps(manifest))
    return ds, base / MANIFEST_NAME


def test_action_generate_when_absent(tmp_path):
    ds = tmp_path / PROCESSED_DS_NAME
    assert cache_action(ds, tmp_path / MANIFEST_NAME, build_manifest(**_WANTED)) == "generate"


def test_action_reuse_on_match(tmp_path):
    want = build_manifest(**_WANTED)
    ds, man = _seed_cache(tmp_path, want)
    assert cache_action(ds, man, want) == "reuse"


def test_action_fallback_on_mismatch(tmp_path):
    ds, man = _seed_cache(tmp_path, build_manifest(**{**_WANTED, "n": 8}))
    assert cache_action(ds, man, build_manifest(**_WANTED)) == "fallback"


def test_action_fallback_when_ds_present_without_manifest(tmp_path):
    ds, _ = _seed_cache(tmp_path, None)
    assert cache_action(ds, tmp_path / MANIFEST_NAME, build_manifest(**_WANTED)) == "fallback"
