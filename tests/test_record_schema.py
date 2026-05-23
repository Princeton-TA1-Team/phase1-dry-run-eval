"""Pin the ``<task_name>_generations_metadata`` JSONL record shape.

Downstream postprocess + aggregate code reads these keys by name. If the
record builder in ``inference/run_model.py::build_record`` drops one of
them, downstream silently regresses. A grep-level guard keeps the
contract visible.

We deliberately don't *import-and-call* ``build_record`` — that would
require setting up a fake vllm RequestOutput + tokenizer. A source-level
check is cheap and catches the failure mode we actually care about
(someone deletes a key).
"""
from __future__ import annotations

import inspect


EXPECTED_METADATA_KEYS = {
    "prompt_hash",
    "n_samples_requested",
    "n_samples_variant_default",
    "num_responses",
    "template_path",
    "template_key",
    "model_config_alias",
}


def test_metadata_schema_keys_present_in_build_record() -> None:
    from contextual_drag.inference import run_model

    src = inspect.getsource(run_model)
    missing = []
    for key in EXPECTED_METADATA_KEYS:
        if f'"{key}"' not in src and f"'{key}'" not in src:
            missing.append(key)
    assert not missing, (
        f"build_record in inference/run_model.py no longer emits expected "
        f"metadata keys: {sorted(missing)}. Downstream eval/aggregate code "
        f"reads these by name; update consumers before removing."
    )


def test_metadata_block_named_generations_metadata() -> None:
    """The metadata sub-dict's key is ``<task_name>_generations_metadata``."""
    from contextual_drag.inference import run_model

    src = inspect.getsource(run_model)
    assert '_generations_metadata' in src, (
        "The contract-defining suffix `_generations_metadata` is missing "
        "from run_model.py; downstream resume + aggregate will break."
    )
