"""Unit tests for cards/nodes/_recursive_init_sampling.py.

Three things to verify without GPU / subprocess execution:

1. The per-task template-key dispatch is correct for the 8 datasets the
   recursive cards target.
2. The pass-through path: when `data_path` already has the
   `init_response_generations_metadata` column, `ensure_init_responses`
   returns the input path unchanged and `init_sampling_skipped=True`,
   with no subprocess.run invoked.
3. The resume-sentinel path: when a complete
   `<intermediate_dir>/init_sampling/processed_flattened_init_responses.ds/`
   already exists, the helper returns it with `init_sampling_skipped=False`
   and runs no subprocess.

The "actually invoke inference run + postprocess" branch is not unit-
testable without GPU — it's covered by the end-to-end GPU smoke described
in the README.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
CARDS_NODES_DIR = REPO_ROOT / "cards" / "nodes"


# Allow `from cards.nodes._recursive_init_sampling import ...` regardless of
# where pytest is invoked from.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_template_key_dispatch() -> None:
    from cards.nodes._recursive_init_sampling import template_key_for_task

    # Per-task overrides
    assert template_key_for_task("crux-i") == "question_only_prompt"
    assert template_key_for_task("crux-o") == "question_only_prompt"
    assert template_key_for_task("gpqa") == "qa_mc_prompt"
    assert template_key_for_task("mmlu") == "qa_mc_prompt"

    # Math-style default
    for t in ("aime24", "aime25", "hmmt24", "hmmt25", "24-game",
              "totally-new-task-name"):
        assert template_key_for_task(t) == "qwen_math_prompt"


def test_eval_subgroup_dispatch() -> None:
    """Picks the right `contextual-drag eval` verb to flatten init-sampling output."""
    from cards.nodes._recursive_init_sampling import eval_subgroup_for_task

    # Per-task overrides
    assert eval_subgroup_for_task("crux-i") == "crux"
    assert eval_subgroup_for_task("crux-o") == "crux"
    assert eval_subgroup_for_task("24-game") == "game_of_24"

    # Math default — covers aime/hmmt and the QA-MC tasks (which fall through
    # to math since there's no dedicated eval verb for gpqa/mmlu).
    for t in ("aime24", "aime25", "hmmt24", "hmmt25", "gpqa", "mmlu",
              "totally-new-task-name"):
        assert eval_subgroup_for_task(t) == "math"


def _make_fake_ds(tmp_path: Path, with_init_response_col: bool) -> Path:
    """Create a tiny HF dataset on disk so the helper's `load_from_disk`
    + column-name check has something real to read.

    `with_init_response_col=True` includes the marker column the helper
    looks for — pass-through case. `False` mimics a raw benchmark .ds.
    """
    from datasets import Dataset

    row: dict[str, Any] = {"id": "p1", "problem": "what is 1+1?", "answer": "2"}
    if with_init_response_col:
        row["init_response_generations_metadata"] = {
            "model_config_alias": "GPT_OSS_20B"
        }
    ds = Dataset.from_list([row])
    ds_dir = tmp_path / "fake.ds"
    ds.save_to_disk(str(ds_dir))
    return ds_dir


def test_passthrough_when_input_already_has_init_response_column(
    tmp_path: Path,
) -> None:
    """If the .ds at data_path already has init_response_generations_metadata,
    the helper must return (data_path, True) without spawning a subprocess.
    """
    from cards.nodes._recursive_init_sampling import ensure_init_responses

    ds_path = _make_fake_ds(tmp_path, with_init_response_col=True)
    intermediate = tmp_path / "work"

    with mock.patch(
        "cards.nodes._recursive_init_sampling.subprocess.run"
    ) as run_mock:
        out_path, skipped = ensure_init_responses(
            data_path=ds_path,
            intermediate_dir=intermediate,
            init_alias="GPT_OSS_20B",
            task="aime24",
            init_template_path=Path("prompt_templates/init_response_prompt_templates.json"),
            init_n_samples=8,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
        )

    assert run_mock.call_count == 0, (
        "Expected no subprocess.run when data_path already has init-response "
        f"columns; got {run_mock.call_args_list}"
    )
    assert out_path == ds_path.resolve()
    assert skipped is True


def test_resume_when_postprocess_output_already_present(tmp_path: Path) -> None:
    """If <intermediate>/init_sampling/processed_flattened_init_responses.ds/
    already exists with state.json + dataset_info.json, the helper must
    return that path with skipped=False (still ran, just not THIS run) and
    NOT invoke subprocess.
    """
    from cards.nodes._recursive_init_sampling import ensure_init_responses

    # Raw .ds (no init_response col -> would normally trigger Stage -1)
    ds_path = _make_fake_ds(tmp_path, with_init_response_col=False)
    intermediate = tmp_path / "work"

    # Pre-populate the sentinel.
    sentinel_dir = (intermediate / "init_sampling"
                    / "processed_flattened_init_responses.ds")
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    (sentinel_dir / "state.json").write_text(
        json.dumps({"_fingerprint": "test", "_split": None}))
    (sentinel_dir / "dataset_info.json").write_text("{}")

    with mock.patch(
        "cards.nodes._recursive_init_sampling.subprocess.run"
    ) as run_mock:
        out_path, skipped = ensure_init_responses(
            data_path=ds_path,
            intermediate_dir=intermediate,
            init_alias="GPT_OSS_20B",
            task="aime24",
            init_template_path=Path("prompt_templates/init_response_prompt_templates.json"),
            init_n_samples=8,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
        )

    assert run_mock.call_count == 0, (
        "Expected no subprocess.run when the postprocess output is already "
        f"complete on disk; got {run_mock.call_args_list}"
    )
    assert out_path == sentinel_dir.resolve()
    assert skipped is False


def test_helper_is_importable_from_wrappers() -> None:
    """Sanity: both wrappers import the helper via the same path
    (`from cards.nodes._recursive_init_sampling import ensure_init_responses`).
    Catches accidental renames.
    """
    rf1_src = (CARDS_NODES_DIR / "run_recursive_filter1.py").read_text()
    naive_src = (CARDS_NODES_DIR / "run_recursive_naive.py").read_text()
    expected = "from cards.nodes._recursive_init_sampling import ensure_init_responses"
    assert expected in rf1_src, "rf1 wrapper does not import ensure_init_responses"
    assert expected in naive_src, "naive wrapper does not import ensure_init_responses"
