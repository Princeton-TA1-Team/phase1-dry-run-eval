"""Unit tests for cards/nodes/_dataset_registry.py — the dataset→family
dispatch shared by the formal-card renderer and the drag/EC/mitigation nodes.
No GPU / subprocess; pure mapping checks plus presence of the ported framing
templates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_MATH = ("aime24", "aime25", "hmmt24", "hmmt25")


def test_eval_verb_dispatch() -> None:
    from cards.nodes._dataset_registry import eval_verb_for

    assert eval_verb_for("crux-i") == "crux"
    assert eval_verb_for("24-game") == "game_of_24"
    for ds in _MATH + ("gpqa", "mmlu"):
        assert eval_verb_for(ds) == "math", ds


def test_aggregate_command_dispatch() -> None:
    from cards.nodes._dataset_registry import aggregate_command_for

    assert aggregate_command_for("crux-i") == "aggregate-crux"
    for ds in _MATH + ("gpqa", "mmlu", "24-game"):
        assert aggregate_command_for(ds) == "aggregate", ds


def test_template_key_dispatch() -> None:
    from cards.nodes._dataset_registry import (
        framing_template_key_for,
        init_template_key_for,
        onef_template_key_for,
        twof_template_key_for,
    )

    assert init_template_key_for("crux-i") == "question_only_prompt"
    assert init_template_key_for("gpqa") == "qa_mc_prompt"
    assert init_template_key_for("aime24") == "qwen_math_prompt"

    assert twof_template_key_for("gpqa") == "2f_qa_mc"
    assert twof_template_key_for("crux-i") == "2f_crux_input"
    assert twof_template_key_for("24-game") == "2f"

    assert onef_template_key_for("mmlu") == "1f_qa_mc"
    assert onef_template_key_for("crux-i") == "1f_crux_input"
    assert onef_template_key_for("aime24") == "1f"

    assert framing_template_key_for("gpqa") == "framing_qa_mc"
    assert framing_template_key_for("crux-i") == "framing_crux_input"
    assert framing_template_key_for("aime24") == "framing"


def test_data_path() -> None:
    from cards.nodes._dataset_registry import data_path_for

    assert data_path_for("mmlu") == "data/full_data/mmlu/mmlu.ds"
    assert data_path_for("crux-i") == "data/full_data/crux-i/crux-i.ds"


def test_framing_templates_were_ported() -> None:
    d = json.loads((REPO_ROOT / "prompt_templates" / "ablation_templates.json").read_text())
    for k in ("framing", "framing_qa_mc", "framing_crux_input"):
        assert k in d, f"missing framing template {k!r}"
        assert "{arg_problem}" in d[k] and "{arg_traj1}" in d[k]
        # the framing prompt must explicitly flag the draft as incorrect
        assert "incorrect" in d[k].lower()
