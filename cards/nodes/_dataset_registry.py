"""Dataset → family dispatch shared by the formal-card renderer
(`cards/render_formal.py`) and the drag / error-conditioning / mitigation card
nodes.

Each benchmark belongs to one of three families — math (aime24/25, hmmt24/25,
24-game), qa_mc (gpqa, mmlu), or crux (crux-i) — which determines the prompt
template suffix, the `contextual-drag eval` verb, and the aggregate command.
This mirrors the upstream mapping (`evals/{1f,2f}.sh`, `inference/work_items.py`,
and the mitigation pipeline's `_solve_template_key`).

The two base dispatchers (`template_key_for_task`, `eval_subgroup_for_task`)
already live in `_recursive_init_sampling.py`; we import and reuse them so the
init-template and eval-verb logic has a single source of truth.
"""
from __future__ import annotations

from cards.nodes._recursive_init_sampling import (
    eval_subgroup_for_task,
    template_key_for_task,
)

# Multiple-choice family (qa_mc prompt/template suffix).
_QA_MC = ("gpqa", "mmlu")
# Crux (code-reasoning) family (crux_input suffix, crux eval, crux aggregate).
_CRUX = ("crux-i", "crux-o")


def data_path_for(dataset: str) -> str:
    """Workspace-relative full-data path for a benchmark."""
    return f"data/full_data/{dataset}/{dataset}.ds"


def init_template_key_for(dataset: str) -> str:
    """Clean/direct-round init template key.

    crux-i → `question_only_prompt`, gpqa/mmlu → `qa_mc_prompt`, else
    `qwen_math_prompt` (delegates to the recursive node's dispatch).
    """
    return template_key_for_task(dataset)


def _family_suffix(dataset: str) -> str:
    if dataset in _CRUX:
        return "_crux_input"
    if dataset in _QA_MC:
        return "_qa_mc"
    return ""


def twof_template_key_for(dataset: str) -> str:
    """2F conditioning template key: `2f` / `2f_qa_mc` / `2f_crux_input`."""
    return "2f" + _family_suffix(dataset)


def onef_template_key_for(dataset: str) -> str:
    """1F conditioning template key: `1f` / `1f_qa_mc` / `1f_crux_input`."""
    return "1f" + _family_suffix(dataset)


def framing_template_key_for(dataset: str) -> str:
    """Framing (external error-conditioning) template key:
    `framing` / `framing_qa_mc` / `framing_crux_input`."""
    return "framing" + _family_suffix(dataset)


def eval_verb_for(dataset: str) -> str:
    """`contextual-drag eval` verb: `crux` / `game_of_24` / `math`."""
    return eval_subgroup_for_task(dataset)


def aggregate_command_for(dataset: str) -> str:
    """`data aggregate-crux` for crux datasets, else `data aggregate`."""
    return "aggregate-crux" if dataset in _CRUX else "aggregate"
