"""Unit tests for cards/render_formal.py.

Render the full formal set for a sample model into a tmp dir (with a small
--max_questions so no dataset is loaded from disk) and assert:
  * the expected <test>/<dataset>.yaml files exist (incl. both EC variants);
  * every rendered card passes the same structural contract as the smoke cards
    (parses, claim compiles, claim Names ⊆ symbols ∪ wrapper-result-keys ∪ builtins);
  * the dataset-family params landed (crux/qa_mc template keys, EC regimes).
No GPU / subprocess.
"""
from __future__ import annotations

import ast
import builtins
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NODES_DIR = REPO_ROOT / "cards" / "nodes"
REQUIRED_YAML_KEYS = {"title", "description", "version", "claim", "pipeline", "symbols"}
_BUILTIN_WHITELIST = (
    set(dir(builtins))
    | {"True", "False", "None"}
    | {"Exception", "ValueError", "RuntimeError", "AssertionError", "TypeError",
       "KeyError", "FileNotFoundError"}
)


def _node_module(card: dict):
    for stage in (card.get("pipeline") or {}).values():
        m = re.search(r"-m\s+(cards\.nodes\.[\w_.]+)", (stage or {}).get("executable", ""))
        if m:
            return m.group(1)
    return None


def _wrapper_result_keys(node_module: str):
    src = NODES_DIR / (node_module.split(".")[-1] + ".py")
    if not src.exists():
        return set()
    keys = set()
    for node in ast.walk(ast.parse(src.read_text())):
        if isinstance(node, ast.Dict):
            keys |= {k.value for k in node.keys
                     if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        elif isinstance(node, ast.Call):
            keys |= {kw.arg for kw in node.keywords if kw.arg}
    return keys


def _algo(card: dict) -> dict:
    return card["pipeline"][next(iter(card["pipeline"]))]["algo_params"]


@pytest.fixture(scope="module")
def rendered(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("formal")
    from cards.render_formal import RenderFormalCLI

    RenderFormalCLI.main(argv=[
        "--model", "Qwen3_8B_NoThinking", "--out_dir", str(out),
        "--max_questions", "16",
    ])
    return out / "Qwen3_8B_NoThinking"


def test_expected_cards_exist(rendered: Path) -> None:
    from cards.render_formal import TESTS

    got = {str(p.relative_to(rendered)) for p in rendered.rglob("*.yaml")}
    assert len(got) == sum(len(s.datasets) for s in TESTS.values())  # 8*4 + 1 = 33
    for expected in ("drag/crux-i.yaml", "drag/24-game.yaml",
                     "drag-1f/crux-i.yaml", "drag-1f/gpqa.yaml",
                     "error-conditioning-posthoc/gpqa.yaml",
                     "error-conditioning-external/gpqa.yaml",
                     "mitigation/crux-i.yaml", "ted/24-game.yaml"):
        assert expected in got, expected


def test_rendered_cards_pass_contract(rendered: Path) -> None:
    for p in sorted(rendered.rglob("*.yaml")):
        card = yaml.safe_load(p.read_text())
        assert isinstance(card, dict), p
        assert not (REQUIRED_YAML_KEYS - set(card)), (p, REQUIRED_YAML_KEYS - set(card))
        claim = card["claim"]["python"]
        compile(claim, f"<{p.name}::claim>", "exec")
        nm = _node_module(card)
        assert nm is not None, f"{p}: no `cards.nodes.<module>` executable"
        allowed = set(card.get("symbols") or {}) | _wrapper_result_keys(nm) | _BUILTIN_WHITELIST
        refs = {n.id for n in ast.walk(ast.parse(claim))
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        assert not (refs - allowed), (str(p), sorted(refs - allowed))


def test_family_params_resolved(rendered: Path) -> None:
    def ap(rel: str) -> dict:
        return _algo(yaml.safe_load((rendered / rel).read_text()))

    # crux family on drag
    d = ap("drag/crux-i.yaml")
    assert d["dataset"] == "crux-i"
    assert d["init_template_key"] == "question_only_prompt"
    assert d["twof_template_key"] == "2f_crux_input"
    assert d["data_path"] == "data/full_data/crux-i/crux-i.ds"

    # drag-1f resolves the 1f conditioning template family (GPU producer)
    o = ap("drag-1f/gpqa.yaml")
    assert o["dataset"] == "gpqa"
    assert o["cond_template_key_1f"] == "1f_qa_mc"
    assert o["init_template_key"] == "qa_mc_prompt"

    # EC posthoc is now a PURE-ANALYSIS card: no regime/templates/sizing — just the
    # cache key. It reuses drag-1f's conditioned-inference cache.
    p = ap("error-conditioning-posthoc/gpqa.yaml")
    assert p["dataset"] == "gpqa"
    assert "cond_cache_root" in p
    assert "regime" not in p and "n" not in p and "max_tokens" not in p

    # EC external = framing + qa_mc family
    e = ap("error-conditioning-external/gpqa.yaml")
    assert e["regime"] == "framing"
    assert e["framing_template_key"] == "framing_qa_mc"

    # mitigation onef family
    m = ap("mitigation/crux-i.yaml")
    assert m["onef_template_key"] == "1f_crux_input"
