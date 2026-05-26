"""Card-level contract tests.

For every ``cards/contextual_drag*.yaml`` we discover, assert:

  * the YAML parses with the schema keys magnet expects;
  * the inline ``claim.python`` snippet compiles;
  * every ``Name`` referenced in the claim is either a declared
    ``symbol``, a builtin/exception, or a key produced by the wrapper's
    ``results.json`` payload (inferred from the wrapper module's AST);
  * the wrapper node's ``--help`` runs cleanly in a subprocess (no
    vllm/GPU touch);
  * every ``subprocess.run`` inside the wrapper invokes the package CLI
    (``[python, '-m', 'contextual_drag', ...]``) — locking in the
    "single-process magnet node, chain via our CLI" contract without
    actually firing the engine.

Cards are discovered at collection time, so adding a new
``contextual_drag_<x>.yaml`` automatically picks up coverage.
"""
from __future__ import annotations

import ast
import builtins
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CARDS_DIR = REPO_ROOT / "cards"
NODES_DIR = CARDS_DIR / "nodes"

REQUIRED_YAML_KEYS = {"title", "description", "version", "claim", "pipeline", "symbols"}

# Names the claim is always allowed to reference (Python keywords / common
# exception types / typing-time builtins).
_BUILTIN_WHITELIST = (
    set(dir(builtins))
    | {"True", "False", "None"}
    | {"Exception", "ValueError", "RuntimeError", "AssertionError", "TypeError",
       "KeyError", "FileNotFoundError"}
)


def _discover_card_paths() -> list[Path]:
    if not CARDS_DIR.exists():
        return []
    return sorted(p for p in CARDS_DIR.glob("contextual_drag*.yaml") if p.is_file())


_CARD_PATHS = _discover_card_paths()
_CARD_IDS = [p.stem for p in _CARD_PATHS]


def _load_card(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _node_module_from_pipeline(card: dict) -> str | None:
    """Extract ``cards.nodes.<module>`` from the first pipeline stage's executable."""
    pipeline = card.get("pipeline") or {}
    for stage in pipeline.values():
        exe = (stage or {}).get("executable", "")
        match = re.search(r"-m\s+(cards\.nodes\.[\w_.]+)", exe)
        if match:
            return match.group(1)
    return None


def _wrapper_result_keys(node_module: str) -> set[str]:
    """Statically scan the wrapper module for keys produced under ``result``.

    Looks for either:

      * ``json.dump({"result": {...keys...}}, ...)`` literals, or
      * the dict literal returned by a ``_write_result`` helper (we just walk
        every dict literal in the file and union all string keys).

    Liberal on purpose: false positives only widen the whitelist, never
    catch a typo we don't have.
    """
    src_path = NODES_DIR / (node_module.split(".")[-1] + ".py")
    if not src_path.exists():
        return set()
    tree = ast.parse(src_path.read_text())
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
        elif isinstance(node, ast.Call):
            # _write_result(... key=value, ...) kwargs are surfaced too.
            for kw in node.keywords:
                if kw.arg:
                    keys.add(kw.arg)
    return keys


# ---------------------------------------------------------------------------
# Parameterized contract tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("card_path", _CARD_PATHS, ids=_CARD_IDS)
def test_card_yaml_parses(card_path: Path) -> None:
    card = _load_card(card_path)
    assert isinstance(card, dict), f"{card_path.name} did not parse as a mapping"
    missing = REQUIRED_YAML_KEYS - set(card)
    assert not missing, f"{card_path.name} missing required keys: {sorted(missing)}"


@pytest.mark.parametrize("card_path", _CARD_PATHS, ids=_CARD_IDS)
def test_card_claim_python_compiles(card_path: Path) -> None:
    card = _load_card(card_path)
    claim = card["claim"]["python"]
    compile(claim, f"<{card_path.name}::claim>", "exec")


@pytest.mark.parametrize("card_path", _CARD_PATHS, ids=_CARD_IDS)
def test_card_claim_symbols_are_declared(card_path: Path) -> None:
    card = _load_card(card_path)
    symbols = set((card.get("symbols") or {}).keys())
    node_module = _node_module_from_pipeline(card)
    wrapper_keys = _wrapper_result_keys(node_module) if node_module else set()
    allowed = symbols | wrapper_keys | _BUILTIN_WHITELIST

    tree = ast.parse(card["claim"]["python"])
    referenced = {
        n.id for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    undeclared = referenced - allowed
    assert not undeclared, (
        f"{card_path.name}: claim.python references undeclared names "
        f"{sorted(undeclared)}. Allowed = symbols ∪ wrapper_result_keys ∪ builtins. "
        f"symbols={sorted(symbols)}, wrapper_keys={sorted(wrapper_keys)}"
    )


def _wrapper_path(node_module: str) -> Path:
    return NODES_DIR / (node_module.split(".")[-1] + ".py")


@pytest.mark.parametrize("card_path", _CARD_PATHS, ids=_CARD_IDS)
def test_card_node_help_runs(card_path: Path) -> None:
    card = _load_card(card_path)
    node_module = _node_module_from_pipeline(card)
    if not node_module:
        pytest.skip(f"{card_path.name}: no `python -m cards.nodes.<x>` executable")
    if not _wrapper_path(node_module).exists():
        pytest.skip(
            f"{card_path.name}: wrapper {node_module} not yet written "
            f"(Wave 3a deliverable)"
        )
    result = subprocess.run(
        [sys.executable, "-m", node_module, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{node_module} --help exited {result.returncode}: {result.stderr}"
    )


@pytest.mark.parametrize("card_path", _CARD_PATHS, ids=_CARD_IDS)
def test_card_node_subprocess_chain_uses_package_cli(card_path: Path) -> None:
    """Lock in the "always chain through `python -m contextual_drag`" wiring.

    Static AST scan of the wrapper: every ``subprocess.run`` first arg must
    be a list whose first three elements are
    ``[sys.executable, "-m", "contextual_drag"]``. This is the contract
    that lets magnet treat the node as a single process while everything
    inside still goes through our own CLI.
    """
    card = _load_card(card_path)
    node_module = _node_module_from_pipeline(card)
    if not node_module:
        pytest.skip(f"{card_path.name}: no wrapper module")
    src_path = _wrapper_path(node_module)
    if not src_path.exists():
        pytest.skip(
            f"{card_path.name}: wrapper {node_module} not yet written "
            f"(Wave 3a deliverable)"
        )
    tree = ast.parse(src_path.read_text())

    # Find every subprocess.run(<first_arg>, ...) call.
    run_calls = [
        node for node in ast.walk(tree)
        if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
            and node.args)
    ]
    assert run_calls, f"{node_module}: no subprocess.run(...) calls found"

    # Pre-compute every Name -> rhs assignment found *anywhere* in the file
    # (module, function, method scopes). The wrappers bind helpers like
    # `cdrag = [sys.executable, "-m", "contextual_drag"]` or
    # `inf_cmd = [sys.executable, "-m", "contextual_drag", "inference", "run", ...]`
    # inside methods, so we have to walk every scope.
    name_to_rhs = _collect_assignments(tree)

    for call in run_calls:
        first = call.args[0]
        head_consts = _list_head_constants(first, name_to_rhs, depth=6)
        joined = "/".join(map(str, head_consts))
        assert "-m" in head_consts and "contextual_drag" in head_consts, (
            f"{node_module}: subprocess.run call does not chain through the "
            f"contextual_drag CLI. First-arg constants seen: {head_consts!r} "
            f"({joined})"
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _list_head_constants(
    node: ast.AST,
    name_to_rhs: dict[str, ast.AST],
    depth: int = 6,
    seen: frozenset[str] = frozenset(),
) -> list:
    """Walk a list-shaped expression and pull out its leading constants.

    Understands:

      * ``[a, b, ...]``  — literal list
      * ``a + b``        — concatenation (recurse both sides)
      * ``my_var``       — substitute the previously-collected RHS for
                           that name, with a `seen` set to short-circuit
                           cycles.
    """
    if isinstance(node, ast.List):
        out: list = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant):
                out.append(elt.value)
            # Non-constants (sys.executable, str(cfg.x), ...) are skipped, not
            # treated as a stop signal: a wrapper's first element is typically
            # `sys.executable`, but `"-m", "contextual_drag"` follow as
            # constants we still want to capture.
            if len(out) >= depth:
                break
        return out
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _list_head_constants(node.left, name_to_rhs, depth, seen)
        right = _list_head_constants(node.right, name_to_rhs, depth, seen)
        return (left + right)[:depth]
    if isinstance(node, ast.Name) and node.id not in seen and node.id in name_to_rhs:
        return _list_head_constants(
            name_to_rhs[node.id], name_to_rhs, depth, seen | {node.id}
        )
    return []


def _collect_assignments(tree: ast.AST) -> dict[str, ast.AST]:
    """Map ``name -> RHS`` for every plain ``name = expr`` assignment in *tree*.

    Walks every scope, so a `cdrag = [...]` defined inside `main()` is
    visible. If a name is reassigned, last write wins (good enough for
    the small wrapper layer).
    """
    out: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    out[tgt.id] = node.value
    return out
