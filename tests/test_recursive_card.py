"""Per-card contract tests for the recursive self-improvement cards.

Parameterised over the two YAMLs:

  * cards/contextual_drag_recursive_filter1.yaml  (rf1, improvement claim)
  * cards/contextual_drag_recursive_naive.yaml    (naive, self-deterioration claim)

For each card we assert:

  1. The YAML parses with the magnet-required top-level keys.
  2. The inline `claim.python` snippet compiles.
  3. Every `Name` referenced by the claim is either a declared
     `symbols.<name>`, a key produced by the wrapper's `results.json`
     payload (inferred via AST scan of the wrapper), or a builtin /
     exception name.
  4. `python cards/nodes/run_recursive_{rf1|naive}.py --help` exits 0
     (no vllm touch).
  5. The wrapper's subprocess chain wires through
     `python -m contextual_drag recursive run --variant {variant}`.

These card-side tests do NOT block on the `recursive-subpackage` CLI
landing — they exercise the YAML + wrapper files, not the package
CLI. They should pass standalone.
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

_BUILTIN_WHITELIST = (
    set(dir(builtins))
    | {"True", "False", "None"}
    | {"Exception", "ValueError", "RuntimeError", "AssertionError", "TypeError",
       "KeyError", "FileNotFoundError"}
)


# (yaml_filename, expected --variant in subprocess chain, expected delta symbol)
_CARDS = [
    ("contextual_drag_recursive_filter1.yaml", "rf1",   "delta_acc_rf1"),
    ("contextual_drag_recursive_naive.yaml",   "naive", "delta_acc_naive"),
]


def _card_path(name: str) -> Path:
    return CARDS_DIR / name


def _load_card(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _node_module_from_pipeline(card: dict) -> str | None:
    pipeline = card.get("pipeline") or {}
    for stage in pipeline.values():
        exe = (stage or {}).get("executable", "")
        match = re.search(r"-m\s+(cards\.nodes\.[\w_.]+)", exe)
        if match:
            return match.group(1)
    return None


def _wrapper_path(node_module: str) -> Path:
    return NODES_DIR / (node_module.split(".")[-1] + ".py")


def _wrapper_result_keys(node_module: str) -> set[str]:
    src_path = _wrapper_path(node_module)
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
            for kw in node.keywords:
                if kw.arg:
                    keys.add(kw.arg)
        elif isinstance(node, ast.Assign):
            # Module-level constants like DELTA_KEY = "delta_acc_rf1" turn into
            # dict keys at runtime; treat their RHS string constants as
            # produced-key candidates.
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                keys.add(node.value.value)
    return keys


@pytest.mark.parametrize(
    ("yaml_name", "expected_variant", "expected_delta_symbol"),
    _CARDS,
    ids=[c[0].replace("contextual_drag_", "").replace(".yaml", "") for c in _CARDS],
)
class TestRecursiveCard:
    """All recursive-card contract assertions, parameterised over both variants."""

    def test_card_yaml_parses(
        self, yaml_name: str, expected_variant: str, expected_delta_symbol: str
    ) -> None:
        path = _card_path(yaml_name)
        assert path.exists(), f"{yaml_name} not present under cards/"
        card = _load_card(path)
        assert isinstance(card, dict), f"{yaml_name} did not parse as a mapping"
        missing = REQUIRED_YAML_KEYS - set(card)
        assert not missing, f"{yaml_name} missing required keys: {sorted(missing)}"

    def test_claim_compiles(
        self, yaml_name: str, expected_variant: str, expected_delta_symbol: str
    ) -> None:
        card = _load_card(_card_path(yaml_name))
        claim = card["claim"]["python"]
        compile(claim, f"<{yaml_name}::claim>", "exec")

    def test_declared_symbols_cover_claim(
        self, yaml_name: str, expected_variant: str, expected_delta_symbol: str
    ) -> None:
        card = _load_card(_card_path(yaml_name))
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
            f"{yaml_name}: claim.python references undeclared names "
            f"{sorted(undeclared)}. Allowed = symbols ∪ wrapper_result_keys ∪ "
            f"builtins. symbols={sorted(symbols)}, "
            f"wrapper_keys={sorted(wrapper_keys)}"
        )
        # Sanity: the variant-specific delta symbol must be in the allowed set.
        assert expected_delta_symbol in allowed, (
            f"{yaml_name}: expected_delta_symbol {expected_delta_symbol!r} "
            f"is not in the wrapper's result keys (got {sorted(wrapper_keys)})"
        )

    def test_wrapper_help_returns_zero(
        self, yaml_name: str, expected_variant: str, expected_delta_symbol: str
    ) -> None:
        card = _load_card(_card_path(yaml_name))
        node_module = _node_module_from_pipeline(card)
        assert node_module, f"{yaml_name}: no `python -m cards.nodes.<x>` executable"
        src = _wrapper_path(node_module)
        assert src.exists(), f"{yaml_name}: wrapper {src} missing"
        # Invoke the wrapper as a script (matches the brief: `python
        # cards/nodes/run_recursive_{rf1|naive}.py --help`).
        result = subprocess.run(
            [sys.executable, str(src), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"{src} --help exited {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )

    def test_subprocess_chain_calls_contextual_drag_recursive(
        self, yaml_name: str, expected_variant: str, expected_delta_symbol: str
    ) -> None:
        """Lock in the `python -m contextual_drag recursive run --variant <x>` wiring."""
        card = _load_card(_card_path(yaml_name))
        node_module = _node_module_from_pipeline(card)
        assert node_module, f"{yaml_name}: no wrapper module"
        src_path = _wrapper_path(node_module)
        assert src_path.exists(), f"{yaml_name}: wrapper {src_path} missing"
        source = src_path.read_text()

        # The wrapper must construct the recursive subcommand. Tolerate
        # both `["recursive", "run", ...]` literal segments and the
        # bound `VARIANT` constant.
        assert "recursive" in source, (
            f"{src_path}: wrapper does not reference the 'recursive' verb"
        )
        assert '"run"' in source or "'run'" in source, (
            f"{src_path}: wrapper does not reference the 'run' verb"
        )
        # Variant must be the expected one.
        variant_marker = f'"{expected_variant}"'
        alt_marker = f"'{expected_variant}'"
        assert variant_marker in source or alt_marker in source, (
            f"{src_path}: wrapper does not reference --variant "
            f"{expected_variant} as a string literal"
        )

        # Static AST check: at least one subprocess.run call must chain
        # through `python -m contextual_drag` (matches the same contract
        # test_cards.py enforces on the other cards).
        tree = ast.parse(source)
        run_calls = [
            n for n in ast.walk(tree)
            if (isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "run"
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "subprocess"
                and n.args)
        ]
        assert run_calls, f"{src_path}: no subprocess.run(...) calls found"
        # Pull constants out of every list-shaped first arg (including via
        # `cdrag + [...]` concatenation, mirroring how the other wrappers
        # build their command lines).
        name_to_rhs = _collect_assignments(tree)
        saw_recursive_run = False
        for call in run_calls:
            head = _list_head_constants(call.args[0], name_to_rhs, depth=8)
            if "-m" in head and "contextual_drag" in head:
                if "recursive" in head and "run" in head:
                    saw_recursive_run = True
        assert saw_recursive_run, (
            f"{src_path}: no subprocess.run(...) call chains through "
            f"`python -m contextual_drag recursive run`. "
            f"Inspected calls: "
            f"{[_list_head_constants(c.args[0], name_to_rhs, 8) for c in run_calls]}"
        )


# ---------------------------------------------------------------------------
# AST helpers (mirror tests/test_cards.py exactly so any drift is local)
# ---------------------------------------------------------------------------


def _list_head_constants(
    node: ast.AST,
    name_to_rhs: dict,
    depth: int = 6,
    seen: frozenset = frozenset(),
) -> list:
    if isinstance(node, ast.List):
        out: list = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant):
                out.append(elt.value)
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


def _collect_assignments(tree: ast.AST) -> dict:
    out: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    out[tgt.id] = node.value
    return out
