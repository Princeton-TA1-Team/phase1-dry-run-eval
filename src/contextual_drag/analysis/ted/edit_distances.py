"""Edit-distance metrics over arithmetic expressions."""

from __future__ import annotations

import ast

import Levenshtein
from zss import Node, simple_distance

from contextual_drag.evaluation.math.utils.math_utils import is_equivalent_math


def _ast_to_tree(node: ast.AST) -> Node:
    """Convert a Python AST expr into a zss.Node tree."""
    if isinstance(node, ast.BinOp):
        root = Node(type(node.op).__name__)
        root.addkid(_ast_to_tree(node.left))
        root.addkid(_ast_to_tree(node.right))
        return root
    if isinstance(node, ast.UnaryOp):
        root = Node(type(node.op).__name__)
        root.addkid(_ast_to_tree(node.operand))
        return root
    if isinstance(node, ast.Constant):
        return Node(str(node.value))
    if isinstance(node, ast.Name):
        return Node(node.id)
    if isinstance(node, ast.Expr):
        return _ast_to_tree(node.value)
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def _parse_expr(expr: str) -> Node:
    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {expr}") from exc
    return _ast_to_tree(parsed.body)


def edit_distance(a: str, b: str, *, metric: str = "levenshtein") -> float:
    """Compute edit distance between two expressions.

    metric: ``"levenshtein"``, ``"tree"``, or ``"binary"``.
    """
    m = metric.lower()
    if m == "levenshtein":
        return float(Levenshtein.distance(str(a).replace(' ', ''),
                                          str(b).replace(' ', '')))
    if m == "tree":
        tree_a, tree_b = _parse_expr(str(a)), _parse_expr(str(b))
        get_children = lambda n: list(n.children) if hasattr(n, "children") else []
        get_label = lambda n: n.label
        return float(
            simple_distance(
                tree_a, tree_b,
                get_children=get_children, get_label=get_label,
            )
        )
    if m == "binary":
        return 1 if is_equivalent_math(a, b) else 0
    raise ValueError(f"Unknown metric: {metric}")


__all__ = ["edit_distance"]
