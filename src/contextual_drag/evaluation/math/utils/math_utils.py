"""
Math utilities for answer extraction and comparison.
"""

import re
from typing import Optional

import sympy
from sympy.parsing.sympy_parser import parse_expr as sympy_parse_expr
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
transformations = (standard_transformations + (implicit_multiplication_application,) + (convert_xor,))

try:
    from math_verify import parse, verify
except ModuleNotFoundError as ex:
    if ex.name != "math_verify":
        raise
    raise ImportError(
        "Missing optional dependency 'math-verify' (module: 'math_verify'). "
        "Install `math-verify` directly, or install `contextual_drag` with the `eval` extra. "
        "For a local checkout, use `pip install '.[eval]'`."
    ) from ex


def strip_leading_zero(ans: str) -> str:
    """Remove leading zeros from numeric strings (except for decimals)."""
    if ans.startswith("0") and "." not in ans and ans != "0":
        ans = ans.lstrip("0")
    return ans


def extract_boxed_answer(response_text: str) -> Optional[str]:
    """
    Extract answer from \boxed{} format in the response text.
    Returns None if no boxed answer is found.
    """
    # Pattern to match \boxed{...} with nested braces
    # This handles cases like \boxed{\frac{1}{4}} properly
    box_pattern = r'\\boxed\{((?:[^{}]|{[^{}]*})*)\}'
    matches = re.findall(box_pattern, response_text)

    if matches:
        # Return the last boxed answer found (most likely the final answer)
        return matches[-1].strip()
    return None


def is_equivalent_sympy(ans1: Optional[str], ans2: Optional[str]) -> bool:
    """
    Check if two mathematical expressions are equivalent using sympy.
    Returns True if equivalent, False otherwise.
    """
    if ans1 is None and ans2 is None:
        return True

    if ans1 is None or ans2 is None:
        return False

    try:
        # Remove leading zeros
        ans1 = strip_leading_zero(ans1)
        ans2 = strip_leading_zero(ans2)

        # Parse expressions
        expr1 = sympy_parse_expr(f"{ans1}", transformations=transformations, evaluate=True)
        expr2 = sympy_parse_expr(f"{ans2}", transformations=transformations, evaluate=True)

        # Check if they simplify to the same expression
        return sympy.simplify(expr1 - expr2) == 0

    except Exception as e:
        # If parsing fails, do string comparison
        return ans1.strip() == ans2.strip()

def is_equivalent_math(ans1: Optional[str], ans2: Optional[str]) -> bool:
    """
    Check if two mathematical expressions are equivalent using sympy.
    Returns True if equivalent, False otherwise.
    Assume that ans1 is the model answer and ans2 is the expected answer.
    """
    if ans1 is None and ans2 is None:
        return True

    if ans1 is None or ans2 is None:
        return False

    if ans1.strip() == ans2.strip():
        return True

    if len(ans1) > 100 or len(ans2) > 100:
        return ans1.strip() == ans2.strip()

    try:
        answer = parse(f"${ans1}$")
        expected = parse(f"${ans2}$")
        return verify(expected, answer)

    except Exception as e:
        print(e)
        # If parsing fails, do string comparison
        return ans1.strip() == ans2.strip()
