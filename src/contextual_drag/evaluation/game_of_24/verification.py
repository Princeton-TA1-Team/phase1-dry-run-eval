"""Verifier for the 24-game task.

A correct answer satisfies two conditions:

1. The arithmetic expression evaluates to 24 (handled by
   :func:`is_equivalent_math` from the math evaluator).
2. The multiset of operand digits in the expression equals the
   multiset of digits the problem supplies as `ground_truth`.
"""
from __future__ import annotations

import re
from typing import Optional

from contextual_drag.evaluation.math.utils.math_utils import is_equivalent_math


_NUMBER_PATTERN = r"\d+(?:\.\d+)?"


def is_correct_game_of_24(extracted_answer, ground_truth: Optional[list]) -> Optional[bool]:
    if ground_truth is None:
        return None
    try:
        extracted_formula = (
            extracted_answer.replace("×", "*").replace("÷", "/").replace(" ", "").split("=")[0].strip()
        )
        numbers = re.findall(_NUMBER_PATTERN, extracted_formula)
        extracted_numbers = [float(n) if "." in n else int(n) for n in numbers]
        if is_equivalent_math(extracted_formula, "24") and sorted(extracted_numbers) == sorted(ground_truth):
            return True
        return False
    except Exception:
        return None
