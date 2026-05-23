"""
Math utilities for answer extraction and comparison.
"""

import re
from typing import Optional

import sympy
from sympy.parsing.sympy_parser import parse_expr as sympy_parse_expr
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
transformations = (standard_transformations + (implicit_multiplication_application,) + (convert_xor,))

from .math_utils import is_equivalent_math

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


def is_correct_game_of_24(extracted_answer, ground_truth: Optional[int]) -> bool:
    try:
        extracted_formula = extracted_answer.replace('×', '*').replace('÷', '/').replace(' ', '').split('=')[0].strip()

        # Regular expression to match any integer or decimal number
        number_pattern = r'\d+(?:\.\d+)?'
        # Find all matches of the pattern in the expression
        numbers = re.findall(number_pattern, extracted_formula)
        # Convert strings to integers or floats as appropriate
        extracted_numbers = []
        for num in numbers:
            if '.' in num:
                extracted_numbers.append(float(num))
            else:
                extracted_numbers.append(int(num))
        if is_equivalent_math(extracted_formula, '24') and sorted(extracted_numbers) == sorted(ground_truth):
            return True
        return False
    except:
        return None
