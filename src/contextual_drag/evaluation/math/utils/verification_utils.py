"""Math-evaluator utilities: leading-zero strip + boxed-answer extraction.

The game-of-24 verifier was moved to
`contextual_drag.evaluation.game_of_24.verification`.
"""

import re
from typing import Optional


def strip_leading_zero(ans: str) -> str:
    """Remove leading zeros from numeric strings (except for decimals)."""
    if ans.startswith("0") and "." not in ans and ans != "0":
        ans = ans.lstrip("0")
    return ans


def extract_boxed_answer(response_text: str) -> Optional[str]:
    """Extract the last `\\boxed{...}` answer in the response, or None."""
    box_pattern = r'\\boxed\{((?:[^{}]|{[^{}]*})*)\}'
    matches = re.findall(box_pattern, response_text)
    if matches:
        return matches[-1].strip()
    return None
