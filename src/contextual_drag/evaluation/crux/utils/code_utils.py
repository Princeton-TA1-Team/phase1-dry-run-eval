"""
Math utilities for answer extraction and comparison.
"""

import re
from typing import Optional


def extract_code_answer_from_assert(response_text: str, mode: str) -> Optional[str]:
    pattern = r"assert\s+(.*?)\s*==\s*(.*?)\s*(?:\n|$)"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        if mode == "input":
            return matches[0][0].strip()
        elif mode == "output":
            return matches[0][1].strip()
    return "EMPTYSTRINGWHICHSHOULDNOTMATCHWITHANYTHING"

def extract_code_answer(response_text: str, mode: str) -> Optional[str]:
    """
    Extract answer from
    [ANSWER]
    assert f('hi there') == "hi there"
    [/ANSWER]
    block in the response text.
    Returns None if no [ANSWER]xxx[/ANSWER] block is found.
    """
    if "</think>" in response_text:
        response_text = response_text.split("</think>")[-1]

    pattern = r"\[ANSWER\](.*?)\[/ANSWER\]"
    matches = re.findall(pattern, response_text, re.DOTALL)

    if len(matches) == 0:
        print("No [ANSWER]xxx[/ANSWER] block found, using assert pattern")
        return extract_code_answer_from_assert(response_text, mode)

    if matches:
        # Return the last [ANSWER] xxx [/ANSWER] block found (most likely the final answer)
        scratch = matches[0].strip().replace('\n', '').split('==')
        if mode == "input":
            scratch = scratch[0].strip()
            if "assert f(" in scratch:
                scratch_list = scratch.split("assert f")
                if len(scratch_list) > 1:
                    return 'f' + scratch_list[1].strip()
                else:
                    return "EMPTYSTRINGWHICHSHOULDNOTMATCHWITHANYTHING"
            else:
                return scratch
        elif mode == "output":
            if len(scratch) > 1:
                return scratch[1].strip()
            else:
                return "EMPTYSTRINGWHICHSHOULDNOTMATCHWITHANYTHING"
    return "EMPTYSTRINGWHICHSHOULDNOTMATCHWITHANYTHING"
