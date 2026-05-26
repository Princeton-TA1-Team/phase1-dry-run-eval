"""Parse self-discovered verdict tags from response text.

The verdict tags `<overall_verdict>...</overall_verdict>` (1f) and
`<overall_verdict_{1,2}>...</overall_verdict_{1,2}>` (2f) live in the model's
*final* answer (after thinking). All eight section-3 model families emit them
verbatim regardless of their thinking-trace format, so a regex match on the
lowercased response is sufficient — no family-specific stripping needed.

Auxiliary keyword fallbacks (`"draft is incorrect"`, `\\boxed{\\text{incorrect}}`)
inherit the original heuristics from `analysis/correct_verification_conditioning/
analysis.py` to handle stragglers that broke the strict tag.
"""
from __future__ import annotations

import re
from typing import Optional


_INCORRECT_KEYWORDS = (
    "draft solution is incorrect",
    "draft is incorrect",
    "draft incorrect",
    r"\boxed{\text{incorrect}}",
)
_CORRECT_KEYWORDS = (
    "draft solution is correct",
    "draft is correct",
    "draft correct",
    r"\boxed{\text{correct}}",
)
_VERDICT_TAG = re.compile(r"<overall_verdict>\s*([^\n<]*?)\s*</overall_verdict>", re.DOTALL)
_VERDICT_TAG_NUM = re.compile(
    r"<overall_verdict_(\d+)>\s*(.*?)\s*</overall_verdict_\1>", re.DOTALL
)


def parse_verdict_1f(output: Optional[str]) -> Optional[bool]:
    """For 1f: return True if the verifier flagged the draft as incorrect.

    Returns None when the response is empty or no verdict can be parsed.
    """
    if output is None or not output.strip():
        return None
    txt = output.lower().replace("*", "").replace("\n", "")

    for kw in _CORRECT_KEYWORDS:
        if kw in txt:
            return False
    for kw in _INCORRECT_KEYWORDS:
        if kw in txt:
            return True

    matches = _VERDICT_TAG.findall(output.lower())
    if not matches:
        return None
    return matches[-1].strip() == "incorrect"


def parse_verdict_2f(output: Optional[str]) -> bool:
    """For 2f: True iff *both* verdict tags are 'incorrect'.

    Mirrors the existing convention: missing tags or mixed verdicts → False
    (the entry is then dropped). We return bool not Optional to match the
    upstream filter logic.
    """
    if output is None or not output.strip():
        return False
    txt = output.lower().replace("*", "").replace("\n", "")
    matches = _VERDICT_TAG_NUM.findall(txt)
    verdicts = [v.strip() for _, v in matches]
    if len(verdicts) >= 2 and verdicts[0] == "incorrect" and verdicts[1] == "incorrect":
        return True
    return False


def annotate_responses_1f(entry: dict) -> dict:
    """In-place: tag each response with `verdict` (bool|None). Set entry['valid_verdict']."""
    valid = False
    for r in entry["init_response_generations"]:
        verdict = parse_verdict_1f(r.get("generated_response"))
        r["verdict"] = verdict
        if verdict is True and r.get("finish_reason") == "stop":
            valid = True
    entry["valid_verdict"] = valid
    return entry


def annotate_responses_2f(entry: dict) -> dict:
    valid = False
    for r in entry["init_response_generations"]:
        verdict = parse_verdict_2f(r.get("generated_response"))
        r["verdict"] = verdict
        if verdict is True:
            valid = True
    entry["valid_verdict"] = valid
    return entry


def filter_responses_1f(entry: dict) -> dict:
    """Keep only responses whose verdict was 'incorrect' AND finish_reason == 'stop'."""
    kept = [
        r for r in entry["init_response_generations"]
        if r.get("verdict") is True and r.get("finish_reason") == "stop"
    ]
    entry["init_response_generations"] = kept
    return entry


def filter_responses_2f(entry: dict) -> dict:
    kept = [
        r for r in entry["init_response_generations"]
        if r.get("verdict") is True
    ]
    entry["init_response_generations"] = kept
    return entry
