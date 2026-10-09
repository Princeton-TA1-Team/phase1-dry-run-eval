"""
Answer normalization for grouping drafts that agree.

Ported verbatim from the draft-composition builder of the original
contextual-drag study (``data_generation/build_draft_composition.py``), so
that wrong-answer multiplicities are counted the way the study counted them.
Grading is a separate question and is not done here: two answers that
normalize alike are treated as the same *claim*, whether or not either is
correct.
"""
from __future__ import annotations

import re

__all__ = ['norm_answer']

_MCQ_JUNK = re.compile(r"^\(|\)$|\.$")


def norm_answer(a):
    r"""
    Canonical form of an extracted answer, or None when there is none.

    Example:
        >>> norm_answer('\\dfrac{1}{2}') == norm_answer('dfrac{1}{2}')
        True
        >>> norm_answer(' (b) ')
        'B'
        >>> norm_answer('$\\text{17}$')
        '17'
        >>> norm_answer(None), norm_answer(''), norm_answer('None')
        (None, None, None)
    """
    if a is None:
        return None
    s = str(a).strip()
    if s in ("", "None", "nan"):
        return None
    s = s.upper().replace("\\TEXT", "").replace("\\MATHRM", "")
    s = s.replace("{", "").replace("}", "").replace("$", "").replace("\\", "")
    s = _MCQ_JUNK.sub("", s.strip()).strip()
    return s or None
