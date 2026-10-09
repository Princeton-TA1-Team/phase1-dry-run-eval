"""
The ladder file: one model's draft-mixture measurements, as JSON.

Written by the measurement stage and read by the leave-one-model-out stage.
It is the only thing the two stages share, so it carries everything the fit
needs and enough provenance to audit it::

    {
      "schema": "contextual_drag.route_model.ladder/1",
      "model": "Qwen3_8B",
      "K": 4,
      "problems": [
        {"id": "aime24:2", "pool_correct": 7, "pool_n": 16,
         "baseline_correct": 9, "baseline_n": 16}
      ],
      "contexts": [
        {"problem_id": "aime24:2", "T": 1,
         "draft_answers": ["204", "17", "17", "9"],
         "draft_correct": [true, false, false, false],
         "wrong_classes": [2, 1],
         "n": 8, "k": 3}
      ]
    }

``wrong_classes`` are the multiplicities of the distinct wrong answers among
the drafts, grouped by :func:`norm_answer`. An unparseable draft answer is
its own class: two drafts that both failed to produce an answer did not
agree on one.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from contextual_drag.analysis.route_model.answers import norm_answer
from contextual_drag.analysis.route_model.model import Ladder

SCHEMA = 'contextual_drag.route_model.ladder/1'

__all__ = ['SCHEMA', 'wrong_classes', 'write_ladder', 'read_ladder',
           'ladder_from_payload']


def wrong_classes(answers: List, correct: List[bool]) -> List[int]:
    """
    Multiplicities of the distinct wrong answers, largest first.

    Answers are grouped by :func:`norm_answer`. A wrong draft with no
    answer is its own class: two drafts that both failed to produce an
    answer did not agree on one.

    Example:
        >>> wrong_classes(['17', '17.', '9', '204'], [False, False, False, True])
        [2, 1]
        >>> wrong_classes([None, '', '3'], [False, False, False])
        [1, 1, 1]
    """
    counts: Dict = {}
    unparsed = 0
    for answer, ok in zip(answers, correct):
        if ok:
            continue
        key = norm_answer(answer)
        if key is None:
            unparsed += 1
            continue
        counts[key] = counts.get(key, 0) + 1
    return sorted(list(counts.values()) + [1] * unparsed, reverse=True)


def write_ladder(fpath, payload: Dict) -> None:
    payload = dict(payload)
    payload.setdefault('schema', SCHEMA)
    fpath = Path(fpath)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.write_text(json.dumps(payload, indent=1) + '\n')


def read_ladder(fpath) -> Ladder:
    return ladder_from_payload(json.loads(Path(fpath).read_text()))


def ladder_from_payload(payload: Dict) -> Ladder:
    """
    Build a :class:`Ladder` from a ladder payload.

    Baseline accuracy ``q`` is the fresh baseline sample, never the draft
    pool: the pool chose the problems, so its accuracy is biased toward the
    selection window.
    """
    if payload.get('schema') != SCHEMA:
        raise ValueError(f'unexpected ladder schema {payload.get("schema")!r}')
    problems = payload['problems']
    ids = [str(p['id']) for p in problems]
    index = {pid: i for i, pid in enumerate(ids)}
    q = np.array([p['baseline_correct'] / p['baseline_n'] for p in problems],
                 dtype=float)
    contexts = payload['contexts']
    kmax = max([len(c['wrong_classes']) for c in contexts] + [1])
    wrong = np.zeros((len(contexts), kmax))
    for j, c in enumerate(contexts):
        wrong[j, :len(c['wrong_classes'])] = c['wrong_classes']
    return Ladder(
        model=str(payload['model']),
        problem_ids=ids,
        q=q,
        ctx_problem=[index[str(c['problem_id'])] for c in contexts],
        ctx_T=[int(c['T']) for c in contexts],
        ctx_wrong=wrong,
        ctx_k=[c['k'] for c in contexts],
        ctx_n=[c['n'] for c in contexts],
        extra={'K': payload.get('K')},
    )
