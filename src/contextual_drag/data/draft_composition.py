"""
Build draft-mixture contexts for the route-model evaluation.

Input: one model's processed clean-slate pool -- ``n`` answers per problem,
each with its correctness, extracted answer, and final text (the output of
``eval`` with ``--flatten_dataset`` followed by
``data initial-sampling-postprocess``).

Faithful to the original study's builder (``build_draft_composition.py``,
arm D, part 1):

* An answer with no gradable result (correctness ``None``) is neither correct
  nor incorrect. It is left out of the pool, so it can neither qualify a
  problem nor be shown as a draft.
* A problem is kept when its pool holds at least ``min_correct`` correct and
  ``min_incorrect`` incorrect answers, so every rung from all-wrong to
  all-correct is built on the same problems.
* For each kept problem and rung ``T`` in ``K..0``, one draft set of ``T``
  correct and ``K - T`` incorrect answers is drawn without replacement, and
  shown in up to ``n_orders`` *distinct* orders: all ``K!`` permutations are
  shuffled and the first ``n_orders`` with distinct draft-text sequences are
  kept. Each order is one output row, so an inference round with ``n = 1``
  yields one response per order. The draft set is fixed within a rung.
* A draft is the answer's final text -- the reasoning trace removed by the
  postprocess step -- and is not truncated unless ``max_draft_chars`` is set.

Randomness for each (problem, rung) comes from its own generator, keyed by
the seed, the problem id and the rung, so adding or removing a problem never
reshuffles another.

A second output, ``kept.ds``, holds the kept problems in benchmark form, so
the baseline round can sample them afresh.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

__all__ = ['build_contexts', 'distinct_permutations', 'main']

#: Benchmark columns carried onto every output row when present.
_BENCHMARK_COLUMNS = ('problem', 'answer', 'source', 'domain', 'label',
                      'llama8b_solve_rate')


def _rng_for(*parts) -> np.random.Generator:
    """
    One generator per key, as in the original builder.

    Example:
        >>> a = _rng_for(1, 'aime24:2', 3).integers(0, 10**9)
        >>> b = _rng_for(1, 'aime24:2', 3).integers(0, 10**9)
        >>> c = _rng_for(1, 'aime24:2', 2).integers(0, 10**9)
        >>> bool(a == b), bool(a == c)
        (True, False)
    """
    digest = hashlib.blake2b('|'.join(map(str, parts)).encode(),
                             digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, 'big'))


def distinct_permutations(drafts, n_orders, rng):
    """
    Up to ``n_orders`` orders of ``drafts`` with distinct text sequences.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> orders = distinct_permutations([{'text': t} for t in 'abcd'], 8, rng)
        >>> len(orders), len({tuple(d['text'] for d in o) for o in orders})
        (8, 8)
        >>> # identical drafts admit only one distinct order
        >>> len(distinct_permutations([{'text': 'x'}] * 3, 8, rng))
        1
    """
    perms = list(itertools.permutations(range(len(drafts))))
    rng.shuffle(perms)
    seen, out = set(), []
    for perm in perms:
        key = tuple(drafts[i]['text'] for i in perm)
        if key in seen:
            continue
        seen.add(key)
        out.append([drafts[i] for i in perm])
        if len(out) == n_orders:
            break
    return out


def _label(value):
    """True, False, or None for a correctness value of any spelling."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {'true', 'correct', '1'}:
            return True
        if text in {'false', 'incorrect', '0'}:
            return False
        return None
    return bool(value)


def build_contexts(pool_rows, *, K=4, n_orders=8, min_correct=4,
                   min_incorrect=4, seed=20260916, max_draft_chars=0,
                   correctness_key='init_response_generations_correctness',
                   answer_key='init_response_generations_extracted_answer',
                   text_key='init_response_final', id_key='id'):
    """
    Select problems and build every (problem, rung, order) row.

    Returns:
        Tuple[list, list, dict]: context rows, kept benchmark rows, and the
            per-problem pool counts keyed by problem id.
    """
    if K < 1:
        raise ValueError('K must be at least 1')
    if min_correct < K or min_incorrect < K:
        raise ValueError(
            f'min_correct and min_incorrect must be >= K={K} so every rung '
            f'from all-wrong to all-correct is buildable on the same problems')

    by_problem = defaultdict(lambda: {'correct': [], 'incorrect': [],
                                      'ungraded': 0, 'row': None})
    for row in pool_rows:
        pid = str(row[id_key])
        entry = by_problem[pid]
        entry['row'] = entry['row'] or row
        label = _label(row.get(correctness_key))
        if label is None:
            entry['ungraded'] += 1
            continue
        text = row.get(text_key) or ''
        if max_draft_chars:
            text = text[:max_draft_chars]
        answer = row.get(answer_key)
        draft = {'text': text,
                 'answer': None if answer is None else str(answer).strip(),
                 'response_id': row.get('response_unique_id')}
        entry['correct' if label else 'incorrect'].append(draft)

    contexts, kept, pool = [], [], {}
    for pid in sorted(by_problem):
        entry = by_problem[pid]
        n_c, n_i = len(entry['correct']), len(entry['incorrect'])
        pool[pid] = {'pool_correct': n_c, 'pool_incorrect': n_i,
                     'pool_ungraded': entry['ungraded']}
        if n_c < min_correct or n_i < min_incorrect:
            continue
        base = {col: entry['row'][col] for col in _BENCHMARK_COLUMNS
                if col in entry['row']}
        kept.append(dict(base, id=pid))
        for T in range(K, -1, -1):
            rng = _rng_for(seed, pid, f'T{T}F{K - T}')
            c_idx = rng.choice(n_c, T, replace=False) if T else []
            i_idx = rng.choice(n_i, K - T, replace=False) if K - T else []
            drafts = ([dict(entry['correct'][j], correct=True) for j in c_idx]
                      + [dict(entry['incorrect'][j], correct=False) for j in i_idx])
            for order, shown in enumerate(distinct_permutations(drafts, n_orders, rng)):
                row = dict(base)
                row['id'] = f'{pid}|T{T}|o{order}'
                row['dc_problem_id'] = pid
                # The arm-D prompt names the drafts' problem separately from
                # the problem to solve; here they are the same problem.
                row['dc_draft_problem_text'] = base.get('problem')
                row['dc_T'] = T
                row['dc_order'] = order
                for i, d in enumerate(shown):
                    row[f'traj{i + 1}'] = d['text']
                    row[f'traj{i + 1}_correctness'] = d['correct']
                    row[f'traj{i + 1}_extracted_answer'] = d['answer']
                    row[f'traj{i + 1}_response_id'] = d['response_id']
                contexts.append(row)
    return contexts, kept, pool


def main(args):
    from datasets import Dataset, load_from_disk

    pool_ds = load_from_disk(str(args.input_dir))
    contexts, kept, pool = build_contexts(
        pool_ds, K=args.K, n_orders=args.n_orders,
        min_correct=args.min_correct, min_incorrect=args.min_incorrect,
        seed=args.seed, max_draft_chars=args.max_draft_chars)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        'K': args.K, 'n_orders': args.n_orders, 'seed': args.seed,
        'min_correct': args.min_correct, 'min_incorrect': args.min_incorrect,
        'n_pool_problems': len(pool), 'n_kept': len(kept),
        'n_contexts': len(contexts),
        'kept_ids': [row['id'] for row in kept], 'pool': pool,
    }
    (out / 'draft_composition.json').write_text(json.dumps(summary, indent=1) + '\n')
    if not kept:
        print('[draft-composition] WARNING: no problem has enough correct and '
              'incorrect answers in the pool; nothing to build.')
        return 1
    Dataset.from_list(contexts).save_to_disk(str(out / 'contexts.ds'))
    Dataset.from_list(kept).save_to_disk(str(out / 'kept.ds'))
    print(f'[draft-composition] kept {len(kept)}/{len(pool)} problems, '
          f'{len(contexts)} context rows -> {out}')
    return 0
