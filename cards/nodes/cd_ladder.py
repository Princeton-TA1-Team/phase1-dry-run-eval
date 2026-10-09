"""
Assemble one model's ladder file from its two generation rounds.

Fans in three artifacts: the composition summary (which problems were kept
and what the pool held), the evaluated ladder round (one response per
order), and the evaluated baseline round (fresh answers per problem). The
output is the ladder file of ``contextual_drag.analysis.route_model.io`` --
the only thing the leave-one-model-out stage reads.

A problem enters the ladder only with a response at every rung and a baseline
estimate. A problem that lost a rung is dropped, never imputed, and the loss
is recorded: problems whose generations failed did not fail at random (the
longest prompts fail first), so a ladder that lost more than
``max_loss_frac`` of its problems is reported INCONCLUSIVE rather than fit on
a biased remainder.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import scriptconfig as scfg

from cards.nodes._step import read_manifest, write_manifest

#: Bump when the summary shape changes incompatibly.
SCHEMA_VERSION = 1


class CDLadderCLI(scfg.DataConfig):
    """Write one model's ladder file."""

    compose_manifest_fpath = scfg.Value(
        None, help='Manifest from the compose node.', tags=['in_path'])
    ladder_eval_manifest_fpath = scfg.Value(
        None, help='Manifest from the ladder-round eval node.', tags=['in_path'])
    baseline_eval_manifest_fpath = scfg.Value(
        None, help='Manifest from the baseline-round eval node.', tags=['in_path'])

    model_config = scfg.Value(
        None, help='The model these measurements belong to.',
        tags=['algo_param'])
    K = scfg.Value(4, type=int, help='Drafts per context.', tags=['algo_param'])
    max_loss_frac = scfg.Value(
        0.05, type=float,
        help=('Largest fraction of kept problems that may be lost to failed '
              'generations before the ladder is INCONCLUSIVE.'),
        tags=['algo_param'])

    summary_fpath = scfg.Value(
        'ladder_summary.json', help="This node's summary.",
        tags=['out_path', 'primary'])
    ladder_fpath = scfg.Value(
        'ladder.json', help='The ladder file.', tags=['out_path'])

    @classmethod
    def main(cls, argv=None, **kwargs):
        from contextual_drag.analysis.route_model.io import (
            SCHEMA, wrong_classes, write_ladder,
        )
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)
        summary_fpath = Path(config.summary_fpath).resolve()
        ladder_fpath = Path(config.ladder_fpath).resolve()
        model = str(config.model_config)
        K = int(config.K)

        compose = read_manifest(config.compose_manifest_fpath)
        ladder_eval = read_manifest(config.ladder_eval_manifest_fpath)
        baseline_eval = read_manifest(config.baseline_eval_manifest_fpath)

        def emit(status, detail, problems=(), contexts=(), **extra):
            write_ladder(ladder_fpath, {
                'schema': SCHEMA, 'model': model, 'K': K,
                'status': status, 'detail': detail,
                'problems': list(problems), 'contexts': list(contexts)})
            write_manifest(summary_fpath, schema_version=SCHEMA_VERSION,
                           model=model, status=status, detail=detail,
                           ladder_fpath=ladder_fpath, **extra)
            print(f'[ladder] {model} status={status} '
                  f'n_problems={len(problems)} {detail}', flush=True)

        if compose.get('skipped'):
            emit('INCONCLUSIVE', compose.get('reason', 'no problems kept'),
                 n_problems=0, n_kept=0, n_lost=0)
            return
        for name, upstream in (('ladder', ladder_eval), ('baseline', baseline_eval)):
            if upstream.get('skipped') or not upstream.get('dataset_dir'):
                emit('INCONCLUSIVE', f'the {name} round produced no output',
                     n_problems=0, n_kept=int(compose.get('n_problems') or 0),
                     n_lost=int(compose.get('n_problems') or 0))
                return

        composition = json.loads(Path(compose['composition_summary']).read_text())
        kept_ids = [str(p) for p in composition['kept_ids']]
        pool = composition['pool']

        rungs = defaultdict(list)
        for row in _read_evaluated(ladder_eval['dataset_dir']):
            gens = row.get('init_response_generations') or []
            if not gens:
                continue
            answers = [row.get(f'traj{i}_extracted_answer') for i in range(1, K + 1)]
            correct = [bool(row.get(f'traj{i}_correctness')) for i in range(1, K + 1)]
            rungs[(str(row['dc_problem_id']), int(row['dc_T']))].append({
                'answers': answers, 'correct': correct,
                'k': sum(_is_correct(g) for g in gens), 'n': len(gens)})

        baseline = {}
        for row in _read_evaluated(baseline_eval['dataset_dir']):
            gens = row.get('init_response_generations') or []
            if gens:
                baseline[str(row['id'])] = (sum(_is_correct(g) for g in gens), len(gens))

        problems, contexts, lost = [], [], []
        for pid in kept_ids:
            have = [rungs.get((pid, T)) for T in range(K + 1)]
            if not all(have) or pid not in baseline:
                lost.append(pid)
                continue
            problems.append({
                'id': pid,
                'pool_correct': pool[pid]['pool_correct'],
                'pool_n': pool[pid]['pool_correct'] + pool[pid]['pool_incorrect'],
                'baseline_correct': baseline[pid][0],
                'baseline_n': baseline[pid][1]})
            for T, orders in enumerate(have):
                first = orders[0]
                contexts.append({
                    'problem_id': pid, 'T': T,
                    'draft_answers': first['answers'],
                    'draft_correct': first['correct'],
                    'wrong_classes': wrong_classes(first['answers'], first['correct']),
                    'n': sum(o['n'] for o in orders),
                    'k': sum(o['k'] for o in orders)})

        n_kept = len(kept_ids)
        loss_frac = len(lost) / n_kept if n_kept else 0.0
        acc = {}
        for T in range(K + 1):
            rates = [c['k'] / c['n'] for c in contexts if c['T'] == T]
            acc[f'acc_t{T}'] = sum(rates) / len(rates) if rates else None
        q = [p['baseline_correct'] / p['baseline_n'] for p in problems]
        extra = dict(n_problems=len(problems), n_kept=n_kept, n_lost=len(lost),
                     lost_ids=lost, baseline_acc=(sum(q) / len(q)) if q else None,
                     **acc)
        if not problems:
            emit('INCONCLUSIVE', 'no kept problem has a full ladder and a baseline',
                 **extra)
        elif loss_frac > float(config.max_loss_frac):
            emit('INCONCLUSIVE',
                 f'{len(lost)} of {n_kept} kept problems lost a rung or the '
                 f'baseline ({loss_frac:.0%}); the survivors are not a random '
                 f'subset', problems, contexts, **extra)
        else:
            emit('OK', '', problems, contexts, **extra)


def _is_correct(generation) -> bool:
    """A generation is correct only when graded True; ungradable is wrong."""
    if not isinstance(generation, dict):
        return False
    return generation.get('correctness') in (True, 'True', 'correct')


def _read_evaluated(dataset_dir):
    path = Path(dataset_dir) / 'evaluated_inference.jsonl'
    if not path.exists():
        return []
    with open(path) as file:
        return [json.loads(line) for line in file if line.strip()]


__cli__ = CDLadderCLI

if __name__ == '__main__':
    CDLadderCLI.main()
