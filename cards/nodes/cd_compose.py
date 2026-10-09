"""
Select problems from one model's clean-slate pool and build its draft ladder.

Thin wrapper over ``contextual_drag data draft-composition``. Two manifests
come out, one per downstream generation round:

* ``manifest_fpath`` names the context dataset -- one row per (problem, rung,
  order) -- for the ladder round, which answers each row once;
* ``baseline_manifest_fpath`` names the kept problems in benchmark form, for
  the baseline round, which answers each problem afresh.

Both carry ``n_kept`` as the number of rows the next round should generate,
which is what ``cd_inference`` reads from an upstream manifest.

Keeping no problem is a real outcome -- the model is too strong or too weak on
this benchmark for any problem to sit in its uncertain zone -- so it is
recorded and the pipeline continues to a legible INCONCLUSIVE.
"""
from __future__ import annotations

import json
from pathlib import Path

import scriptconfig as scfg

from cards.nodes._step import read_manifest, run_contextual_drag, write_manifest


class CDComposeCLI(scfg.DataConfig):
    """Build the draft-mixture contexts for one model."""

    postprocess_manifest_fpath = scfg.Value(
        None, help='Manifest written by the pool postprocess node.',
        tags=['in_path'])

    K = scfg.Value(4, type=int, help='Drafts per context.', tags=['algo_param'])
    n_orders = scfg.Value(8, type=int, help='Distinct orders per draft set.',
                          tags=['algo_param'])
    min_correct = scfg.Value(
        4, type=int, help='Keep a problem only with >= this many correct pool answers.',
        tags=['algo_param'])
    min_incorrect = scfg.Value(
        4, type=int, help='Keep a problem only with >= this many incorrect pool answers.',
        tags=['algo_param'])
    seed = scfg.Value(20260916, type=int, help='Composition seed.',
                      tags=['algo_param'])
    max_draft_chars = scfg.Value(
        0, type=int, help='Truncate each draft to this many characters (0 = never).',
        tags=['algo_param'])

    manifest_fpath = scfg.Value(
        'compose.json', help='Manifest naming the context dataset.',
        tags=['out_path', 'primary'])
    baseline_manifest_fpath = scfg.Value(
        'compose_baseline.json', help='Manifest naming the kept problems.',
        tags=['out_path'])

    @classmethod
    def main(cls, argv=None, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)

        manifest_fpath = Path(config.manifest_fpath).resolve()
        baseline_fpath = Path(config.baseline_manifest_fpath).resolve()
        upstream = read_manifest(config.postprocess_manifest_fpath)

        def skip(reason, **extra):
            for fpath in (manifest_fpath, baseline_fpath):
                write_manifest(fpath, skipped=True, reason=reason, n_kept=0,
                               dataset_fpath=None, **extra)

        if upstream.get('skipped') or not upstream.get('processed_ds'):
            skip('no processed pool upstream')
            return

        output_dir = manifest_fpath.parent / 'composition'
        returncode = run_contextual_drag([
            'data', 'draft-composition',
            '--input_dir', upstream['processed_ds'],
            '--output_dir', output_dir,
            '--K', config.K,
            '--n_orders', config.n_orders,
            '--min_correct', config.min_correct,
            '--min_incorrect', config.min_incorrect,
            '--seed', config.seed,
            '--max_draft_chars', config.max_draft_chars,
        ], check=False)

        summary_fpath = output_dir / 'draft_composition.json'
        if not summary_fpath.exists():
            raise RuntimeError(
                f'draft-composition exited {returncode} without writing '
                f'{summary_fpath}')
        summary = json.loads(summary_fpath.read_text())
        common = dict(composition_summary=summary_fpath,
                      n_problems=summary['n_kept'],
                      n_pool_problems=summary['n_pool_problems'])
        if returncode != 0 or not summary['n_kept']:
            skip(f'no problem has >= {config.min_correct} correct and '
                 f'>= {config.min_incorrect} incorrect pool answers', **common)
            return

        write_manifest(manifest_fpath, skipped=False,
                       dataset_fpath=output_dir / 'contexts.ds',
                       n_kept=summary['n_contexts'], **common)
        write_manifest(baseline_fpath, skipped=False,
                       dataset_fpath=output_dir / 'kept.ds',
                       n_kept=summary['n_kept'], **common)


__cli__ = CDComposeCLI

if __name__ == '__main__':
    CDComposeCLI.main()
