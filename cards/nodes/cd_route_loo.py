"""
Terminal node of the route-model card: leave-one-model-out transfer.

Fans in the ladder file of every model the card measured, through a kwdagger
gather, adds the bundled ladders of already-characterized source models, runs
``contextual_drag analysis route_model loo`` with the measured models as the
targets -- a fold per target, then the joint problem bootstrap -- and emits
the card's whole result. A source ladder is never a target: it only fixes the
shared exponent and never sees the target's predicted rungs.

The verdict is the claim's to make. This node reports the errors, the
intervals and the same pass rule, so the artifact is readable on its own, and
it reports INCONCLUSIVE when the measurements cannot support a verdict: a
target without a usable ladder, or fewer targets than the card declares.
"""
from __future__ import annotations

import json
from pathlib import Path

import scriptconfig as scfg

from cards.nodes._step import run_contextual_drag, write_manifest
import magnet.theory as theory

#: Bump when the terminal-result shape changes incompatibly.
SCHEMA_VERSION = 1


# The pass criterion -- tolerance, rungs, how many models must pass -- and the
# fitting settings are card defaults fixed before any result is seen, which
# is what `hprespecified` asks.
@theory.satisfies('Hygiene.Inference.threshold_exceeds_sampling_error::hprespecified',
                  note='tolerance_pp, primary rungs and min_models_pass are card '
                       'defaults fixed in the evaluation plan before submission')
class CDRouteLooCLI(scfg.DataConfig):
    """Fit, transfer and score the route model across models."""

    ladders_manifest_fpath = scfg.Value(
        None, help='kwdagger gather manifest: one ladder file path per line.',
        tags=['in_path'])

    source_ladders_dpath = scfg.Value(
        '', type=str,
        help=('Directory of bundled source ladders (*.json): models measured '
              'before the run that serve only as sources. Empty means every '
              'source is measured by the card itself.'),
        tags=['algo_param'])
    n_targets_expected = scfg.Value(
        1, type=int,
        help='Target models the card measures. Fewer is INCONCLUSIVE.',
        tags=['algo_param'])
    calibration_rungs = scfg.Value('0,1', type=str, tags=['algo_param'])
    primary_rungs = scfg.Value('2,3', type=str, tags=['algo_param'])
    secondary_rungs = scfg.Value('4', type=str, tags=['algo_param'])
    tolerance_pp = scfg.Value(5.0, type=float, tags=['algo_param'])
    min_models_pass = scfg.Value(1, type=int, tags=['algo_param'])
    n_boot = scfg.Value(2000, type=int, tags=['algo_param'])
    boot_seed = scfg.Value(20260904, type=int, tags=['algo_param'])
    confidence = scfg.Value(0.95, type=float, tags=['algo_param'])

    summary_fpath = scfg.Value(
        'results.json', help="The card's terminal artifact.",
        tags=['out_path', 'primary'])

    # What the card's verdict stands on, stated where the verdict's inputs are
    # assembled. The transfer conjecture is what the card TESTS; its premises
    # are the route model's own modelling assumptions, and none of them is
    # established by running this node -- they are assumed, and named, so a
    # reader sees exactly what a VERIFIED does and does not show.
    @theory.tests('AIQ.Teams.Princeton.RouteModel.parameter_transfer')
    @theory.assumes('AIQ.Teams.Princeton.RouteModel.parameter_transfer::hcommon',
                    note='one (a, b) per model and one gamma for all models; '
                         'not checked against per-problem heterogeneity')
    @theory.assumes('AIQ.Teams.Princeton.RouteModel.parameter_transfer::hretain',
                    note='the independent route keeps the fresh no-draft accuracy '
                         'q in context; the card measures q, it does not test this')
    @theory.satisfies('AIQ.Teams.Princeton.RouteModel.parameter_transfer::hself',
                      note='every model sees only drafts it wrote itself: the '
                           'pool, ladder and baseline rounds share one model config, '
                           'and the bundled sources were built the same way')
    @theory.satisfies('AIQ.Teams.Princeton.RouteModel.parameter_transfer::hheldout',
                      note='the target contributes only its baseline and its '
                           'calibration rungs; the predicted rungs never enter a fit')
    # The sampling-error premises. The plan computes, and this node reports,
    # simultaneous intervals; at 18 to 30 problems per model they are expected
    # to be wider than the tolerance, so the verdict rests on point estimates.
    @theory.tests('Hygiene.Inference.threshold_exceeds_sampling_error')
    @theory.assumes('Hygiene.Inference.threshold_exceeds_sampling_error::hn_sufficient',
                    note='about 22 eligible problems for the target; Bonferroni-adjusted '
                         'bootstrap intervals are reported and are expected to be '
                         'wider than 5 points; 50-110 problems would be needed')
    @theory.satisfies('Hygiene.Inference.threshold_exceeds_sampling_error::hmultiple',
                      note='the primary errors (T = 2, 3 per target) form one '
                           'Bonferroni family and the T = 4 errors another')
    @theory.approximates('Hygiene.Measurement.measured_score_tracks_construct')
    @theory.approximates('Hygiene.Measurement.measured_score_tracks_construct::hscorer',
                         note='the last boxed answer, compared to the key with '
                              'math_verify; equivalent forms are matched, unboxed '
                              'answers are wrong')
    @theory.assumes('Hygiene.Measurement.measured_score_tracks_construct::hcontam',
                    note='AIME and HMMT 2024-25 are public; no contamination check')
    @theory.assumes('Hygiene.Measurement.measured_score_tracks_construct::hstable',
                    note='decoding seeds and draft orders are fixed and averaged over; '
                         'hardware and serving stack are not controlled')
    @classmethod
    def main(cls, argv=None, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)
        summary_fpath = Path(config.summary_fpath).resolve()

        ladder_fpaths = [
            Path(line.strip()) for line in
            Path(config.ladders_manifest_fpath).read_text().splitlines()
            if line.strip()
        ]
        ladder_status = {}
        usable = []
        for fpath in ladder_fpaths:
            payload = json.loads(fpath.read_text())
            ladder_status[payload['model']] = {
                'status': payload.get('status', 'OK'),
                'detail': payload.get('detail', ''),
                'n_problems': len(payload.get('problems', [])),
            }
            if payload.get('status', 'OK') == 'OK' and payload.get('problems'):
                usable.append(fpath)
        targets = sorted(ladder_status)

        sources = []
        if str(config.source_ladders_dpath or '').strip():
            sources = sorted(Path(config.source_ladders_dpath).glob('*.json'))
            if not sources:
                raise FileNotFoundError(
                    f'no source ladders in {config.source_ladders_dpath}')
        source_models = [json.loads(f.read_text())['model'] for f in sources]
        clash = sorted(set(source_models) & set(targets))
        if clash:
            raise ValueError(
                f'{clash} is both a measured target and a bundled source; a '
                'target must never be its own source')

        problems = [f'{m}: {s["detail"] or s["status"]}'
                    for m, s in sorted(ladder_status.items())
                    if s['status'] != 'OK' or not s['n_problems']]
        if problems or len(usable) != int(config.n_targets_expected):
            detail = '; '.join(problems) or (
                f'{len(usable)} usable target ladders, the card declares '
                f'{config.n_targets_expected}')
            _emit(summary_fpath, status='INCONCLUSIVE', detail=detail,
                  ladders=ladder_status, sources=source_models)
            return

        analysis_fpath = summary_fpath.parent / 'route_model_loo.json'
        run_contextual_drag([
            'analysis', 'route_model', 'loo',
            '--ladders', *usable, *sources,
            '--targets', *targets,
            '--calibration_rungs', config.calibration_rungs,
            '--primary_rungs', config.primary_rungs,
            '--secondary_rungs', config.secondary_rungs,
            '--tolerance_pp', config.tolerance_pp,
            '--min_models_pass', config.min_models_pass,
            '--n_boot', config.n_boot,
            '--seed', config.boot_seed,
            '--confidence', config.confidence,
            '--out', analysis_fpath,
        ])
        analysis = json.loads(analysis_fpath.read_text())
        _emit(summary_fpath,
              status='VERIFIED' if analysis['evaluation_pass'] else 'FALSIFIED',
              detail='', ladders=ladder_status, sources=source_models,
              analysis=analysis, analysis_fpath=analysis_fpath)


def _emit(summary_fpath, *, status, detail, ladders, sources=(),
          analysis=None, analysis_fpath=None):
    """
    Write the terminal artifact.

    Scalars sit at the top level so a card can label them as metrics;
    ``per_model`` nests one mapping per target model.
    """
    analysis = analysis or {}
    keys = ('n_models', 'n_models_valid', 'n_models_pass', 'n_models_supported',
            'n_predictions', 'n_predictions_scored', 'mae_pp', 'max_abs_err_pp',
            'frac_within_tol', 'mae_pp_primary', 'tolerance_pp', 'min_models_pass')
    write_manifest(
        summary_fpath,
        schema_version=SCHEMA_VERSION,
        status=status,
        detail=detail,
        **{k: analysis.get(k) for k in keys},
        per_model=analysis.get('per_model', {}),
        ladders=ladders,
        sources=list(sources),
        bonferroni=analysis.get('bonferroni'),
        analysis_fpath=analysis_fpath,
    )
    print(f'[route_loo] status={status} n_models_pass={analysis.get("n_models_pass")} '
          f'mae_pp={analysis.get("mae_pp")} {detail}', flush=True)


__cli__ = CDRouteLooCLI

if __name__ == '__main__':
    CDRouteLooCLI.main()
