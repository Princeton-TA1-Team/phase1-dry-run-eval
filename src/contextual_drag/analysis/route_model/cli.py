"""CLI surface for the route-model transfer analysis.

One verb:

  ``loo``  -- read one ladder file per model, run every leave-one-model-out
              fold and the joint problem bootstrap, and write a summary JSON.
"""
from __future__ import annotations

import scriptconfig as scfg


class RouteModelLooCLI(scfg.DataConfig):
    """Leave-one-model-out transfer of the route model.

    Each ladder is one model's draft-mixture measurements (see
    ``contextual_drag.analysis.route_model.io``). Every model is the target
    once; the others are its sources.
    """
    ladders = scfg.Value(
        None, required=True, nargs='+',
        help='Ladder JSON files, one per model.', tags=['in_path'])
    targets = scfg.Value(
        None, nargs='*',
        help=('Models to hold out and score. Unset scores every model; the '
              'others then serve only as sources.'), tags=['algo_param'])
    calibration_rungs = scfg.Value(
        '0,1', type=str, help='Target rungs used to set its weights.',
        tags=['algo_param'])
    primary_rungs = scfg.Value(
        '2,3', type=str, help='Predicted rungs that decide the pass rule.',
        tags=['algo_param'])
    secondary_rungs = scfg.Value(
        '4', type=str, help='Predicted rungs reported separately.', tags=['algo_param'])
    tolerance_pp = scfg.Value(5.0, type=float, tags=['algo_param'])
    min_models_pass = scfg.Value(2, type=int, tags=['algo_param'])
    n_boot = scfg.Value(2000, type=int, help='0 skips the bootstrap.',
                        tags=['algo_param'])
    seed = scfg.Value(20260904, type=int, tags=['algo_param'])
    confidence = scfg.Value(0.95, type=float, tags=['algo_param'])
    out = scfg.Value(None, required=True, tags=['out_path'])

    @classmethod
    def main(cls, argv=None, **kwargs):
        import json
        from pathlib import Path

        cfg = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)
        result = run_from_files(
            cfg.ladders,
            calibration_rungs=_rungs(cfg.calibration_rungs),
            primary_rungs=_rungs(cfg.primary_rungs),
            secondary_rungs=_rungs(cfg.secondary_rungs),
            tolerance_pp=float(cfg.tolerance_pp),
            min_models_pass=int(cfg.min_models_pass),
            n_boot=int(cfg.n_boot), seed=int(cfg.seed),
            confidence=float(cfg.confidence),
            targets=list(cfg.targets) if cfg.targets else None)
        out = Path(cfg.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, allow_nan=False,
                                  default=_jsonable) + '\n')


def _rungs(text):
    """
    Example:
        >>> _rungs('2,3')
        (2, 3)
        >>> _rungs([4])
        (4,)
    """
    if isinstance(text, (list, tuple)):
        return tuple(int(t) for t in text)
    return tuple(int(t) for t in str(text).replace(' ', '').split(',') if t)


def _jsonable(value):
    raise TypeError(f'not JSON serializable: {type(value)}')


def _finite_or_none(obj):
    """NaN and infinities become None so the summary is strict JSON."""
    import math
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _finite_or_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite_or_none(v) for v in obj]
    return obj


def run_from_files(ladder_fpaths, *, calibration_rungs, primary_rungs,
                   secondary_rungs, tolerance_pp, min_models_pass, n_boot,
                   seed, confidence, targets=None):
    """Load ladders, fit every fold, bootstrap, and summarize."""
    from contextual_drag.analysis.route_model.io import read_ladder
    from contextual_drag.analysis.route_model.loo import (
        bootstrap_loo, run_loo, summarize,
    )
    ladders = [read_ladder(f) for f in ladder_fpaths]
    names = [l.model for l in ladders]
    if len(set(names)) != len(names):
        raise ValueError(f'duplicate models among ladders: {names}')
    # A fixed order makes the fold list, and so the seeded bootstrap,
    # independent of the order the files were gathered in.
    ladders = sorted(ladders, key=lambda l: l.model)
    predict_rungs = tuple(primary_rungs) + tuple(secondary_rungs)
    folds = run_loo(ladders, calibration_rungs, predict_rungs,
                    targets=targets)
    boot = None
    if n_boot > 0:
        boot = bootstrap_loo(ladders, n_boot, seed, calibration_rungs,
                             predict_rungs, progress=True, targets=targets)
    summary = summarize(folds, boot, tolerance_pp=tolerance_pp,
                        min_models_pass=min_models_pass,
                        primary_rungs=primary_rungs,
                        secondary_rungs=secondary_rungs,
                        confidence=confidence)
    summary['config'] = {
        'calibration_rungs': list(calibration_rungs),
        'primary_rungs': list(primary_rungs),
        'secondary_rungs': list(secondary_rungs),
        'n_boot': n_boot, 'seed': seed,
        'models': [l.model for l in ladders],
        'targets': [f['target'] for f in folds],
    }
    summary['folds'] = folds
    return _finite_or_none(summary)


class RouteModelCLI(scfg.ModalCLI):
    loo = RouteModelLooCLI
