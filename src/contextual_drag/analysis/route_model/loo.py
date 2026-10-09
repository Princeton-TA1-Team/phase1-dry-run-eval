"""
Leave-one-model-out transfer of the route model, with a joint bootstrap.

Each model serves once as the target. The other models are the sources: their
full measurements give one shared exponent and their own weights. The target
contributes only its baseline accuracy and its accuracy at the calibration
rungs (zero and one correct draft); its weights are fit there with the shared
exponent held fixed, and the route model predicts the remaining rungs. Each
round is a fold.

The pass rule is the plan's: a model passes when its errors at every primary
rung (two and three correct drafts) are at most ``tolerance_pp`` points, and
the evaluation passes when at least ``min_models_pass`` models pass. A fold
whose calibration implies a negative weight is a failed prediction.

Uncertainty comes from a bootstrap over problems that resamples the problem
universe jointly across models -- a problem drawn twice is drawn twice for
every model that kept it -- and refits every stage, sources included. The
primary errors form one Bonferroni family and the secondary ones another.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from contextual_drag.analysis.route_model.model import (
    Ladder, fit_sources, fit_target, observed_accuracy, predict_accuracy,
)

__all__ = ['run_fold', 'run_loo', 'bootstrap_loo', 'summarize']


def run_fold(target: Ladder, sources: Sequence[Ladder],
             calibration_rungs: Sequence[int], predict_rungs: Sequence[int],
             weights: Optional[Dict[str, np.ndarray]] = None) -> Dict:
    """
    Fit on the sources, calibrate the target, and predict its other rungs.

    Returns:
        dict: the fold's exponent, weights, and per-rung prediction,
            observation and signed error in percentage points. ``valid`` is
            False when the calibration implies a negative weight.
    """
    weights = weights or {}
    source_fit = fit_sources(sources, weights=weights)
    target_w = weights.get(target.model)
    target_fit = fit_target(target, source_fit.gamma, rungs=calibration_rungs,
                            weights=target_w)
    valid = (not target_fit.negative_weight
             and math.isfinite(target_fit.a) and math.isfinite(target_fit.b))
    rungs = {}
    for T in list(calibration_rungs) + list(predict_rungs):
        obs = observed_accuracy(target, T, target_w)
        pred = (predict_accuracy(target, T, target_fit.a, target_fit.b,
                                 source_fit.gamma, target_w)
                if valid else float('nan'))
        rungs[int(T)] = {
            'pred': pred, 'obs': obs,
            'err_pp': 100.0 * (pred - obs) if valid else float('nan'),
            'role': 'calibration' if T in calibration_rungs else 'prediction',
        }
    if target_w is None:
        baseline = float(np.mean(target.q)) if len(target.q) else float('nan')
        n_problems = len(target.problem_ids)
    else:
        total = target_w.sum()
        baseline = float(np.sum(target_w * target.q) / total) if total else float('nan')
        n_problems = int(total)
    return {
        'target': target.model,
        'sources': [s.model for s in sources],
        'gamma': source_fit.gamma,
        'source_weights': source_fit.weights,
        'sources_converged': source_fit.converged,
        'a': target_fit.a, 'b': target_fit.b,
        'target_converged': target_fit.converged,
        'valid': bool(valid),
        'detail': target_fit.detail,
        'baseline_acc': baseline,
        'n_problems': n_problems,
        'rungs': rungs,
    }


def run_loo(ladders: Sequence[Ladder], calibration_rungs=(0, 1),
            predict_rungs=(2, 3, 4),
            weights: Optional[Dict[str, np.ndarray]] = None,
            targets: Optional[Sequence[str]] = None) -> List[Dict]:
    """
    One fold per target, in the order the ladders are given.

    Args:
        targets: models to hold out. None holds out every model in turn. A
            ladder that is never a target only ever serves as a source -- the
            way already-characterized models enter a card that measures one
            new model.
    """
    names = [l.model for l in ladders]
    if targets is not None:
        unknown = sorted(set(targets) - set(names))
        if unknown:
            raise ValueError(f'targets without a ladder: {unknown}')
    folds = []
    for i, target in enumerate(ladders):
        if targets is not None and target.model not in targets:
            continue
        sources = [l for j, l in enumerate(ladders) if j != i]
        folds.append(run_fold(target, sources, calibration_rungs,
                              predict_rungs, weights=weights))
    return folds


def _problem_universe(ladders):
    universe = sorted({pid for l in ladders for pid in l.problem_ids})
    index = {pid: i for i, pid in enumerate(universe)}
    maps = {l.model: np.array([index[p] for p in l.problem_ids], dtype=int)
            for l in ladders}
    return universe, maps


def bootstrap_loo(ladders: Sequence[Ladder], n_boot: int, seed: int,
                  calibration_rungs=(0, 1), predict_rungs=(2, 3, 4),
                  progress: bool = False,
                  targets: Optional[Sequence[str]] = None) -> Dict:
    """
    Signed fold errors under a joint problem bootstrap.

    Returns:
        dict: ``errors[model][T]`` is an array of ``n_boot`` signed errors in
            points (NaN where a replicate's fold failed or the target drew no
            problems), plus ``n_boot`` and ``seed``.
    """
    universe, maps = _problem_universe(ladders)
    rng = np.random.default_rng(seed)
    scored = [l for l in ladders if targets is None or l.model in targets]
    errors = {l.model: {int(T): np.full(n_boot, np.nan) for T in predict_rungs}
              for l in scored}
    invalid = {l.model: 0 for l in scored}
    for r in range(n_boot):
        draw = rng.integers(0, len(universe), size=len(universe))
        counts = np.bincount(draw, minlength=len(universe)).astype(float)
        weights = {model: counts[idx] for model, idx in maps.items()}
        folds = run_loo(ladders, calibration_rungs, predict_rungs, weights,
                        targets=targets)
        for fold in folds:
            if not fold['valid']:
                invalid[fold['target']] += 1
                continue
            for T in predict_rungs:
                errors[fold['target']][int(T)][r] = fold['rungs'][int(T)]['err_pp']
        if progress and (r + 1) % max(1, n_boot // 20) == 0:
            print(f'[route_model] bootstrap {r + 1}/{n_boot}', flush=True)
    return {'errors': errors, 'invalid': invalid, 'n_boot': n_boot,
            'seed': seed}


def _interval(samples, alpha):
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return [float('nan'), float('nan')], 0
    lo, hi = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return [float(lo), float(hi)], int(finite.size)


def summarize(folds: List[Dict], boot: Optional[Dict], *, tolerance_pp: float,
              min_models_pass: int, primary_rungs=(2, 3),
              secondary_rungs=(4,), confidence: float = 0.95) -> Dict:
    """
    Reduce folds and bootstrap to the card's terminal result.

    The verdict itself is the claim's to make; this reports the same rule so
    the artifact is readable on its own.
    """
    alpha = 1.0 - confidence
    n_models = len(folds)
    m_primary = n_models * len(primary_rungs)
    m_secondary = n_models * len(secondary_rungs)

    per_model = {}
    all_abs, primary_abs = [], []
    n_pass = n_supported = 0
    for fold in folds:
        model = fold['target']
        entry = {
            'valid': fold['valid'], 'detail': fold['detail'],
            'n_problems': fold['n_problems'], 'baseline_acc': fold['baseline_acc'],
            'gamma': fold['gamma'], 'a': fold['a'], 'b': fold['b'],
        }
        model_pass = fold['valid']
        model_support = fold['valid'] and boot is not None
        for T, rung in sorted(fold['rungs'].items()):
            entry[f'obs_t{T}'] = rung['obs']
            entry[f'pred_t{T}'] = rung['pred']
            if rung['role'] != 'prediction':
                continue
            err = rung['err_pp']
            entry[f'err_t{T}'] = err
            if math.isfinite(err):
                all_abs.append(abs(err))
                if T in primary_rungs:
                    primary_abs.append(abs(err))
            if T in primary_rungs:
                model_pass = model_pass and math.isfinite(err) and abs(err) <= tolerance_pp
            if boot is not None:
                family = m_primary if T in primary_rungs else m_secondary
                ci, n_ok = _interval(boot['errors'][model][T], alpha / family)
                entry[f'ci_t{T}'] = ci
                entry[f'boot_ok_t{T}'] = n_ok
                inside = (math.isfinite(ci[0]) and -tolerance_pp <= ci[0]
                          and ci[1] <= tolerance_pp)
                if T in primary_rungs:
                    model_support = model_support and inside
        entry['pass'] = bool(model_pass)
        entry['supported'] = bool(model_support)
        if boot is not None:
            entry['boot_invalid'] = boot['invalid'][model]
        n_pass += int(model_pass)
        n_supported += int(model_support)
        per_model[model] = entry

    n_pred = sum(1 for f in folds for r in f['rungs'].values()
                 if r['role'] == 'prediction')
    return {
        'n_models': n_models,
        'n_models_valid': sum(1 for f in folds if f['valid']),
        'n_models_pass': n_pass,
        'n_models_supported': n_supported,
        'evaluation_pass': n_pass >= min_models_pass,
        'tolerance_pp': tolerance_pp,
        'min_models_pass': min_models_pass,
        'n_predictions': n_pred,
        'n_predictions_scored': len(all_abs),
        'mae_pp': float(np.mean(all_abs)) if all_abs else float('nan'),
        'max_abs_err_pp': float(np.max(all_abs)) if all_abs else float('nan'),
        'frac_within_tol': (float(np.mean(np.asarray(all_abs) <= tolerance_pp))
                            if all_abs else float('nan')),
        'mae_pp_primary': float(np.mean(primary_abs)) if primary_abs else float('nan'),
        'bonferroni': {'confidence': confidence,
                       'primary_family': m_primary,
                       'secondary_family': m_secondary},
        'per_model': per_model,
    }
