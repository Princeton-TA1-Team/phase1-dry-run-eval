"""
The route model of contextual drag and its maximum-likelihood fits.

A final answer is a choice among routes. The model either adopts an answer
already in its context or sets the drafts aside and solves independently.
Routes are taken with probability proportional to their weights. With ``K``
drafts of which ``T`` are correct, wrong drafts falling into answer classes
with multiplicities ``lambda = (n_1, ..., n_r)``, and baseline accuracy ``q``
(the probability of solving the problem with no drafts),

    F_K(T, lambda, q; a, b, gamma) = (a T^gamma + q) / (a T^gamma + b sum_j n_j^gamma + 1)

``a`` and ``b`` are the weights of one correct and one incorrect draft relative
to the independent route, which has unit weight. ``gamma`` says how agreement
compounds. Absent routes have zero weight, so ``T = 0`` contributes nothing
to the correct mass and an empty class contributes nothing to the wrong mass,
whatever ``gamma`` is.

Everything here is pure numpy/scipy over a :class:`Ladder`, so it can be unit
tested without a model, a GPU, or a dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    'Ladder', 'route_probability', 'fit_sources', 'fit_target',
    'predict_accuracy', 'observed_accuracy', 'TargetFit', 'SourceFit',
]

#: Probabilities are clipped away from 0 and 1 inside the likelihood only, so
#: a baseline of exactly 0 or 1 cannot produce log(0).
_EPS = 1e-9

#: Box for the log-weights of source models. exp(+-20) is far outside any
#: weight seen in practice; the bound only keeps the optimizer finite.
_LOG_BOUND = 20.0
_GAMMA_BOUNDS = (0.0, 10.0)


@dataclass
class Ladder:
    """
    One model's draft-mixture measurements on its eligible problems.

    A *context* is one draft set: a problem and a rung ``T`` with its fixed
    drafts, shown in several orders. Responses to the orders are pooled into
    ``k`` correct out of ``n``.

    Attributes:
        model (str): the model these measurements belong to.
        problem_ids (List[str]): eligible problems, one entry each.
        q (ndarray): baseline accuracy per problem, shape (P,).
        ctx_problem (ndarray): problem index of each context, shape (C,).
        ctx_T (ndarray): correct drafts in each context, shape (C,).
        ctx_wrong (ndarray): wrong-answer class multiplicities, zero padded,
            shape (C, Kmax).
        ctx_k (ndarray): correct responses per context, shape (C,).
        ctx_n (ndarray): responses per context, shape (C,).
    """
    model: str
    problem_ids: List[str]
    q: np.ndarray
    ctx_problem: np.ndarray
    ctx_T: np.ndarray
    ctx_wrong: np.ndarray
    ctx_k: np.ndarray
    ctx_n: np.ndarray
    extra: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.q = np.asarray(self.q, dtype=float)
        self.ctx_problem = np.asarray(self.ctx_problem, dtype=int)
        self.ctx_T = np.asarray(self.ctx_T, dtype=int)
        self.ctx_wrong = np.atleast_2d(np.asarray(self.ctx_wrong, dtype=float))
        self.ctx_k = np.asarray(self.ctx_k, dtype=float)
        self.ctx_n = np.asarray(self.ctx_n, dtype=float)
        n_ctx = len(self.ctx_T)
        if self.ctx_wrong.shape[0] != n_ctx and n_ctx == 0:
            self.ctx_wrong = np.zeros((0, 1))
        for name in ('ctx_problem', 'ctx_wrong', 'ctx_k', 'ctx_n'):
            if len(getattr(self, name)) != n_ctx:
                raise ValueError(f'{name} has {len(getattr(self, name))} '
                                 f'rows, expected {n_ctx}')
        if len(self.q) != len(self.problem_ids):
            raise ValueError('q and problem_ids differ in length')

    @property
    def rungs(self) -> List[int]:
        return sorted(set(int(t) for t in self.ctx_T))

    def select(self, rungs: Sequence[int]) -> np.ndarray:
        """Boolean mask over contexts whose rung is in ``rungs``."""
        return np.isin(self.ctx_T, list(rungs))


def _masses(T, wrong, gamma):
    """
    Unweighted correct and wrong masses and their gamma-derivatives.

    ``T ** gamma`` is replaced by 0 where ``T == 0`` and each ``n ** gamma`` by
    0 where ``n == 0``: an absent route has no weight even at ``gamma = 0``,
    where ``0 ** 0`` would otherwise count it once.
    """
    T = np.asarray(T, dtype=float)
    wrong = np.asarray(wrong, dtype=float)
    present_T = T > 0
    safe_T = np.where(present_T, T, 1.0)
    cT = np.where(present_T, safe_T ** gamma, 0.0)
    dcT = np.where(present_T, cT * np.log(safe_T), 0.0)

    present_w = wrong > 0
    safe_w = np.where(present_w, wrong, 1.0)
    w_terms = np.where(present_w, safe_w ** gamma, 0.0)
    S = w_terms.sum(axis=-1)
    dS = np.where(present_w, w_terms * np.log(safe_w), 0.0).sum(axis=-1)
    return cT, dcT, S, dS


def route_probability(T, wrong, q, a, b, gamma):
    """
    Probability of a correct final answer under the route model.

    Args:
        T (int | ndarray): correct drafts in context.
        wrong (ndarray): wrong-answer class multiplicities, (..., r).
        q (float | ndarray): baseline accuracy.
        a, b (float): correct- and incorrect-draft weights.
        gamma (float): agreement exponent.

    Returns:
        ndarray

    Example:
        >>> # no drafts at all leaves the baseline untouched
        >>> float(route_probability(0, [0, 0, 0, 0], 0.4, 2.0, 1.0, 1.3))
        0.4
        >>> # four correct drafts with gamma = 1: (4a + q) / (4a + 1)
        >>> round(float(route_probability(4, [0, 0, 0, 0], 0.5, 1.0, 1.0, 1.0)), 6)
        0.9
        >>> # gamma = 0 counts distinct answers: two wrong classes, b = 1, q = 0.6
        >>> round(float(route_probability(0, [3, 1, 0, 0], 0.6, 1.0, 1.0, 0.0)), 6)
        0.2
    """
    cT, _, S, _ = _masses(T, wrong, gamma)
    q = np.asarray(q, dtype=float)
    cm = a * cT
    return (cm + q) / (cm + b * S + 1.0)


def _nll_and_grad(ladder, mask, weights, a, b, gamma, want):
    """
    Negative binomial log-likelihood of the masked contexts, with gradient.

    Args:
        want (str): which gradient to return: ``'log'`` for (d/dlog a,
            d/dlog b, d/dgamma) or ``'raw'`` for (d/da, d/db).
    """
    T = ladder.ctx_T[mask]
    wrong = ladder.ctx_wrong[mask]
    k = ladder.ctx_k[mask]
    n = ladder.ctx_n[mask]
    q = ladder.q[ladder.ctx_problem[mask]]
    w = np.ones_like(k) if weights is None else weights[ladder.ctx_problem[mask]]

    cT, dcT, S, dS = _masses(T, wrong, gamma)
    cm = a * cT
    wm = b * S
    u = cm + q
    v = cm + wm + 1.0
    F = np.clip(u / v, _EPS, 1.0 - _EPS)
    nll = -np.sum(w * (k * np.log(F) + (n - k) * np.log1p(-F)))

    dl_dF = w * (k / F - (n - k) / (1.0 - F))
    dF_dcm = (v - u) / v ** 2
    dF_dwm = -u / v ** 2
    if want == 'log':
        g_loga = -np.sum(dl_dF * dF_dcm * cm)
        g_logb = -np.sum(dl_dF * dF_dwm * wm)
        g_gamma = -np.sum(dl_dF * (dF_dcm * a * dcT + dF_dwm * b * dS))
        return nll, np.array([g_loga, g_logb, g_gamma])
    g_a = -np.sum(dl_dF * dF_dcm * cT)
    g_b = -np.sum(dl_dF * dF_dwm * S)
    return nll, np.array([g_a, g_b])


@dataclass
class SourceFit:
    """A shared exponent and per-source weights."""
    gamma: float
    weights: Dict[str, Dict[str, float]]
    nll: float
    converged: bool


@dataclass
class TargetFit:
    """A target's weights at a fixed exponent."""
    a: float
    b: float
    gamma: float
    nll: float
    converged: bool
    negative_weight: bool
    detail: str = ''


def fit_sources(ladders: Sequence[Ladder], weights: Optional[Dict[str, np.ndarray]] = None,
                rungs: Optional[Sequence[int]] = None,
                gamma_starts: Sequence[float] = (1.0, 2.0),
                maxiter: int = 500) -> SourceFit:
    """
    Joint maximum likelihood over source models: one gamma, own (a, b) each.

    Args:
        ladders: the source models' full measurements.
        weights: optional per-problem weights per model (bootstrap
            multiplicities). Missing means 1 for every problem.
        rungs: rungs to fit on; None uses every rung present.
        gamma_starts: starting exponents; the best optimum is kept.

    Returns:
        SourceFit
    """
    from scipy.optimize import minimize

    weights = weights or {}
    masks = []
    for ladder in ladders:
        mask = ladder.select(rungs) if rungs is not None else np.ones(len(ladder.ctx_T), bool)
        w = weights.get(ladder.model)
        if w is not None:
            mask = mask & (w[ladder.ctx_problem] > 0)
        masks.append(mask)
    n_models = len(ladders)

    def objective(theta):
        gamma = theta[-1]
        total = 0.0
        grad = np.zeros_like(theta)
        for j, (ladder, mask) in enumerate(zip(ladders, masks)):
            if not mask.any():
                continue
            a = np.exp(theta[2 * j])
            b = np.exp(theta[2 * j + 1])
            nll, g = _nll_and_grad(ladder, mask, weights.get(ladder.model),
                                   a, b, gamma, 'log')
            total += nll
            grad[2 * j] += g[0]
            grad[2 * j + 1] += g[1]
            grad[-1] += g[2]
        return total, grad

    bounds = [(-_LOG_BOUND, _LOG_BOUND)] * (2 * n_models) + [_GAMMA_BOUNDS]
    best = None
    for gamma0 in gamma_starts:
        theta0 = np.zeros(2 * n_models + 1)
        theta0[-1] = gamma0
        result = minimize(objective, theta0, jac=True, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': maxiter})
        if best is None or result.fun < best.fun:
            best = result
    theta = best.x
    fitted = {
        ladder.model: {'a': float(np.exp(theta[2 * j])),
                       'b': float(np.exp(theta[2 * j + 1]))}
        for j, ladder in enumerate(ladders)
    }
    return SourceFit(gamma=float(theta[-1]), weights=fitted,
                     nll=float(best.fun), converged=bool(best.success))


def fit_target(ladder: Ladder, gamma: float, rungs: Sequence[int] = (0, 1),
               weights: Optional[np.ndarray] = None, start=(1.0, 1.0),
               maxiter: int = 500, boundary_tol: float = 1e-8) -> TargetFit:
    """
    The target's weights from its calibration rungs, gamma held fixed.

    Weights are fit on their natural scale with a lower bound of zero, rather
    than on the log scale, so that a calibration which pushes a weight below
    zero shows up as a boundary optimum with an outward gradient instead of
    as a weight that merely becomes very small. That is the case the plan
    reports as a failed prediction: the target's calibration numbers imply a
    negative weight, and the route model cannot represent them.

    Args:
        ladder: the target model's measurements.
        gamma: the exponent estimated from the sources.
        rungs: calibration rungs, (0, 1) in the plan.
        weights: optional per-problem bootstrap multiplicities.

    Returns:
        TargetFit
    """
    from scipy.optimize import minimize

    mask = ladder.select(rungs)
    if weights is not None:
        mask = mask & (weights[ladder.ctx_problem] > 0)
    if not mask.any():
        return TargetFit(a=float('nan'), b=float('nan'), gamma=gamma,
                         nll=float('nan'), converged=False,
                         negative_weight=False,
                         detail='no calibration contexts')

    def objective(theta):
        return _nll_and_grad(ladder, mask, weights, theta[0], theta[1],
                             gamma, 'raw')

    result = minimize(objective, np.asarray(start, float), jac=True,
                      method='L-BFGS-B', bounds=[(0.0, 1e6), (0.0, 1e6)],
                      options={'maxiter': maxiter})
    a, b = (float(x) for x in result.x)
    _, grad = objective(result.x)
    # A weight pinned at zero whose NLL still falls as the weight decreases
    # is a weight the data want to be negative.
    scale = max(1.0, float(np.sum(ladder.ctx_n[mask])))
    negative = []
    if a <= boundary_tol and grad[0] > 1e-6 * scale:
        negative.append('a')
    if b <= boundary_tol and grad[1] > 1e-6 * scale:
        negative.append('b')
    detail = ''
    if negative:
        detail = ('calibration implies a negative weight for '
                  + ' and '.join(negative))
    return TargetFit(a=a, b=b, gamma=gamma, nll=float(result.fun),
                     converged=bool(result.success),
                     negative_weight=bool(negative), detail=detail)


def _weighted_mean(values, problem_index, weights):
    if weights is None:
        return float(np.mean(values)) if len(values) else float('nan')
    w = weights[problem_index]
    total = w.sum()
    return float(np.sum(w * values) / total) if total > 0 else float('nan')


def predict_accuracy(ladder: Ladder, T: int, a: float, b: float, gamma: float,
                     weights: Optional[np.ndarray] = None) -> float:
    """
    Predicted accuracy at rung ``T``: the problem average of F over that
    rung's actual draft sets.
    """
    mask = ladder.ctx_T == T
    if not mask.any():
        return float('nan')
    F = route_probability(T, ladder.ctx_wrong[mask],
                          ladder.q[ladder.ctx_problem[mask]], a, b, gamma)
    return _per_problem_mean(F, ladder.ctx_problem[mask], weights)


def observed_accuracy(ladder: Ladder, T: int,
                      weights: Optional[np.ndarray] = None) -> float:
    """
    Measured accuracy at rung ``T``: averaged over orders, then problems.
    """
    mask = ladder.ctx_T == T
    if not mask.any():
        return float('nan')
    rate = ladder.ctx_k[mask] / ladder.ctx_n[mask]
    return _per_problem_mean(rate, ladder.ctx_problem[mask], weights)


def _per_problem_mean(values, problem_index, weights):
    """
    Average within problem first, then across problems.

    A rung holds one draft set per problem, so within-problem averaging is a
    no-op today; it is done anyway so a ladder with several draft sets per
    rung cannot weight its problems unevenly.
    """
    uniq, inverse = np.unique(problem_index, return_inverse=True)
    per = np.bincount(inverse, weights=values) / np.bincount(inverse)
    return _weighted_mean(per, uniq, weights)
