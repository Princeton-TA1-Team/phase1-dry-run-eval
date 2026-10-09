"""Unit tests for the route model, its fits, and the leave-one-model-out rule."""
import json

import numpy as np
import pytest

pytest.importorskip('scipy')

from contextual_drag.analysis.route_model.io import (  # noqa: E402
    SCHEMA, ladder_from_payload, wrong_classes, write_ladder, read_ladder,
)
from contextual_drag.analysis.route_model.loo import (  # noqa: E402
    bootstrap_loo, run_loo, summarize,
)
from contextual_drag.analysis.route_model.model import (  # noqa: E402
    Ladder, fit_sources, fit_target, observed_accuracy, predict_accuracy,
    route_probability,
)


def _synthetic(model, a, b, gamma, n_problems, rng, n_orders=8, exact=False):
    """A ladder drawn from the route model itself."""
    q = rng.uniform(0.25, 0.75, n_problems)
    rows = {'p': [], 'T': [], 'w': [], 'k': [], 'n': []}
    for i in range(n_problems):
        for T in range(5):
            cls = rng.integers(0, 3, 4 - T)
            counts = np.sort(np.bincount(cls, minlength=4)[:4])[::-1]
            F = route_probability(T, counts, q[i], a, b, gamma)
            rows['p'].append(i)
            rows['T'].append(T)
            rows['w'].append(counts)
            rows['k'].append(F * n_orders if exact else rng.binomial(n_orders, F))
            rows['n'].append(n_orders)
    return Ladder(model, [f'p{i}' for i in range(n_problems)], q, rows['p'],
                  rows['T'], np.array(rows['w']), rows['k'], rows['n'])


def test_route_probability_limits():
    # an empty context returns the baseline whatever the weights
    assert route_probability(0, [0, 0, 0, 0], 0.3, 5.0, 5.0, 2.0) == pytest.approx(0.3)
    # gamma = 0 counts distinct answers, not drafts
    assert route_probability(0, [3, 1], 0.6, 1.0, 1.0, 0.0) == pytest.approx(0.6 / 3)
    # gamma = 1 counts drafts
    assert route_probability(0, [3, 1], 0.6, 1.0, 1.0, 1.0) == pytest.approx(0.6 / 5)
    # all-correct context approaches 1 as the correct weight grows
    assert route_probability(4, [0], 0.0, 1e6, 1.0, 1.0) == pytest.approx(1.0, abs=1e-5)


def test_fits_recover_parameters_on_noiseless_ladders():
    rng = np.random.default_rng(0)
    truth = {'m1': (3.0, 1.2), 'm2': (2.0, 0.8), 'm3': (4.0, 1.5)}
    ladders = [_synthetic(m, a, b, 1.4, 40, rng, exact=True)
               for m, (a, b) in truth.items()]
    fit = fit_sources(ladders)
    assert fit.gamma == pytest.approx(1.4, abs=1e-3)
    for m, (a, b) in truth.items():
        assert fit.weights[m]['a'] == pytest.approx(a, rel=1e-3)
        assert fit.weights[m]['b'] == pytest.approx(b, rel=1e-3)
    target = fit_target(ladders[0], 1.4)
    assert not target.negative_weight
    assert target.a == pytest.approx(3.0, rel=1e-3)
    assert target.b == pytest.approx(1.2, rel=1e-3)


def test_loo_is_exact_when_the_model_is_true():
    rng = np.random.default_rng(1)
    ladders = [_synthetic(f'm{i}', a, b, 1.3, 30, rng, exact=True)
               for i, (a, b) in enumerate([(3, 1), (2, .7), (4, 1.6), (1.5, 1.1)])]
    folds = run_loo(ladders)
    for fold in folds:
        assert fold['valid']
        for T in (2, 3, 4):
            assert abs(fold['rungs'][T]['err_pp']) < 1e-2
    summary = summarize(folds, None, tolerance_pp=5.0, min_models_pass=2)
    assert summary['n_models_pass'] == 4
    assert summary['evaluation_pass']
    assert summary['max_abs_err_pp'] < 1e-2


def test_negative_weight_is_a_failed_fold():
    rng = np.random.default_rng(2)
    good = [_synthetic(f'm{i}', 2.0, 1.0, 1.4, 20, rng) for i in range(3)]
    bad = _synthetic('bad', 2.0, 1.0, 1.4, 20, rng)
    # wrong drafts that HELP: accuracy with four wrong drafts above baseline
    bad.q[:] = 0.3
    bad.ctx_k = np.where(bad.ctx_T == 0, bad.ctx_n, bad.ctx_k)
    fit = fit_target(bad, 1.4)
    assert fit.negative_weight
    folds = run_loo(good + [bad])
    bad_fold = [f for f in folds if f['target'] == 'bad'][0]
    assert not bad_fold['valid']
    summary = summarize(folds, None, tolerance_pp=5.0, min_models_pass=2)
    assert summary['per_model']['bad']['pass'] is False


def test_observed_accuracy_averages_orders_then_problems():
    ladder = Ladder('m', ['a', 'b'], [0.5, 0.5], [0, 1], [2, 2],
                    [[1, 1], [2, 0]], [8, 0], [8, 4])
    # problem a: 8/8, problem b: 0/4 -> mean of rates, not pooled 8/12
    assert observed_accuracy(ladder, 2) == pytest.approx(0.5)
    weights = np.array([3.0, 1.0])
    assert observed_accuracy(ladder, 2, weights) == pytest.approx(0.75)
    assert np.isnan(predict_accuracy(ladder, 3, 1.0, 1.0, 1.0))


def test_bootstrap_is_seeded_and_joint():
    rng = np.random.default_rng(3)
    ladders = [_synthetic(f'm{i}', 2.0 + i, 1.0, 1.2, 12, rng) for i in range(3)]
    one = bootstrap_loo(ladders, 4, seed=7)
    two = bootstrap_loo(ladders, 4, seed=7)
    for m in one['errors']:
        for T in one['errors'][m]:
            np.testing.assert_array_equal(one['errors'][m][T], two['errors'][m][T])
    summary = summarize(run_loo(ladders), one, tolerance_pp=5.0,
                        min_models_pass=2)
    assert summary['bonferroni']['primary_family'] == 6
    assert summary['bonferroni']['secondary_family'] == 3
    assert len(summary['per_model']['m0']['ci_t2']) == 2


def test_ladder_file_round_trip(tmp_path):
    payload = {
        'schema': SCHEMA, 'model': 'X', 'K': 4,
        'problems': [{'id': 'aime24:2', 'pool_correct': 7, 'pool_n': 16,
                      'baseline_correct': 4, 'baseline_n': 16}],
        'contexts': [{'problem_id': 'aime24:2', 'T': 1,
                      'draft_answers': ['204', '17', '17', None],
                      'draft_correct': [True, False, False, False],
                      'wrong_classes': wrong_classes(
                          ['204', '17', '17', None], [True, False, False, False]),
                      'n': 8, 'k': 3}],
    }
    fpath = tmp_path / 'ladder.json'
    write_ladder(fpath, payload)
    ladder = read_ladder(fpath)
    assert ladder.model == 'X'
    assert ladder.q[0] == pytest.approx(0.25)
    assert list(ladder.ctx_wrong[0]) == [2, 1]
    with pytest.raises(ValueError):
        ladder_from_payload(dict(payload, schema='other'))


def test_cli_writes_strict_json(tmp_path):
    from contextual_drag.analysis.route_model.cli import run_from_files
    rng = np.random.default_rng(4)
    files = []
    for i in range(3):
        ladder = _synthetic(f'm{i}', 2.0 + i, 1.0, 1.2, 10, rng)
        payload = {
            'schema': SCHEMA, 'model': ladder.model, 'K': 4,
            'problems': [{'id': pid, 'pool_correct': 8, 'pool_n': 16,
                          'baseline_correct': int(round(q * 16)), 'baseline_n': 16}
                         for pid, q in zip(ladder.problem_ids, ladder.q)],
            'contexts': [{'problem_id': ladder.problem_ids[p], 'T': int(T),
                          'wrong_classes': [int(x) for x in w if x > 0],
                          'n': int(n), 'k': int(k)}
                         for p, T, w, n, k in zip(ladder.ctx_problem, ladder.ctx_T,
                                                  ladder.ctx_wrong, ladder.ctx_n,
                                                  ladder.ctx_k)],
        }
        fpath = tmp_path / f'{i}.json'
        write_ladder(fpath, payload)
        files.append(fpath)
    result = run_from_files(files, calibration_rungs=(0, 1), primary_rungs=(2, 3),
                            secondary_rungs=(4,), tolerance_pp=5.0,
                            min_models_pass=2, n_boot=3, seed=1, confidence=0.95)
    text = json.dumps(result, allow_nan=False)
    assert '"per_model"' in text
    assert result['config']['models'] == ['m0', 'm1', 'm2']


def test_targets_score_only_the_held_out_model():
    rng = np.random.default_rng(5)
    ladders = [_synthetic(f'm{i}', a, b, 1.3, 25, rng, exact=True)
               for i, (a, b) in enumerate([(3, 1), (2, .7), (4, 1.6), (1.5, 1.1)])]
    folds = run_loo(ladders, targets=['m1'])
    assert [f['target'] for f in folds] == ['m1']
    assert folds[0]['sources'] == ['m0', 'm2', 'm3']
    full = {f['target']: f for f in run_loo(ladders)}
    assert folds[0]['rungs'][2]['err_pp'] == pytest.approx(full['m1']['rungs'][2]['err_pp'])
    boot = bootstrap_loo(ladders, 3, seed=1, targets=['m1'])
    assert set(boot['errors']) == {'m1'}
    summary = summarize(folds, boot, tolerance_pp=5.0, min_models_pass=1)
    assert summary['n_models'] == 1 and summary['bonferroni']['primary_family'] == 2
    with pytest.raises(ValueError):
        run_loo(ladders, targets=['nope'])
