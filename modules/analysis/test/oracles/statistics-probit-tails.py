# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Float64 probit maxima for fits whose linear predictor goes far into the tails, and for
separated data (statsmodels, SciPy), for the closing pass of the math correctness review
(2026-11-11, regression-01).

Writes fixtures/statistics-probit-tails.json next to this script, consumed by
test/gpu-spatial-analysis/spatial-regression/math-review-probit-bound.spec.ts
(`GPUProbitRegression`). Run from anywhere with
    <venv>/bin/python -I modules/analysis/test/oracles/statistics-probit-tails.py
in the pinned environment of requirements.txt: CPython 3.14, NumPy 2.5.4, SciPy 1.16.3,
statsmodels 0.15.0 (the versions are recorded in the fixture).

Scenes. One predictor and an intercept. The inputs come from a 32-bit linear congruential
stream and IEEE double arithmetic only, so the spec rebuilds the same inputs; the predictors are
rounded to float32 (what the GPU receives) and `predictorSum` is their float64 sum. A deviate is
the sum of twelve uniforms minus 6; the response is 1 when `intercept + slope x + deviate > 0`
(without the deviate for the separated scenes).
- `control`: 2000 rows, latent index `0.2 + 2 x`, seed 777.
- `correctOutlier`: `control` plus `(x = 9, y = 1)`, a correctly classified row whose linear
  predictor is 18.9 at the maximum.
- `misclassifiedOutlier`: `control` plus `(x = 4, y = 0)`, a row on the wrong side at 7.7.
- `grossMisclassifiedOutlier`: `control` plus `(x = 9, y = 0)`, on the wrong side at 13.2, where
  `1 - Phi` is below the float32 range.
- `justBeyondBound`: 400 rows, latent index `0.2 + 3.5 x`, seed 12345; the largest linear
  predictor at the maximum is 12.06, on a correctly classified row.
- `completeSeparation`: 400 rows, `y = 1` exactly when `0.2 + 3.5 x > 0`, seed 4242.
- `quasiCompleteSeparation`: the same plus six rows at `x = -0.2 / 3.5` with responses
  1, 0, 1, 0, 1, 0.
- `steepSlope`: 2000 rows, latent index `0.2 + 8 x`, seed 5: not separated, largest linear
  predictor 29.5 at the maximum (frc-regression-05 of the fix review: an unfinished fit of such
  data is status 3, not separation).

Scenes with several predictors (the review fixes of the closing pass, frc-regression-03):
quasi-separation by an indicator of a few rows. `x1` and `y1` are 600 rows of the latent index
`0.2 + x`, seed 31337; an indicator column is 1 on the listed rows and the response is set to 1
there, so the likelihood rises without bound in the indicator coefficient while every other
coefficient has a finite limit. `predictorSums` holds the sequential float64 sum of each
float32 predictor column.
- `indicatorOfThreeRows`: `x1` and the indicator of rows 17, 301, 555.
- `indicatorOfOneRow`: `x1` and the indicator of row 17.
- `indicatorAmongElevenPredictors`: `x1`, nine independent deviate columns (seeds 1000 to 1008)
  and the indicator of rows 17, 301, 555.
- `indicatorOfEveryTwentiethRow`, `indicatorOfEverySecondRow`: `x1` and that indicator.
- `indicatorAsMiddleOfThree`: `x1`, the indicator of rows 17, 301, 555, and a deviate column
  (seed 2024).

Large samples (the second round of review fixes, frc-regression-10): the same construction with
100,000 to 1,000,000 rows of the latent index `0.2 + x`, seed 31337, where the separated group
is 2e-5 of the rows or less. The rows of the indicator are the first of 17, 3001, 15555, 40000,
77777, or `13 + 9973 k` for `k` below 20.
- `indicatorOfOneRowIn100000`, `indicatorOfThreeRowsIn200000`, `indicatorOfFiveRowsIn400000`,
  `indicatorOfFiveRowsIn1000000`, `indicatorOfTwentyRowsIn1000000`: quasi-separated.
- `indicatorWithBothOutcomesIn100000`: the indicator of rows 17, 3001, 15555 with responses
  1, 1, 0 there: not separated, a finite maximum.
The large samples are built by `create_sample_array`, a NumPy form of `create_sample` (the
same stream by modular powers, the same order of additions; asserted equal on 600 rows).

Third round of review fixes (frc2-regression-03, -04 and -06), inputs written out, no stream:
- `nearlySeparated`: 1,000 rows `(x = -1, y = 0)`, 2,000 rows `(x = 1, y = 1)`, then
  `(float32(-1e-5), 1)` and `(float32(1e-5), 0)`. Not separated (the linear program is 0), with a
  finite maximum at a slope of 6.02; the first Fisher direction moves the two central rows
  towards the wrong side by 1.5e-5 of their own terms, which a tolerance took for separation.
- `largeOffsetPredictor`: five groups of 1,000 rows at `x = float32(2^24 + 2 z)`, `z = -2..2`,
  with 136, 345, 618, 841 and 955 ones (row `r` of a group is 1 when `37 r mod 1000` is below
  the count). The references are fitted on `x - 2^24`, an exact shift that moves only the
  intercept: `coefficients[0]` is the intercept of the shifted model, `coefficients[1]` the
  slope of either. `statsmodels.groupFitted` (`Probit.predict`) and
  `statsmodels.averageMarginalEffect` (`get_margeff(at='overall')`) do not depend on the shift.
  `spreg.diagnostics` are the Pinkse, Kelejian-Prucha and Pinkse-Slade tests of
  `spreg.Probit(spat_diag=True)` (spreg 1.9.1) on the ring of weights 0.5 over the row order;
  its coefficients, fitted values and `slopes` must agree with statsmodels (asserted).
- `threeValuedMiddle`: 40,000 rows `(0, 0)`, 60,000 rows `(0, 1)`, then `(-1, 0)` and `(1, 1)`:
  quasi-separated by the direction `(intercept, slope) = (0, 1)`.

References.
- `separationObjective`: an oracle for separation that does not fit anything. The linear program
  `max sum_i s_i x_i'd` subject to `s_i x_i'd >= 0` for every row and `|d_j| <= 1` (columns
  scaled to unit largest magnitude, `scipy.optimize.linprog`, HiGHS), with `s_i = 2 y_i - 1`.
  Its optimum is 0 exactly when no direction moves every row to the right side, that is when
  the data are not separated; a positive optimum is complete or quasi-complete separation. The
  generator asserts 0 on every finite scene and a positive value on every separated one.
- `coefficients`, `logLikelihood`: a Newton solve (observed information, halved steps) of the
  tail-exact likelihood `sum log_ndtr((2 y - 1) eta)` with `scipy.special.log_ndtr`.
- `standardErrors`: the expected information `X' diag(lambda(eta) lambda(-eta)) X` at that
  maximum, `lambda(t) = phi(t) / Phi(t)`, which is what `GPUProbitRegression` reports
  (statsmodels reports observed-information standard errors, which differ).
- `statsmodels`: `sm.Probit(y, X).fit(method='newton', maxiter=200, tol=1e-12)`. It must agree
  with the tail-exact solve to 1e-9 in the coefficients on every finite scene but the gross
  outlier (asserted). There statsmodels clips probabilities at 2.2e-16, so the misclassified row
  has a flat likelihood and it returns the `control` fit, which is not the maximum; the
  tail-exact solve is the reference. On the separated scenes statsmodels does not converge and
  its coefficients run away; only that fact is stored.
- `statsmodels.expectedInformationStandardErrors`: `sm.GLM(y, X, Binomial(Probit))` fitted by
  IRLS, whose `bse` are expected-information standard errors, the library's own value of
  `standardErrors`. Stored where statsmodels is at the maximum; it must agree with
  `standardErrors` to 1e-6 (asserted), and to 2e-5 on `misclassifiedOutlier`, where the
  library's probit link clips the probability of the row at a linear predictor of 7.7.
"""

import contextlib
import io
import json
import os
import sys
import warnings

import libpysal
import numpy as np
import scipy
import spreg
import statsmodels
import statsmodels.api as sm
from scipy.optimize import linprog
from scipy.special import log_ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE_PATH = os.path.join(HERE, 'fixtures', 'statistics-probit-tails.json')
HALF_LOG_TWO_PI = 0.5 * np.log(2 * np.pi)


def create_sample(row_count, intercept, slope, seed, noisy=True):
    state = [seed & 0xFFFFFFFF]

    def get_uniform():
        state[0] = (state[0] * 1664525 + 1013904223) & 0xFFFFFFFF
        return (state[0] + 0.5) / 4294967296.0

    def get_deviate():
        total = 0.0
        for _ in range(12):
            total += get_uniform()
        return total - 6.0

    predictor = []
    response = []
    for _ in range(row_count):
        value = get_deviate()
        predictor.append(value)
        latent = intercept + slope * value + (get_deviate() if noisy else 0.0)
        response.append(1.0 if latent > 0 else 0.0)
    return predictor, response


def create_sample_array(row_count, intercept, slope, seed):
    """`create_sample` (noisy) for large samples: the same bits, as float64 arrays."""
    count = 24 * row_count
    multiplier = np.full(count, 1664525, dtype=np.uint32)
    multiplier[0] = 1
    # powers[k] = 1664525^k and offsets[k] = sum_{i < k} 1664525^i, both modulo 2^32.
    powers = np.multiply.accumulate(multiplier, dtype=np.uint32)
    offsets = np.concatenate([np.zeros(1, dtype=np.uint32), np.add.accumulate(powers[:-1], dtype=np.uint32)])
    # state[k] is the state after k + 1 steps of state <- state * 1664525 + 1013904223.
    first = np.uint32(((seed & 0xFFFFFFFF) * 1664525 + 1013904223) & 0xFFFFFFFF)
    states = powers * first + np.uint32(1013904223) * offsets
    uniforms = ((states.astype(np.float64) + 0.5) / 4294967296.0).reshape(row_count, 24)
    predictor = np.zeros(row_count)
    deviate = np.zeros(row_count)
    for term in range(12):
        predictor = predictor + uniforms[:, term]
        deviate = deviate + uniforms[:, 12 + term]
    predictor = predictor - 6.0
    latent = intercept + slope * predictor + (deviate - 6.0)
    return predictor, np.where(latent > 0, 1.0, 0.0)


def get_design(predictor):
    """Intercept and float32 predictors; `predictor` is one column or a list of columns."""
    columns = predictor if isinstance(predictor[0], (list, np.ndarray)) else [predictor]
    values = [np.array(column, dtype=np.float32).astype(np.float64) for column in columns]
    return np.column_stack([np.ones_like(values[0])] + values)


def get_separation_objective(design, response):
    """Optimum of `max sum s_i x_i'd` over `s_i x_i'd >= 0`, `|d_j| <= 1`: 0 if not separated."""
    signed = design * (2 * response - 1)[:, None]
    signed = signed / np.abs(signed).max(axis=0)
    result = linprog(
        -signed.sum(axis=0),
        A_ub=-signed,
        b_ub=np.zeros(len(response)),
        bounds=[(-1, 1)] * signed.shape[1],
        method='highs',
    )
    assert result.status == 0, result.message
    return float(max(-result.fun, 0.0)) + 0.0


def get_expected_information_standard_errors(design, response):
    """statsmodels GLM (IRLS) standard errors of the probit fit: expected information."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = sm.GLM(
            response, design, family=sm.families.Binomial(link=sm.families.links.Probit())
        ).fit(tol=1e-12, maxiter=200)
    return result.bse.tolist()


def get_inverse_mills_ratio(value):
    """phi(value) / Phi(value), accurate in the lower tail."""
    return np.exp(-0.5 * value * value - HALF_LOG_TWO_PI - log_ndtr(value))


def fit_tail_exact(design, response):
    side = 2 * response - 1
    coefficients = np.zeros(design.shape[1])
    for _ in range(200):
        signed_linear = side * (design @ coefficients)
        ratio = get_inverse_mills_ratio(signed_linear)
        gradient = design.T @ (side * ratio)
        observed_information = (design * (ratio * (signed_linear + ratio))[:, None]).T @ design
        step = np.linalg.solve(observed_information, gradient)
        current = np.sum(log_ndtr(signed_linear))
        scale = 1.0
        while (
            np.sum(log_ndtr(side * (design @ (coefficients + scale * step)))) < current
            and scale > 1e-12
        ):
            scale *= 0.5
        coefficients = coefficients + scale * step
        if np.max(np.abs(scale * step)) < 1e-13:
            break
    linear = design @ coefficients
    signed_linear = side * linear
    expected_information = (
        design * (get_inverse_mills_ratio(linear) * get_inverse_mills_ratio(-linear))[:, None]
    ).T @ design
    return {
        'coefficients': coefficients.tolist(),
        'standardErrors': np.sqrt(np.diag(np.linalg.inv(expected_information))).tolist(),
        'logLikelihood': float(np.sum(log_ndtr(signed_linear))),
        'largestLinear': float(np.max(np.abs(linear))),
        'largestMisclassifiedLinear': float(np.max(np.where(signed_linear < 0, np.abs(linear), 0.0))),
    }


def fit_statsmodels(design, response):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = sm.Probit(response, design).fit(method='newton', maxiter=200, tol=1e-12, disp=0)
    return {
        'coefficients': result.params.tolist(),
        'logLikelihood': float(result.llf),
        'converged': bool(result.mle_retvals['converged']),
        'warnings': sorted({type(warning.message).__name__ for warning in caught}),
    }


def get_scene(predictor, response, finite, statsmodels_is_maximum=True, information_tolerance=1e-6):
    design = get_design(predictor)
    outcomes = np.array(response)
    scene = {'rowCount': len(response), 'predictorSum': float(np.sum(design[:, 1]))}
    if design.shape[1] > 2:
        # Sequential sums, column by column, as the spec forms them.
        scene['predictorSums'] = [
            sum(float(value) for value in design[:, column]) for column in range(1, design.shape[1])
        ]
    scene['separationObjective'] = get_separation_objective(design, outcomes)
    library = fit_statsmodels(design, outcomes)
    if finite:
        assert scene['separationObjective'] == 0.0, scene['separationObjective']
        scene.update(fit_tail_exact(design, outcomes))
        difference = float(
            np.max(np.abs(np.array(library['coefficients']) - np.array(scene['coefficients'])))
        )
        if statsmodels_is_maximum:
            assert library['converged'] and difference < 1e-9, difference
            expected = get_expected_information_standard_errors(design, outcomes)
            error = float(np.max(np.abs(np.array(expected) / np.array(scene['standardErrors']) - 1)))
            assert error < information_tolerance, error
            library['expectedInformationStandardErrors'] = expected
        else:
            assert difference > 0.1, difference
        scene['statsmodels'] = library
    else:
        assert scene['separationObjective'] > 1e-6, scene['separationObjective']
        assert not library['converged']
        scene['statsmodels'] = {'converged': False, 'warnings': library['warnings']}
    return scene


def get_indicator(rows, row_count=600):
    return [1.0 if row in rows else 0.0 for row in range(row_count)]


LARGE_SAMPLE_ROWS = [17, 3001, 15555, 40000, 77777]


def get_large_sample_scenes():
    """Quasi-separation by an indicator of 2e-5 of the rows or less, and one finite twin."""
    check_predictor, check_response = create_sample(600, 0.2, 1.0, 31337)
    array_predictor, array_response = create_sample_array(600, 0.2, 1.0, 31337)
    assert array_predictor.tolist() == check_predictor and array_response.tolist() == check_response
    scenes = {}
    samples = {}

    def add(name, row_count, rows, outcomes, finite):
        if row_count not in samples:
            samples[row_count] = create_sample_array(row_count, 0.2, 1.0, 31337)
        predictor, response = samples[row_count]
        indicator = np.zeros(row_count)
        indicator[rows] = 1.0
        response = response.copy()
        response[rows] = outcomes
        scenes[name] = get_scene([predictor, indicator], response.tolist(), finite)

    add('indicatorOfOneRowIn100000', 100000, LARGE_SAMPLE_ROWS[:1], 1.0, False)
    add('indicatorOfThreeRowsIn200000', 200000, LARGE_SAMPLE_ROWS[:3], 1.0, False)
    add('indicatorOfFiveRowsIn400000', 400000, LARGE_SAMPLE_ROWS, 1.0, False)
    add('indicatorOfFiveRowsIn1000000', 1000000, LARGE_SAMPLE_ROWS, 1.0, False)
    add('indicatorOfTwentyRowsIn1000000', 1000000, [13 + 9973 * index for index in range(20)], 1.0, False)
    add('indicatorWithBothOutcomesIn100000', 100000, LARGE_SAMPLE_ROWS[:3], [1.0, 1.0, 0.0], True)
    return scenes


def get_indicator_scenes():
    """Quasi-separation by an indicator: the response is 1 wherever the indicator is."""
    predictor, response = create_sample(600, 0.2, 1.0, 31337)
    scenes = {}

    def add(name, indicator, columns):
        outcomes = [1.0 if indicator[row] else response[row] for row in range(600)]
        scenes[name] = get_scene(columns, outcomes, False)

    three = get_indicator({17, 301, 555})
    add('indicatorOfThreeRows', three, [predictor, three])
    one = get_indicator({17})
    add('indicatorOfOneRow', one, [predictor, one])
    noise = [create_sample(600, 0.0, 1.0, 1000 + column)[0] for column in range(9)]
    add('indicatorAmongElevenPredictors', three, [predictor] + noise + [three])
    twentieth = get_indicator(set(range(0, 600, 20)))
    add('indicatorOfEveryTwentiethRow', twentieth, [predictor, twentieth])
    second = get_indicator(set(range(0, 600, 2)))
    add('indicatorOfEverySecondRow', second, [predictor, second])
    add(
        'indicatorAsMiddleOfThree',
        three,
        [predictor, three, create_sample(600, 0.0, 1.0, 2024)[0]],
    )
    return scenes


LARGE_OFFSET = 16777216.0
LARGE_OFFSET_GROUP_SIZE = 1000
LARGE_OFFSET_POSITIVE_COUNTS = [136, 345, 618, 841, 955]


def get_large_offset_scene():
    """frc2-regression-04: a predictor near 2^24 whose spread is five float32 steps."""
    predictor = []
    response = []
    for group, positive_count in enumerate(LARGE_OFFSET_POSITIVE_COUNTS):
        value = float(np.float32(LARGE_OFFSET + 2.0 * (group - 2)))
        for row in range(LARGE_OFFSET_GROUP_SIZE):
            predictor.append(value)
            # `row * 37 mod 1000` is a permutation: exactly `positive_count` ones, spread out.
            response.append(1.0 if (row * 37) % LARGE_OFFSET_GROUP_SIZE < positive_count else 0.0)
    outcomes = np.array(response)
    # The shift is exact in float64 and changes only the intercept.
    shifted = np.array(predictor) - LARGE_OFFSET
    design = np.column_stack([np.ones_like(shifted), shifted])
    scene = {
        'rowCount': len(response),
        'predictorSum': float(np.sum(np.array(predictor))),
        'predictorOffset': LARGE_OFFSET,
        'separationObjective': get_separation_objective(design, outcomes),
    }
    assert scene['separationObjective'] == 0.0
    scene.update(fit_tail_exact(design, outcomes))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = sm.Probit(outcomes, design).fit(method='newton', maxiter=200, tol=1e-12, disp=0)
    assert bool(result.mle_retvals['converged'])
    assert float(np.max(np.abs(result.params - np.array(scene['coefficients'])))) < 1e-9
    fitted = result.predict()
    row_count = len(response)
    ring = libpysal.weights.W(
        {row: sorted([(row + row_count - 1) % row_count, (row + 1) % row_count]) for row in range(row_count)},
        {row: [0.5, 0.5] for row in range(row_count)},
        silence_warnings=True,
    )
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        spatial = spreg.Probit(outcomes.reshape(-1, 1), shifted.reshape(-1, 1), w=ring, spat_diag=True, maxiter=200)
    assert float(np.max(np.abs(spatial.betas.flatten() - result.params))) < 1e-6
    library_fitted = np.asarray(spatial.predy).flatten()
    assert float(np.max(np.abs(library_fitted - fitted))) < 1e-7
    marginal_effect = float(result.get_margeff(at='overall', method='dydx').margeff[0])
    assert abs(float(np.asarray(spatial.slopes).flatten()[0]) / marginal_effect - 1) < 1e-6
    scene['statsmodels'] = {
        'coefficients': result.params.tolist(),
        'logLikelihood': float(result.llf),
        'converged': True,
        'expectedInformationStandardErrors': get_expected_information_standard_errors(design, outcomes),
        # One fitted probability per group, and the average marginal effect of the predictor.
        'groupFitted': [float(fitted[group * LARGE_OFFSET_GROUP_SIZE]) for group in range(5)],
        'averageMarginalEffect': marginal_effect,
    }
    scene['spreg'] = {
        # Pinkse (statistic, p), Kelejian-Prucha (z, p), Pinkse-Slade (statistic, p), on the ring
        # of weights 0.5 over the row order.
        'diagnostics': [
            float(value)
            for test in (spatial.Pinkse_error, spatial.KP_error, spatial.PS_error)
            for value in np.asarray(test).flatten()
        ],
    }
    return scene


def get_third_round_scenes():
    """Third round of review fixes (2026-11-11): frc2-regression-03, -04 and -06."""
    # frc2-regression-03: not separated (the two rows at -1e-5 and 1e-5 exclude every direction),
    # but a Fisher direction moves them towards the wrong side by 1.5e-5 of their own terms only.
    predictor = [-1.0] * 1000 + [1.0] * 2000 + [float(np.float32(-1e-5)), float(np.float32(1e-5))]
    response = [0.0] * 1000 + [1.0] * 2000 + [1.0, 0.0]
    # The expected information of the 3,000 outer rows is 4e-8 a row, so the library's IRLS
    # standard error of the slope (100.69) is 3.5e-5 from the one at the tail-exact maximum.
    scenes = {'nearlySeparated': get_scene(predictor, response, True, information_tolerance=1e-4)}
    scenes['largeOffsetPredictor'] = get_large_offset_scene()
    # frc2-regression-06: quasi-separated by the direction (0, 1); the rows with both outcomes
    # are at the middle value of a three-valued predictor.
    predictor = [0.0] * 100000 + [-1.0, 1.0]
    response = [0.0] * 40000 + [1.0] * 60000 + [0.0, 1.0]
    scenes['threeValuedMiddle'] = get_scene(predictor, response, False)
    return scenes


def main():
    predictor, response = create_sample(2000, 0.2, 2.0, 777)
    separated_predictor, separated_response = create_sample(400, 0.2, 3.5, 4242, noisy=False)
    scenes = {
        'control': get_scene(predictor, response, True),
        'correctOutlier': get_scene(predictor + [9.0], response + [1.0], True),
        'misclassifiedOutlier': get_scene(
            predictor + [4.0], response + [0.0], True, information_tolerance=2e-5
        ),
        'grossMisclassifiedOutlier': get_scene(
            predictor + [9.0], response + [0.0], True, statsmodels_is_maximum=False
        ),
        'justBeyondBound': get_scene(*create_sample(400, 0.2, 3.5, 12345), True),
        'completeSeparation': get_scene(separated_predictor, separated_response, False),
        'quasiCompleteSeparation': get_scene(
            separated_predictor + [-0.2 / 3.5] * 6, separated_response + [1.0, 0.0] * 3, False
        ),
        'steepSlope': get_scene(*create_sample(2000, 0.2, 8.0, 5), True),
        **get_indicator_scenes(),
        **get_large_sample_scenes(),
        **get_third_round_scenes(),
    }
    fixture = {
        'generator': 'modules/analysis/test/oracles/statistics-probit-tails.py',
        'versions': {
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'statsmodels': statsmodels.__version__,
            'libpysal': libpysal.__version__,
            'spreg': spreg.__version__,
        },
        'scenes': scenes,
    }
    with open(FIXTURE_PATH, 'w') as file:
        json.dump(fixture, file, indent=1)
        file.write('\n')


if __name__ == '__main__':
    main()
