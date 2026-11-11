# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Float64 references for the panel spatial lag estimator on fits whose residual sum of squares
is a small fraction of the response's (spreg, libpysal, NumPy), for the second round of review
fixes of the closing pass of the math correctness review (2026-11-11, frc-regression-09).

Writes fixtures/statistics-regression-panel-lag-conditioning.json next to this script, consumed
by test/gpu-spatial-analysis/spatial-regression/math-review-panel-lag-conditioning.spec.ts
(`GPUSpatialPanelLagMaximumLikelihood`). Run from anywhere with
    <venv>/bin/python -I modules/analysis/test/oracles/statistics-regression-panel-lag-conditioning.py
in the pinned environment of requirements.txt: CPython 3.14, NumPy 2.5.4, SciPy 1.16.3,
libpysal 4.15.0, spreg 1.9.1 (the versions are recorded in the fixture).

Scenes. 16 entities on a ring (two neighbours, weights 0.5), 16 periods, rows period-major
(`row = period * entities + entity`), one predictor. Every input comes from a 32-bit linear
congruential stream and IEEE double arithmetic in a fixed order, so the spec rebuilds
bit-identical float32 inputs from the scene parameters stored in the fixture (`responseSum`
checks it). The noise stream of `seed` gives the 16 unit effects (deviation 0.7) and then the
innovations in row order; the stream of `seed + 5000` gives the predictor, uniform on [-2, 2).
`y_t = (I - 0.4 W)^-1 (0.6 center + slope x_t + effects + deviation e_t)` by the power series
with 400 terms, then float32.
- `ordinary`: slope 0.5, deviation 1 (`u'u` is 0.7 of the residual sum of squares of the
  response on the predictor).
- `slope-2`, `slope-20`: deviation 0.004; `u'u` is 3e-5 and 3e-7 of it.
- `slope-200-center-*-stream-*`: deviation 0.004, response centres 0 and 10,000, four streams;
  `u'u` is 3e-9 of it.

References, float64 on the float32 inputs.
- `unit`: `spreg.Panel_FE_Lag(y, x, w, epsilon=1e-9)`: slope, `rho`, `sig2`, `logll` and the
  standard errors (`sqrt(diag(varb))`). A float64 maximum of the within-transformed concentrated
  likelihood (grid and golden section, ring eigenvalues) must agree with it (asserted: `rho`
  within 2e-8, `sig2` within 1e-7).
- `random`: `spreg.Panel_RE_Lag(y, x, w, epsilon=1e-9)`: intercept, slope, `rho`, `phi`,
  `sig2`, `logll` and the standard errors of the intercept, the slope and `rho`. A float64
  maximum of the random-effects likelihood `-NT/2 ln(sig2) + T ln|I - rho W| + N ln(phi)` found
  by nested golden sections (not spreg's alternation) is stored as `maximum`; spreg's value of
  that likelihood must not exceed it and must be within 1e-3 of it, and the two `sig2` must
  agree to 1e-5 (asserted). spreg's `logll` leaves out the `N ln(phi)` term; `maximum` stores
  the log-likelihood in spreg's convention.

`opposingEffectScene` (the third round of review fixes, frc2-regression-05) is one more scene
outside `scenes`: slope 2, deviation 0.004, stream 41, with `opposingEffect = 3000 (1 + rho)`
added to the unit effect of every even entity and subtracted from that of every odd one, so
that the response of an entity is near +3000 or -3000. `unit` is as above. `random` (fourth
round, frc-regression-13) is `spreg.Panel_RE_Lag(epsilon=1e-9)`, whose `phi` is 3e-7 here; it
is held to the first-order condition of the likelihood in `phi` at its own `rho` and
coefficients (1e-3), because `fit_random_maximum` brackets `phi` from 1e-6.
`phiBelowLimitScene` has entity effects of +16,000,000 and -16,000,000 (in runs of two entities,
`effectRunLength`) and a residual deviation of 0.5, so that spreg's `phi` (stored, asserted
below 1e-8) is under the lowest value the kernel follows.
"""

import contextlib
import io
import json
import math
import os
import sys
import warnings

import libpysal
import numpy as np
import spreg

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE_PATH = os.path.join(HERE, 'fixtures', 'statistics-regression-panel-lag-conditioning.json')
ENTITY_COUNT = 16
PERIOD_COUNT = 16
ROW_COUNT = ENTITY_COUNT * PERIOD_COUNT
SERIES_TERM_COUNT = 400
PARAMETER = 0.4
EFFECT_DEVIATION = 0.7
GOLDEN_RATIO = (math.sqrt(5) - 1) / 2
EIGENVALUES = np.cos(2 * np.pi * np.arange(ENTITY_COUNT) / ENTITY_COUNT)


def create_stream(seed):
    """Uniform stream in [0, 1): a 32-bit linear congruential state, top 24 bits."""
    state = [seed & 0xFFFFFFFF]

    def get_uniform():
        state[0] = (state[0] * 1664525 + 1013904223) & 0xFFFFFFFF
        return (state[0] >> 8) / 16777216.0

    return get_uniform


def get_ring_lag(values):
    """Lag of one period on the ring: `0.5 previous + 0.5 next`."""
    return [
        0.5 * values[(entity + ENTITY_COUNT - 1) % ENTITY_COUNT] + 0.5 * values[(entity + 1) % ENTITY_COUNT]
        for entity in range(ENTITY_COUNT)
    ]


def create_scene(scene):
    get_noise_uniform = create_stream(scene['seed'])
    get_predictor_uniform = create_stream(scene['seed'] + 5000)

    def get_deviate():
        total = get_noise_uniform() + get_noise_uniform() + get_noise_uniform() + get_noise_uniform()
        return (total - 2.0) * 1.7320508075688772

    effects = [EFFECT_DEVIATION * get_deviate() for _ in range(ENTITY_COUNT)]
    if 'opposingEffect' in scene:
        # Entity effects of alternating sign: an eigenvector of the ring (eigenvalue -1), so the
        # response of entity `i` carries `(-1)^i opposingEffect / (1 + rho)`.
        # With `effectRunLength` 2 the signs run + + - - around the ring: that vector is in the
        # null space of the ring weights, so the response carries the effect itself and no
        # `rho` removes it.
        run_length = scene.get('effectRunLength', 1)
        effects = [
            effects[entity]
            + (scene['opposingEffect'] if (entity // run_length) % 2 == 0 else -scene['opposingEffect'])
            for entity in range(ENTITY_COUNT)
        ]
    predictor = [float(np.float32(4.0 * get_predictor_uniform() - 2.0)) for _ in range(ROW_COUNT)]
    response = []
    for period in range(PERIOD_COUNT):
        signal = [
            0.6 * scene['center']
            + scene['slope'] * predictor[period * ENTITY_COUNT + entity]
            + effects[entity]
            + scene['deviation'] * get_deviate()
            for entity in range(ENTITY_COUNT)
        ]
        total = list(signal)
        term = list(signal)
        for _ in range(1, SERIES_TERM_COUNT):
            term = [PARAMETER * value for value in get_ring_lag(term)]
            total = [total[entity] + term[entity] for entity in range(ENTITY_COUNT)]
        response.extend(float(np.float32(value)) for value in total)
    return np.array(predictor), np.array(response)


def minimize_by_golden_section(objective, low, high, steps=200):
    for _ in range(steps):
        left = high - GOLDEN_RATIO * (high - low)
        right = low + GOLDEN_RATIO * (high - low)
        if objective(left) <= objective(right):
            high = right
        else:
            low = left
    return (low + high) / 2


def get_panel_lag(response):
    lagged = np.zeros(ROW_COUNT)
    for period in range(PERIOD_COUNT):
        start = period * ENTITY_COUNT
        lagged[start : start + ENTITY_COUNT] = get_ring_lag(list(response[start : start + ENTITY_COUNT]))
    return lagged


def get_entity_means(values):
    return values.reshape(PERIOD_COUNT, ENTITY_COUNT).mean(axis=0)


def fit_unit_maximum(predictor, response):
    """Float64 maximum of the within-transformed concentrated likelihood."""
    lagged = get_panel_lag(response)

    def demean(values):
        return values - np.tile(get_entity_means(values), PERIOD_COUNT)

    design, target, lag = demean(predictor), demean(response), demean(lagged)
    response_residual = target - design * (design @ target) / (design @ design)
    lagged_residual = lag - design * (design @ lag) / (design @ design)

    def get_negative_likelihood(parameter):
        residual = response_residual - parameter * lagged_residual
        return ROW_COUNT / 2 * math.log(residual @ residual / ROW_COUNT) - PERIOD_COUNT * float(
            np.sum(np.log(1 - parameter * EIGENVALUES))
        )

    grid = np.linspace(-0.95, 0.95, 1901)
    best = int(np.argmin([get_negative_likelihood(parameter) for parameter in grid]))
    parameter = minimize_by_golden_section(get_negative_likelihood, grid[best - 1], grid[best + 1])
    residual = response_residual - parameter * lagged_residual
    return {
        'parameter': float(parameter),
        'sigmaSquared': float(residual @ residual) / ROW_COUNT,
        'residualRatio': float((residual @ residual) / (response_residual @ response_residual)),
    }


def fit_random_maximum(predictor, response):
    """Float64 maximum of the random-effects likelihood by nested golden sections."""
    lagged = get_panel_lag(response)
    predictor_means = np.tile(get_entity_means(predictor), PERIOD_COUNT)
    response_means = np.tile(get_entity_means(response), PERIOD_COUNT)
    lagged_means = np.tile(get_entity_means(lagged), PERIOD_COUNT)

    def fit(parameter, phi):
        design = np.column_stack([np.full(ROW_COUNT, phi), predictor - (1 - phi) * predictor_means])
        target = (response - (1 - phi) * response_means) - parameter * (lagged - (1 - phi) * lagged_means)
        estimate = np.linalg.lstsq(design, target, rcond=None)[0]
        residual = target - design @ estimate
        variance = float(residual @ residual) / ROW_COUNT
        likelihood = (
            -ROW_COUNT / 2 * math.log(variance)
            + PERIOD_COUNT * float(np.sum(np.log(1 - parameter * EIGENVALUES)))
            + ENTITY_COUNT * math.log(phi)
        )
        return likelihood, variance

    def get_best_phi(parameter):
        return minimize_by_golden_section(lambda phi: -fit(parameter, phi)[0], 1e-6, 1.0, 100)

    grid = np.linspace(-0.9, 0.9, 181)
    best = int(np.argmax([fit(parameter, get_best_phi(parameter))[0] for parameter in grid]))
    parameter = minimize_by_golden_section(
        lambda value: -fit(value, get_best_phi(value))[0], grid[best - 1], grid[best + 1], 100
    )
    phi = get_best_phi(parameter)
    likelihood, variance = fit(parameter, phi)
    return {
        'parameter': float(parameter),
        'phi': float(phi),
        'sigmaSquared': variance,
        # spreg's logll: the likelihood above without its N ln(phi) term, less NT/2 (ln 2 pi + 1).
        'logLikelihood': float(
            likelihood - ENTITY_COUNT * math.log(phi) - ROW_COUNT / 2 * (math.log(2 * math.pi) + 1)
        ),
    }


def get_scene_references(scene, weights):
    predictor, response = create_scene(scene)
    with contextlib.redirect_stdout(io.StringIO()):
        unit_model = spreg.Panel_FE_Lag(response.reshape(-1, 1), predictor.reshape(-1, 1), weights, epsilon=1e-9)
        random_model = spreg.Panel_RE_Lag(response.reshape(-1, 1), predictor.reshape(-1, 1), weights, epsilon=1e-9)
    unit_coefficients = np.asarray(unit_model.betas).flatten()
    unit_errors = np.sqrt(np.diag(np.asarray(unit_model.varb)))
    unit = {
        'slope': float(unit_coefficients[0]),
        'parameter': float(unit_coefficients[1]),
        'standardErrors': [float(unit_errors[0]), float(unit_errors[1])],
        'sigmaSquared': float(np.asarray(unit_model.sig2).reshape(-1)[0]),
        'logLikelihood': float(np.asarray(unit_model.logll).reshape(-1)[0]),
    }
    unit_maximum = fit_unit_maximum(predictor, response)
    assert abs(unit_maximum['parameter'] - unit['parameter']) < 2e-8, (scene['name'], unit_maximum, unit)
    assert abs(unit_maximum['sigmaSquared'] / unit['sigmaSquared'] - 1) < 1e-7, (scene['name'], unit_maximum, unit)
    unit['residualRatio'] = unit_maximum['residualRatio']

    random_coefficients = np.asarray(random_model.betas).flatten()
    random_errors = np.sqrt(np.diag(np.asarray(random_model.varb)))
    random = {
        'intercept': float(random_coefficients[0]),
        'slope': float(random_coefficients[1]),
        'parameter': float(random_coefficients[2]),
        'phi': float(random_coefficients[3]),
        'standardErrors': [float(value) for value in random_errors[:3]],
        'sigmaSquared': float(np.asarray(random_model.sig2).reshape(-1)[0]),
        'logLikelihood': float(np.asarray(random_model.logll).reshape(-1)[0]),
    }
    random['maximum'] = fit_random_maximum(predictor, response)
    # The maximized likelihood has the N ln(phi) term that spreg's logll leaves out.
    gap = (
        random['maximum']['logLikelihood']
        + ENTITY_COUNT * math.log(random['maximum']['phi'])
        - random['logLikelihood']
        - ENTITY_COUNT * math.log(random['phi'])
    )
    assert -1e-6 < gap < 1e-3, (scene['name'], gap, random)
    assert abs(random['maximum']['sigmaSquared'] / random['sigmaSquared'] - 1) < 1e-5, (scene['name'], random)
    return {'responseSum': float(np.sum(response)), 'unit': unit, 'random': random}


def get_opposing_effect_scene(weights):
    """Third round of review fixes (frc2-regression-05): large entity effects of opposing sign
    and small variation within an entity. The response of an entity is within 5 of +3000 or of
    -3000 (one float32 binade, a step of 2.4e-4), the pooled mean is near zero, and the
    residual deviation is 0.004: the quasi-demeaned difference `(y - pooled mean) - entity
    mean` has to be formed before the fitted part is subtracted. `random` is the fourth-round
    reference (frc-regression-13): the random-effects `phi` of such data is 2.5e-7."""
    scene = {
        'name': 'opposing-effects-3000',
        'slope': 2.0,
        'center': 0.0,
        'deviation': 0.004,
        'seed': 41,
        'opposingEffect': 3000.0 * (1 + PARAMETER),
    }
    predictor, response = create_scene(scene)
    with contextlib.redirect_stdout(io.StringIO()):
        model = spreg.Panel_FE_Lag(response.reshape(-1, 1), predictor.reshape(-1, 1), weights, epsilon=1e-9)
    coefficients = np.asarray(model.betas).flatten()
    errors = np.sqrt(np.diag(np.asarray(model.varb)))
    unit = {
        'slope': float(coefficients[0]),
        'parameter': float(coefficients[1]),
        'standardErrors': [float(errors[0]), float(errors[1])],
        'sigmaSquared': float(np.asarray(model.sig2).reshape(-1)[0]),
        'logLikelihood': float(np.asarray(model.logll).reshape(-1)[0]),
    }
    maximum = fit_unit_maximum(predictor, response)
    assert abs(maximum['parameter'] - unit['parameter']) < 2e-8, (maximum, unit)
    assert abs(maximum['sigmaSquared'] / unit['sigmaSquared'] - 1) < 1e-7, (maximum, unit)
    unit['residualRatio'] = maximum['residualRatio']
    entity_means = get_entity_means(response)
    return {
        **scene,
        'responseSum': float(np.sum(response)),
        'largestEntityMean': float(np.max(np.abs(entity_means))),
        'pooledMean': float(np.mean(response)),
        'unit': unit,
        'random': get_small_phi_reference(predictor, response, weights),
    }


def get_small_phi_reference(predictor, response, weights, stationarity_tolerance=1e-3):
    """`spreg.Panel_RE_Lag(epsilon=1e-9)` on a panel whose `phi` is far below 1e-5 (fourth
    round, frc-regression-13). `fit_random_maximum` brackets `phi` from 1e-6, so the check here
    is the first-order condition of the likelihood in `phi` instead: at spreg's `rho` and
    coefficients, `phi^2 = within / ((T - 1) between)` with `within` and `between` the sums of
    squares of the within-entity deviations and of the entity means of `y - rho W y - a - x b`."""
    with contextlib.redirect_stdout(io.StringIO()):
        model = spreg.Panel_RE_Lag(response.reshape(-1, 1), predictor.reshape(-1, 1), weights, epsilon=1e-9)
    coefficients = np.asarray(model.betas).flatten()
    errors = np.sqrt(np.diag(np.asarray(model.varb)))
    reference = {
        'intercept': float(coefficients[0]),
        'slope': float(coefficients[1]),
        'parameter': float(coefficients[2]),
        'phi': float(coefficients[3]),
        'standardErrors': [float(value) for value in errors[:3]],
        'sigmaSquared': float(np.asarray(model.sig2).reshape(-1)[0]),
        'logLikelihood': float(np.asarray(model.logll).reshape(-1)[0]),
    }
    structural = response - reference['parameter'] * get_panel_lag(response) - reference['intercept'] - reference['slope'] * predictor
    means = np.tile(get_entity_means(structural), PERIOD_COUNT)
    within = float(np.sum((structural - means) ** 2))
    between = float(np.sum(means**2))
    stationary_phi = math.sqrt(within / ((PERIOD_COUNT - 1) * between))
    assert abs(stationary_phi / reference['phi'] - 1) < stationarity_tolerance, (stationary_phi, reference)
    return reference


def get_phi_below_limit_scene(weights):
    """Fourth round (frc-regression-13): a panel whose random-effects `phi` is below the lowest
    value the kernel follows (1e-8). Entity effects of +16,000,000 and -16,000,000 in runs of
    two entities (a vector the ring weights map to zero, so that no `rho` removes it; with
    alternating signs the likelihood of such effects peaks at `rho = -1`), where a float32 step
    is 1, with a residual deviation of 0.5: the response is close to integers and its float32
    rounding is part of the residual. Only spreg's `phi` is used."""
    scene = {
        'name': 'opposing-effects-16000000',
        'slope': 2.0,
        'center': 0.0,
        'deviation': 0.5,
        'seed': 43,
        'opposingEffect': 16000000.0,
        'effectRunLength': 2,
    }
    predictor, response = create_scene(scene)
    # spreg stops its alternation 1.6% from the stationary `phi` here (9.07e-9 against 8.92e-9).
    reference = get_small_phi_reference(predictor, response, weights, stationarity_tolerance=0.05)
    assert reference['phi'] < 9.5e-9, reference
    return {**scene, 'responseSum': float(np.sum(response)), 'random': {'phi': reference['phi']}}


def get_scenes():
    scenes = [
        {'name': 'ordinary', 'slope': 0.5, 'center': 0.0, 'deviation': 1.0, 'seed': 11},
        {'name': 'slope-2', 'slope': 2.0, 'center': 0.0, 'deviation': 0.004, 'seed': 32},
        {'name': 'slope-20', 'slope': 20.0, 'center': 0.0, 'deviation': 0.004, 'seed': 31},
    ]
    for center in (0.0, 10000.0):
        for seed in (21, 22, 23, 24):
            scenes.append(
                {
                    'name': f'slope-200-center-{center:g}-stream-{seed}',
                    'slope': 200.0,
                    'center': center,
                    'deviation': 0.004,
                    'seed': seed,
                }
            )
    return scenes


def main():
    warnings.simplefilter('ignore')
    weights = libpysal.weights.W(
        {
            entity: sorted([(entity + ENTITY_COUNT - 1) % ENTITY_COUNT, (entity + 1) % ENTITY_COUNT])
            for entity in range(ENTITY_COUNT)
        },
        {entity: [0.5, 0.5] for entity in range(ENTITY_COUNT)},
        silence_warnings=True,
    )
    fixture = {
        'generator': 'modules/analysis/test/oracles/statistics-regression-panel-lag-conditioning.py',
        'versions': {
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'scipy': __import__('scipy').__version__,
            'libpysal': libpysal.__version__,
            'spreg': spreg.__version__,
        },
        'entityCount': ENTITY_COUNT,
        'periodCount': PERIOD_COUNT,
        'seriesTermCount': SERIES_TERM_COUNT,
        'parameter': PARAMETER,
        'effectDeviation': EFFECT_DEVIATION,
        'scenes': [{**scene, **get_scene_references(scene, weights)} for scene in get_scenes()],
        'opposingEffectScene': get_opposing_effect_scene(weights),
        'phiBelowLimitScene': get_phi_below_limit_scene(weights),
    }
    with open(FIXTURE_PATH, 'w') as file:
        json.dump(fixture, file, indent=1)
        file.write('\n')


if __name__ == '__main__':
    main()
