# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent panel oracle: NumPy SVD least squares, SciPy nested bounded optimization.

Run /usr/bin/python3 on this file. Inputs are quantized to float32 before fitting. Random
effects maximize the full concentrated likelihood in (rho, log(phi)), including N log(phi),
rather than replaying the contributor's alternating analytic phi update. Unit effects optimize
the within-transformed model. Expected information is assembled with matrix products. Output
likelihood follows spreg's reporting convention. No existing oracle files are modified.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2


def fit_panel(predictors, response, weights, periods, unit):
    entities = len(weights)
    rows, predictor_count = predictors.shape
    columns = predictor_count + 1
    response = response.reshape(periods, entities)
    predictors = predictors.reshape(periods, entities, predictor_count)
    lagged = response @ weights.T

    def fit(parameter, phi):
        design = np.concatenate((np.ones((periods, entities, 1)), predictors), axis=2)
        if unit:
            design = design[:, :, 1:]
        design = (design - (1 - phi) * design.mean(axis=0)).reshape(rows, -1)
        target = (response - (1 - phi) * response.mean(axis=0)).ravel()
        lag = (lagged - (1 - phi) * lagged.mean(axis=0)).ravel()
        estimate = np.linalg.lstsq(design, target - parameter * lag, rcond=None)[0]
        residual = target - parameter * lag - design @ estimate
        variance = residual @ residual / rows
        likelihood = -rows / 2 * np.log(variance) + periods * np.linalg.slogdet(np.eye(entities) - parameter * weights)[1]
        if not unit:
            likelihood += entities * np.log(phi)
        return likelihood, design, estimate, variance, target - residual, residual

    def get_best_phi(parameter):
        if unit:
            return 0.0
        optimum = minimize_scalar(lambda value: -fit(parameter, np.exp(value))[0], bounds=(np.log(1e-8), 0), method='bounded', options={'xatol': 1e-13})
        return max([np.exp(optimum.x), 1.0, 1e-8], key=lambda value: fit(parameter, value)[0])

    # A coarse bracket followed by a different library optimizer. The fixture maxima are
    # isolated away from the parameter bounds; this is an oracle, not a global-fit API.
    grid = np.linspace(-0.998, 0.998, 101)
    best = int(np.argmax([fit(value, get_best_phi(value))[0] for value in grid]))
    assert 0 < best < len(grid) - 1
    optimum = minimize_scalar(lambda value: -fit(value, get_best_phi(value))[0], bounds=(grid[best - 1], grid[best + 1]), method='bounded', options={'xatol': 1e-13})
    parameter = optimum.x
    phi = get_best_phi(parameter)
    likelihood, design, estimate, variance, fitted, residual = fit(parameter, phi)
    product = weights @ np.linalg.inv(np.eye(entities) - parameter * weights)
    prediction = (design @ estimate).reshape(periods, entities)
    lag_prediction = (prediction @ product.T).ravel()
    information_design = np.column_stack((design, lag_prediction))
    information = information_design.T @ information_design / variance
    information[-1, -1] += periods * (np.trace(product @ product) + np.sum(product ** 2))
    covariance = np.linalg.inv(information)
    if unit:
        expanded = np.full((columns + 1, columns + 1), np.nan)
        expanded[1:, 1:] = covariance
        covariance = expanded
        estimate = np.r_[np.nan, estimate]
    likelihood -= (0 if unit else entities * np.log(phi)) + rows / 2 * (1 + np.log(2 * np.pi))
    if unit:
        likelihood -= rows / 2 * np.log(rows)
    return {'coefficients': np.r_[estimate, parameter, phi].tolist(), 'covariance': covariance.ravel().tolist(), 'variance': variance, 'logLikelihood': likelihood, 'fitted': fitted.tolist(), 'residuals': residual.tolist()}


def create_fixture(seed, unit=False, effect_scale=0.7, predictor_count=2):
    generator = np.random.default_rng(seed)
    entities, periods = 8, 6
    weights = np.zeros((entities, entities))
    for entity in range(entities):
        weights[entity, (entity - 1) % entities] = 0.25
        weights[entity, (entity + 1) % entities] = 0.75
    predictors = generator.normal(size=(periods * entities, predictor_count)).astype(np.float32).astype(float)
    slopes = np.array([0.8, -0.4][:predictor_count])
    effects = effect_scale * generator.normal(size=entities)
    signal = (1.3 + predictors @ slopes).reshape(periods, entities) + effects + generator.normal(size=(periods, entities))
    response = np.linalg.solve(np.eye(entities) - 0.35 * weights, signal.T).T.ravel().astype(np.float32).astype(float)
    expected = fit_panel(predictors, response, weights, periods, unit)
    if not unit:
        comparison = fit_panel(predictors, response, weights, periods, True)
        difference = np.array(expected['covariance']).reshape(predictor_count + 2, -1)[1:, 1:] - np.array(comparison['covariance']).reshape(predictor_count + 2, -1)[1:, 1:]
        if np.linalg.eigvalsh(difference).min() > 0:
            contrast = np.array(comparison['coefficients'][1:-1]) - np.array(expected['coefficients'][1:-1])
            statistic = contrast @ np.linalg.solve(difference, contrast)
            expected['hausman'] = {'status': 0, 'tests': [statistic, predictor_count + 2, chi2.sf(statistic, predictor_count + 2)]}
        else:
            expected['hausman'] = {'status': 2, 'tests': [None, None, None]}
    offsets, neighbors, values = [0], [], []
    for row in weights:
        for neighbor in np.flatnonzero(row):
            neighbors.append(int(neighbor))
            values.append(float(row[neighbor]))
        offsets.append(len(neighbors))
    return {'name': f'{"unit" if unit else "random"}-{seed}', 'entityCount': entities, 'periodCount': periods, 'predictorCount': predictor_count, 'effects': 'unit' if unit else 'random', 'weights': {'offsets': offsets, 'neighbors': neighbors, 'weights': values}, 'predictors': predictors.ravel().tolist(), 'response': response.tolist(), 'expected': expected}


def replace_nonfinite(value):
    if isinstance(value, dict):
        return {key: replace_nonfinite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_nonfinite(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


fixtures = [create_fixture(19), create_fixture(19, unit=True), create_fixture(25, effect_scale=0), create_fixture(38, predictor_count=1)]
Path(__file__).with_suffix('.json').write_text(json.dumps(replace_nonfinite(fixtures), indent=2) + '\n')


def create_review_scene(scene, reference):
    """Rebuild the review's exact float32 inputs; the committed spreg values stay read-only."""
    def create_stream(seed):
        state = seed
        def get_uniform():
            nonlocal state
            state = (state * 1664525 + 1013904223) & 0xffffffff
            return (state >> 8) / 16777216
        return get_uniform
    noise = create_stream(scene['seed'])
    predictor_stream = create_stream(scene['seed'] + 5000)
    def get_deviate():
        return (noise() + noise() + noise() + noise() - 2) * 1.7320508075688772
    entities, periods = reference['entityCount'], reference['periodCount']
    effects = [reference['effectDeviation'] * get_deviate() for _ in range(entities)]
    if 'opposingEffect' in scene:
        effects = [value + (1 if (entity // scene.get('effectRunLength', 1)) % 2 == 0 else -1) * scene['opposingEffect'] for entity, value in enumerate(effects)]
    predictors = [float(np.float32(4 * predictor_stream() - 2)) for _ in range(entities * periods)]
    response = []
    for period in range(periods):
        total = [0.6 * scene['center'] + scene['slope'] * predictors[period * entities + entity] + effects[entity] + scene['deviation'] * get_deviate() for entity in range(entities)]
        term = total[:]
        for _ in range(1, reference['seriesTermCount']):
            term = [reference['parameter'] * (0.5 * term[(entity - 1) % entities] + 0.5 * term[(entity + 1) % entities]) for entity in range(entities)]
            total = [value + term[entity] for entity, value in enumerate(total)]
        response.extend(float(np.float32(value)) for value in total)
    assert sum(response) == scene['responseSum']
    offsets, neighbors = [0], []
    for entity in range(entities):
        neighbors.extend(sorted([(entity - 1) % entities, (entity + 1) % entities]))
        offsets.append(len(neighbors))
    return {'entityCount': entities, 'periodCount': periods, 'predictorCount': 1, 'predictors': predictors, 'response': response, 'weights': {'offsets': offsets, 'neighbors': neighbors, 'weights': [0.5] * len(neighbors)}}


reference = json.loads((Path(__file__).parents[2] / 'oracles/fixtures/statistics-regression-panel-lag-conditioning.json').read_text())
edges = {name: create_review_scene(scene, reference) for name, scene in [('belowLimit', reference['phiBelowLimitScene']), ('highFit', reference['scenes'][3])]}
edges['reviewCases'] = [
    {'name': name, 'entityCount': 4, 'periodCount': 2, 'predictorCount': 1,
     'weights': {'offsets': [0, 1, 2, 3, 4], 'neighbors': [2, 3, 0, 1], 'weights': [1, 1, 1, 1]},
     'predictors': [0, 0, -1, 1, 0, 0, 1, -1],
     'response': [scale * value for value in [-1, 1, 0, 0, 1, -1, 0, 0]]}
    for name, scale in [('varianceOverflow', 2.0**70), ('varianceUnderflow', 2.0**-90)]
] + [
    {'name': 'responseCopy', 'entityCount': 4, 'periodCount': 2, 'predictorCount': 2,
     'weights': {'offsets': [0, 1, 2, 3, 4], 'neighbors': [1, 2, 3, 0], 'weights': [1, 1, 1, 1]},
     'predictors': [2, 0, 0, 2, 0, -2, -2, 0, -2, 0, 0, -2, 0, 2, 2, 0],
     'response': [2, 0, 0, -2, -2, 0, 0, 2]},
    {'name': 'normalizationBoundary', 'entityCount': 4, 'periodCount': 2, 'predictorCount': 1,
     'weights': {'offsets': [0, 3, 4, 5, 6], 'neighbors': [1, 2, 3, 2, 3, 0],
                 'weights': [0.25, 0.2499999701976776, 0.5000100135803223, 1, 1, 1]},
     'predictors': [-1, 1, 2, -2, 1, -1, -2, 2], 'response': [1, 4, 2, -1, 3, -2, 1, 5]},
    {'name': 'originalPole', 'entityCount': 4, 'periodCount': 2, 'predictorCount': 1,
     'weights': {'offsets': [0, 1, 2, 3, 4], 'neighbors': [1, 2, 3, 0], 'weights': [1, 1, 1, 1]},
     'predictors': [-1.5, 0, 3, -1.5, 1.5, 0, -3, 1.5], 'response': [-1, 1, 2, -2, 1, -1, -2, 2]}
]
Path(__file__).with_name('spatial-panel-lag-maximum-likelihood-operation-edge-fixtures.json').write_text(json.dumps(edges, indent=2) + '\n')
