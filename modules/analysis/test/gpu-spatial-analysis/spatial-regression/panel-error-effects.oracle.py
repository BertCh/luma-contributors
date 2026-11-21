# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent dense Gaussian ML oracle. Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/panel-error-effects.oracle.py

No implementation imports. Fixed effects are explicit nuisance intercepts in the ORIGINAL
panel, eliminated by dense least squares AFTER spatial filtering. This deliberately does
not use the contributor's within transformation or its trace-series objective.
"""
import json
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.special import erfc

entities, periods, predictors = 12, 6, 2
rows = entities * periods
generator = np.random.default_rng(18372)
weights = np.zeros((entities, entities))
for entity in range(entities - 1):
    neighbors = sorted({(entity + 1) % (entities - 1), (entity + 4) % (entities - 1)})
    weights[entity, neighbors] = [0.25 if entity % 2 else 0.625, 0.75 if entity % 2 else 0.375]
spatial = np.kron(np.eye(periods), weights)
design = np.round(generator.normal(size=(periods, entities, predictors)) * 32) / 32
design[:, :, 0] += np.array([-1, 1, -2, 2, -0.5, 0.5])[:, None]
unit = np.round(generator.normal(size=entities) * 64) / 16
shocks = np.array([-2, 2, -3, 3, -1, 1])
disturbance = np.linalg.solve(np.eye(rows) - 0.35 * spatial, generator.normal(size=rows))
response = np.round((design.reshape(rows, predictors) @ [1.25, -0.75] +
                     np.tile(unit, periods) + np.repeat(shocks, entities) + disturbance) * 32) / 32
design = design.reshape(rows, predictors).astype(np.float32).astype(float)
response = response.astype(np.float32).astype(float)
maximum_power = 128
low, high = 0., 0.99
for _ in range(60):
    middle = (low + high) / 2
    if middle ** (maximum_power + 1) / ((maximum_power + 1) * (1 - middle)) <= 1e-5:
        low = middle
    else:
        high = middle
bound = low

def fit(effects, design=design, response=response, weights=weights, periods=periods):
    entities = len(weights)
    rows, predictors = design.shape
    spatial = np.kron(np.eye(periods), weights)
    entity_intercepts = np.tile(np.eye(entities), (periods, 1))
    period_intercepts = np.repeat(np.eye(periods), entities, axis=0)
    nuisance = {'unit': entity_intercepts, 'time': period_intercepts,
                'twoWay': np.column_stack((entity_intercepts, period_intercepts[:, :-1]))}[effects]
    full_design = np.column_stack((design, nuisance))
    def evaluate(parameter):
        filter_matrix = np.eye(rows) - parameter * spatial
        filtered_design = filter_matrix @ full_design
        beta = np.linalg.lstsq(filtered_design, filter_matrix @ response, rcond=None)[0]
        residual = response - full_design @ beta
        error = filter_matrix @ residual
        variance = error @ error / rows
        determinant = periods * np.linalg.slogdet(np.eye(entities) - parameter * weights)[1]
        return rows / 2 * np.log(variance) - determinant, beta, residual, variance, determinant, filtered_design
    # Bounded scalar minimization is local. Enumerate bracketing minima and both eligible
    # endpoints so a better boundary mode cannot be missed by one golden-section search.
    grid = np.linspace(-bound, bound, 1025)
    objectives = np.array([evaluate(value)[0] for value in grid])
    candidates = []
    if objectives[0] <= objectives[1]:
        candidates.append((-bound, True))
    for index in range(1, 1024):
        if objectives[index] < objectives[index-1] and objectives[index] <= objectives[index+1]:
            optimum = minimize_scalar(lambda value: evaluate(value)[0], bounds=(grid[index-1], grid[index+1]),
                                      method='bounded', options={'xatol': 1e-13})
            candidates.append((optimum.x, False))
    if objectives[-1] <= objectives[-2]:
        candidates.append((bound, True))
    parameter, at_boundary = min(candidates, key=lambda candidate: evaluate(candidate[0])[0])
    objective, beta, residual, variance, determinant, filtered_design = evaluate(parameter)
    covariance = variance * np.linalg.inv(filtered_design.T @ filtered_design)
    inverse_product = weights @ np.linalg.inv(np.eye(entities) - parameter * weights)
    trace_one = periods * np.trace(inverse_product)
    information = periods * np.trace(inverse_product @ inverse_product + inverse_product.T @ inverse_product) - 2 * trace_one ** 2 / rows
    estimates = np.r_[beta[:predictors], parameter]
    errors = np.r_[np.sqrt(np.diag(covariance)[:predictors]), 1 / np.sqrt(information)]
    statistics = estimates / errors
    table = np.r_[np.full(4, np.nan), np.column_stack((estimates, errors, statistics, erfc(np.abs(statistics) / np.sqrt(2)))).ravel()]
    panel = np.column_stack((response, design)).reshape(periods, entities, predictors + 1)
    if effects in ('unit', 'twoWay'):
        panel = panel - panel.mean(axis=0, keepdims=True)
    if effects in ('time', 'twoWay'):
        panel = panel - panel.mean(axis=1, keepdims=True)
    transformed = panel.reshape(rows, predictors + 1)
    profile_design = transformed[:, 1:]
    if effects != 'unit':
        profile_design = np.column_stack((profile_design, period_intercepts[:, :-1]))
    means = profile_design.mean(axis=0)
    centered_design = np.column_stack((np.ones(rows), profile_design - means))
    filter_matrix = np.eye(rows) - parameter * spatial
    filtered_centered = filter_matrix @ centered_design
    filtered_raw = filter_matrix @ np.column_stack((np.ones(rows), profile_design))
    normalized = filtered_centered / np.linalg.norm(filtered_centered, axis=0)
    correlation = normalized.T @ normalized
    raw_normalized = filtered_raw / np.linalg.norm(filtered_raw, axis=0)
    conditioning = [np.linalg.cond(raw_normalized), np.linalg.cond(normalized) ** 2,
                    np.diag(np.linalg.inv(correlation))[1:].max()]
    fitted = transformed[:, 0] - residual
    pseudo = np.corrcoef(transformed[:, 0], fitted)[0, 1] ** 2
    summary = [rows, parameter, variance, -objective - rows/2*(np.log(2*np.pi)+1),
               np.nan, np.nan, pseudo, np.nan, residual @ residual, *conditioning,
               int(conditioning[1] > 1e4), bound, 0, int(not at_boundary), int(at_boundary), 1,
               rows * abs(parameter)**(maximum_power+1)/((maximum_power+1)*(1-abs(parameter))),
               1, len(candidates), determinant]
    return {'table': table.tolist(), 'summary': summary, 'residuals': residual.tolist()}

def fit_pooled_diagnostics(design, response, spatial):
    rows = len(response)
    pooled = np.column_stack((np.ones(rows), design))
    pooled_beta = np.linalg.lstsq(pooled, response, rcond=None)[0]
    pooled_residual = response - pooled @ pooled_beta
    pooled_fit = pooled @ pooled_beta
    variance = pooled_residual @ pooled_residual / (rows - pooled.shape[1])
    lagged_fit = spatial @ pooled_fit
    projected = lagged_fit - pooled @ np.linalg.lstsq(pooled, lagged_fit, rcond=None)[0]
    trace = np.trace(spatial.T @ spatial + spatial @ spatial)
    information = projected @ projected / variance + trace
    residual_lag = pooled_residual @ spatial @ pooled_residual / variance
    response_lag = pooled_residual @ spatial @ response / variance
    tests = [residual_lag**2 / trace, (response_lag-residual_lag)**2 / (information-trace),
             (residual_lag-trace/information*response_lag)**2/(trace*(1-trace/information))]
    return np.column_stack((tests, erfc(np.sqrt(np.array(tests)/2)))).ravel().tolist()

pooled = np.column_stack((np.ones(rows), design))
diagnostics = fit_pooled_diagnostics(design, response, spatial)

# Independent unit effects: between covariance is A^-1 A^-T + (phi^-2-1) I.
# This extra determinant cannot be supplied by a scalar quasi-demeaning of W alone.
def random_objective(parameters):
    parameter, phi = parameters
    if abs(parameter) >= bound or not 1e-8 <= phi <= 1:
        return 1e100
    inverse = np.linalg.inv(np.eye(entities) - parameter * weights)
    covariance = np.kron(np.eye(periods), inverse @ inverse.T) + (phi**-2 - 1)/periods * np.kron(np.ones((periods, periods)), np.eye(entities))
    factor = np.linalg.cholesky(covariance)
    whitened_design = np.linalg.solve(factor, pooled)
    whitened_response = np.linalg.solve(factor, response)
    beta = np.linalg.lstsq(whitened_design, whitened_response, rcond=None)[0]
    residual = whitened_response - whitened_design @ beta
    return rows/2 * np.log(residual @ residual / rows) + np.log(np.diag(factor)).sum()

nonzero = np.nonzero(weights)
fixture = {'entityCount': entities, 'periodCount': periods, 'predictorCount': predictors,
           'predictors': design.ravel().tolist(), 'response': response.tolist(),
           'weights': {'offsets': np.r_[0, np.cumsum(np.count_nonzero(weights, axis=1))].tolist(),
                       'neighbors': nonzero[1].tolist(), 'weights': weights[nonzero].tolist()},
           'diagnostics': diagnostics, 'models': {effect: fit(effect) for effect in ('unit', 'time', 'twoWay')}}
random_fit = minimize(random_objective, [0.3, 0.3], method='Nelder-Mead',
                      options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000})
fixture['randomReference'] = {'parameter': random_fit.x.tolist(), 'objective': random_fit.fun}

# Random123 Philox4x32-10 equations, using Python's exact full-width integer products rather
# than the implementation's 16-bit multiplication helper. Check the published all-zero KAT.
def philox(counter, key):
    for _ in range(10):
        first = 0xD2511F53 * counter[0]
        second = 0xCD9E8D57 * counter[2]
        counter = [(second >> 32) ^ counter[1] ^ key[0], second & 0xFFFFFFFF,
                   (first >> 32) ^ counter[3] ^ key[1], first & 0xFFFFFFFF]
        key = [(key[0] + 0x9E3779B9) & 0xFFFFFFFF, (key[1] + 0xBB67AE85) & 0xFFFFFFFF]
    return counter

assert philox([0, 0, 0, 0], [0, 0]) == [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]
probes = np.array([[1 - 2 * (philox([probe, row//4, 0, 0x50524F42], [11, 0x54524143])[row%4] & 1)
                    for row in range(rows)] for probe in range(4)])
trace_errors = []
for power in range(1, 9):
    dense_power = np.linalg.matrix_power(spatial, power)
    estimates = np.einsum('pi,ij,pj->p', probes, dense_power, probes)
    trace_errors.append(float(estimates.std(ddof=1) / 2))
fixture['stochasticTraceErrors'] = trace_errors

# Exceeds the GPU's 15-slope-plus-period-contrast limit. All four entities have the same
# period membership; one is an island, and the non-island graph is asymmetric.
extended_entities, extended_periods = 4, 16
extended_weights = np.array([[0, .5, .5, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0.]])
extended_spatial = np.kron(np.eye(extended_periods), extended_weights)
extended_generator = np.random.default_rng(19834)
extended_design = np.round(extended_generator.normal(size=(64, 1)) * 8) / 8
extended_design[:, 0] += np.repeat(np.array([-1, 1, -.5, .5] * 4), 4)
extended_error = np.linalg.solve(np.eye(64) - .5 * extended_spatial, np.random.default_rng(9745).normal(size=64))
extended_response = np.round((extended_design[:, 0]*1.75 +
                              np.repeat(np.array([-2, 2, -1, 1]*4), 4) + extended_error)*8)/8
extended_nonzero = np.nonzero(extended_weights)
fixture['extendedPeriods'] = {
    'entityCount': extended_entities, 'periodCount': extended_periods, 'predictorCount': 1,
    'predictors': extended_design.ravel().tolist(), 'response': extended_response.tolist(),
    'weights': {'offsets': [0, 2, 3, 4, 4], 'neighbors': extended_nonzero[1].tolist(),
                'weights': extended_weights[extended_nonzero].tolist()},
    'diagnostics': fit_pooled_diagnostics(extended_design, extended_response, extended_spatial),
    'models': {effects: fit(effects, extended_design, extended_response, extended_weights, extended_periods)
               for effects in ('time', 'twoWay')}
}

# Same W power traces and infinity norm, distinct independent-unit between determinants.
nilpotent_first = np.array([[0, 1., 0], [0, 0, 0], [0, 0, 0]])
nilpotent_second = np.array([[0, 1., 0], [0, 0, 1.], [0, 0, 0]])
between_log_determinants = []
for nilpotent in [nilpotent_first, nilpotent_second]:
    filter_matrix = np.eye(3) - .4 * nilpotent
    between_log_determinants.append(float(np.linalg.slogdet(np.eye(3) + 3 * filter_matrix @ filter_matrix.T)[1]))
fixture['randomCounterexample'] = {'lambda': .4, 'phi': .5, 'extraLogDeterminants': between_log_determinants}
print(json.dumps(fixture, indent=2, allow_nan=True))
