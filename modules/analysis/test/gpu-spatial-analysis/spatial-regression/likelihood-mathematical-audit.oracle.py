# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Independent dense ML oracle, including the full information matrix and sigma2 nuisance.

Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/likelihood-mathematical-audit.oracle.py
Prints JSON for likelihood-mathematical-audit-fixture.json. No production helpers are imported.
The likelihood uses dense slogdet, fresh least squares and scipy.optimize.minimize_scalar.
Both models use ML variance SSE/n; AIC follows spreg (lambda is uncounted for ML_Error).
"""

import json
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm


def fit_model(matrix, predictors, response, kind):
    row_count = len(response)
    design = np.column_stack((np.ones(row_count), predictors))
    width = design.shape[1]
    infinity_norm = np.max(np.abs(matrix).sum(axis=1))
    maximum_power = 64
    limit = brentq(lambda value: value ** 65 / (65 * (1 - value)) - 1e-5, 0, .99)
    bound = limit / infinity_norm

    def evaluate(parameter):
        filtering = np.eye(row_count) - parameter * matrix
        filtered_design = filtering @ design if kind == 'error' else design
        coefficients = np.linalg.lstsq(filtered_design, filtering @ response, rcond=None)[0]
        errors = filtering @ response - filtered_design @ coefficients
        variance = errors @ errors / row_count
        objective = -.5 * row_count * np.log(variance) + np.linalg.slogdet(filtering)[1]
        return objective, coefficients, variance, filtered_design

    # This fixture is unimodal; check that explicitly with a dense grid before bounded search.
    grid = np.linspace(-bound, bound, 2001)
    objectives = np.array([evaluate(parameter)[0] for parameter in grid])
    peaks = np.flatnonzero((objectives[1:-1] > objectives[:-2]) & (objectives[1:-1] > objectives[2:])) + 1
    assert len(peaks) == 1
    peak = peaks[0]
    parameter = minimize_scalar(lambda value: -evaluate(value)[0], method='bounded',
                                bounds=(grid[peak-1], grid[peak+1]), options={'xatol': 1e-14}).x
    objective, coefficients, variance, filtered_design = evaluate(parameter)
    filtering = np.eye(row_count) - parameter * matrix
    inverse = np.linalg.inv(filtering)
    transformed = matrix @ inverse
    trace_one = np.trace(transformed)
    trace_two = np.trace(transformed @ transformed)
    trace_three = np.trace(transformed.T @ transformed)
    if kind == 'lag':
        # Full expected information on (beta, rho, sigma2), prior to nuisance elimination.
        weighted_prediction = transformed @ design @ coefficients
        information = np.zeros((width + 2, width + 2))
        information[:width, :width] = design.T @ design / variance
        information[:width, width] = design.T @ weighted_prediction / variance
        information[width, :width] = information[:width, width]
        information[width, width] = trace_two + trace_three + weighted_prediction @ weighted_prediction / variance
        information[width, -1] = information[-1, width] = trace_one / variance
        information[-1, -1] = row_count / (2 * variance ** 2)
        standard_errors = np.sqrt(np.diag(np.linalg.inv(information))[:-1])
        residuals = filtering @ response - design @ coefficients
        spatial_correlation = np.corrcoef(response, inverse @ design @ coefficients)[0, 1] ** 2
    else:
        nuisance_information = np.array([[trace_two + trace_three, trace_one / variance],
                                         [trace_one / variance, row_count / (2 * variance ** 2)]])
        standard_errors = np.append(np.sqrt(variance * np.diag(np.linalg.inv(filtered_design.T @ filtered_design))),
                                    np.sqrt(np.linalg.inv(nuisance_information)[0, 0]))
        residuals = response - design @ coefficients
        spatial_correlation = None
    estimates = np.append(coefficients, parameter)
    statistics = estimates / standard_errors
    table = np.column_stack((estimates, standard_errors, statistics, 2 * norm.sf(np.abs(statistics)))).ravel()
    centered = design.copy()
    centered[:, 1:] -= predictors.mean(axis=0)
    if kind == 'error':
        centered = filtering @ centered
    normalized = centered / np.linalg.norm(centered, axis=0)
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    raw_singular_values = np.linalg.svd(filtered_design / np.linalg.norm(filtered_design, axis=0), compute_uv=False)
    conditioning = [raw_singular_values[0] / raw_singular_values[-1],
                    (singular_values[0] / singular_values[-1]) ** 2,
                    np.diag(np.linalg.inv(normalized.T @ normalized))[1:].max()]
    log_likelihood = objective - .5 * row_count * (np.log(2 * np.pi) + 1)
    parameter_count = width + (kind == 'lag')
    summary = [row_count, parameter, variance, log_likelihood,
               2 * parameter_count - 2 * log_likelihood, parameter_count * np.log(row_count) - 2 * log_likelihood,
               np.corrcoef(response, response - residuals)[0, 1] ** 2, spatial_correlation,
               residuals @ residuals, *conditioning, 0, bound, 0, 1, 0, 1,
               row_count * abs(parameter * infinity_norm) ** 65 / (65 * (1 - abs(parameter * infinity_norm))),
               1, 1, np.linalg.slogdet(filtering)[1]]
    # For this fixture |rho*s| is small enough that all 64-term omissions are below 1e-13.
    assert summary[18] < 1e-13
    powers = [np.linalg.matrix_power(matrix / infinity_norm, power) for power in range(1, 65)]
    result = dict(table=table.tolist(), summary=summary, residuals=residuals.tolist(),
                  powerTraceStandardErrors=[0.] * 64, status=[0], converged=[1])
    if kind == 'lag':
        result.update(powerTraces=[np.trace(power) for power in powers],
                      powerSums=[power.sum() for power in powers], infinityNorm=[infinity_norm])
    return result


generator = np.random.default_rng(8054)
matrix = np.zeros((12, 12))
for row in range(10):
    matrix[row, (row + 1) % 10] = matrix[(row + 1) % 10, row] = .5
matrix[0, 10] = matrix[10, 0] = .25
# Row 11 is an island; tied integer predictor rows exercise identity/order errors.
predictors = generator.integers(-3, 4, size=(12, 2)).astype(float)
predictors[3] = predictors[2]
response = np.linalg.solve(np.eye(12) - .35 * matrix,
                           .25 + predictors @ [.4, -.3] + generator.normal(scale=.7, size=12))
response = response.astype(np.float32).astype(float)
offsets, neighbors, weights = [0], [], []
for row in matrix:
    for column, value in enumerate(row):
        if value:
            neighbors.append(column)
            weights.append(value)
    offsets.append(len(neighbors))
fixture = dict(weights=dict(offsets=offsets, neighbors=neighbors, weights=weights),
               predictors=predictors.ravel().tolist(), response=response.tolist(), expected={})
for kind in ['lag', 'error']:
    fixture['expected'][kind] = fit_model(matrix, predictors, response, kind)
    # Predictor translation, response rescaling, and reciprocal parameter rescaling for W.
    fixture['expected'][kind + 'Transformed'] = fit_model(2 * matrix, predictors + [4, -2], 4 * response, kind)
print(json.dumps(fixture, indent=2, allow_nan=False))
