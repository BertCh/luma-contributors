# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Independent dense A1/A2 matrices, NumPy SVD fits and SciPy bounded scalar search.

Run with /usr/bin/python3. Writes only the adjacent fixture. The implementation under test
uses sparse scalar moments, QR and polynomial stationary points instead.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm


def minimize_global(objective, bound):
    points = np.linspace(-bound, bound, 4097)
    values = np.array([objective(point) for point in points])
    candidates = [(values[0], points[0]), (values[-1], points[-1])]
    for position in range(1, len(points) - 1):
        if values[position] <= min(values[position - 1], values[position + 1]):
            optimum = minimize_scalar(objective, bounds=(points[position - 1], points[position + 1]),
                                      method='bounded', options={'xatol': 1e-15})
            candidates.append((optimum.fun, optimum.x))
    return min(candidates)[1]


def create_fixture(name, shifted, estimator):
    count = 18
    random = np.random.default_rng(122)
    matrix = np.zeros((count, count))
    for row in range(count):
        if shifted and row in [0, 7]:
            continue
        matrix[row, (row + 1) % count] = 0.5
        matrix[row, (row + 4) % count] = 0.25
        matrix[row, (row + 7) % count] = 0.125 if row % 3 else 0.25
    predictors = random.normal(size=(count, 2))
    if shifted:
        predictors += [4096, -256]
    predictors = predictors.astype(np.float32).astype(float)
    noise = np.linalg.solve(np.eye(count) - 0.55 * matrix, random.normal(size=count))
    response = 0.3 + predictors @ [0.7, -0.4] + noise * (0.015 if shifted else 0.5)
    if shifted:
        response += 10000
    response = response.astype(np.float32).astype(float)
    means = predictors.mean(axis=0)
    response_mean = response.mean()
    design = np.column_stack([np.ones(count), predictors - means])
    centered_response = response - response_mean
    initial = np.linalg.lstsq(design, centered_response, rcond=None)[0]
    residuals = centered_response - design @ initial
    bound = 0.99 / np.max(np.sum(np.abs(matrix), axis=1))
    trace = np.trace(matrix.T @ matrix)
    first_matrix = (matrix.T @ matrix - trace / count * np.eye(count)) / (1 + (trace / count) ** 2)
    second_matrix = (matrix + matrix.T) / 2
    moment_matrices = [first_matrix, second_matrix]

    def get_objective(values, weight=None):
        if estimator == 'homoskedastic':
            def objective(parameter):
                filtered = values - parameter * matrix @ values
                moments = np.array([filtered @ moment @ filtered / count for moment in moment_matrices])
                return moments @ (np.eye(2) if weight is None else weight) @ moments
        else:
            def objective(parameter):
                filtered = values - parameter * matrix @ values
                lagged = matrix @ filtered
                moments = np.array([filtered @ filtered, lagged @ lagged, filtered @ lagged]) / (values @ values)
                variance_column = np.array([1, trace / count, 0])
                variance = max(0, variance_column @ moments / (variance_column @ variance_column))
                differences = moments - variance * variance_column
                return differences @ differences
        return objective

    first_parameter = minimize_global(get_objective(residuals), bound)
    first_design = design - first_parameter * matrix @ design
    first_response = centered_response - first_parameter * matrix @ centered_response
    coefficients = np.linalg.lstsq(first_design, first_response, rcond=None)[0]
    residuals = centered_response - design @ coefficients
    parameter = first_parameter
    lambda_error = np.nan

    def get_variance(filtered):
        sigma_squared = np.mean(filtered ** 2)
        fourth = np.mean(filtered ** 4)
        variance = np.zeros((2, 2))
        for first in range(2):
            for second in range(2):
                variance[first, second] = (2 * sigma_squared ** 2 * np.trace(moment_matrices[first] @ moment_matrices[second]) +
                                          (fourth - 3 * sigma_squared ** 2) * np.diag(moment_matrices[first]) @ np.diag(moment_matrices[second])) / count
        return variance

    if estimator == 'homoskedastic':
        weight = np.linalg.inv(get_variance(residuals - first_parameter * matrix @ residuals))
        objective = get_objective(residuals, weight)
        parameter = minimize_global(objective, bound)
        filtered = residuals - parameter * matrix @ residuals
        derivative = np.array([-2 * (matrix @ residuals) @ moment @ filtered / count for moment in moment_matrices])
        lambda_error = np.sqrt(1 / (derivative @ np.linalg.solve(get_variance(filtered), derivative)) / count)
    else:
        objective = get_objective(centered_response - design @ initial)
    filtered = residuals - parameter * matrix @ residuals
    sigma_squared = filtered @ filtered / count
    final_design = design - parameter * matrix @ design
    covariance = sigma_squared * np.linalg.inv(final_design.T @ final_design)
    transform = np.eye(3)
    transform[0, 1:] = -means
    covariance = transform @ covariance @ transform.T
    coefficients[0] += response_mean - means @ coefficients[1:]
    estimates = np.append(coefficients, parameter)
    errors = np.append(np.sqrt(np.diag(covariance)), lambda_error)
    statistics = estimates / errors
    table = np.column_stack([estimates, errors, statistics, 2 * norm.sf(np.abs(statistics))]).ravel()
    raw_design = final_design.copy()
    raw_design[:, 1:] += final_design[:, :1] * means
    normalized = final_design / np.linalg.norm(final_design, axis=0)
    correlation_condition = np.linalg.cond(normalized) ** 2
    inflation = np.diag(np.linalg.inv(normalized.T @ normalized))[1:].max()
    fitted = response - residuals
    summary = [count, parameter, sigma_squared, np.corrcoef(response, fitted)[0, 1] ** 2,
               residuals @ residuals, objective(parameter), np.linalg.cond(raw_design / np.linalg.norm(raw_design, axis=0)),
               correlation_condition, inflation, float(correlation_condition > 1e4), bound, 0, float(abs(parameter) >= bound * 0.9999)]
    offsets = [0]
    neighbors = []
    weights = []
    for row in matrix:
        columns = np.flatnonzero(row)
        neighbors.extend(columns.tolist())
        weights.extend(row[columns].tolist())
        offsets.append(len(neighbors))
    def encode(values):
        return [float(value) if np.isfinite(value) else None for value in values]
    return {'name': name, 'estimator': estimator, 'predictorCount': 2,
            'weights': {'offsets': offsets, 'neighbors': neighbors, 'weights': weights},
            'predictors': predictors.ravel().tolist(), 'response': response.tolist(),
            'expected': {'table': encode(table), 'summary': encode(summary), 'residuals': encode(residuals)}}


fixtures = [create_fixture(('offset-islands' if shifted else 'asymmetric') + '-' + estimator, shifted, estimator)
            for shifted in [False, True] for estimator in ['homoskedastic', 'kelejian-prucha']]
Path(__file__).with_suffix('.json').write_text(json.dumps(fixtures, indent=2) + '\n')
