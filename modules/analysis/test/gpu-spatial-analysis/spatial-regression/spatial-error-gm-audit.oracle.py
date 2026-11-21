# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/spatial-error-gm-audit.oracle.py

Print independent fixtures: dense quadratic forms, NumPy SVD, SciPy scalar optimization.
No production code or implementation outputs are read. The stochastic trace convention uses
Random123 Philox4x32-10 (Salmon et al.), counter (probe, row//4, 0, 'PROB'), key (seed, 'TRAC').
"""
import json
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm


def probe_value(seed, probe, row):
    words = [probe, row // 4, 0, 0x50524F42]
    key = [seed, 0x54524143]
    for _ in range(10):
        first = 0xD2511F53 * words[0]
        second = 0xCD9E8D57 * words[2]
        words = [(second >> 32) ^ words[1] ^ key[0], second & 0xFFFFFFFF,
                 (first >> 32) ^ words[3] ^ key[1], first & 0xFFFFFFFF]
        key = [(key[0] + 0x9E3779B9) & 0xFFFFFFFF,
               (key[1] + 0xBB67AE85) & 0xFFFFFFFF]
    return 1 - 2 * (words[row % 4] & 1)


def minimize(objective, bound):
    # Independently bracket all local basins of the quartic, then Brent search each.
    grid = np.linspace(-bound, bound, 8193)
    values = [objective(value) for value in grid]
    candidates = [(values[0], grid[0]), (values[-1], grid[-1])]
    for position in range(1, len(grid) - 1):
        if values[position] < min(values[position - 1], values[position + 1]):
            result = minimize_scalar(objective, method='bounded',
                                     bounds=(grid[position - 1], grid[position + 1]),
                                     options={'xatol': 1e-15})
            candidates.append((result.fun, result.x))
    return min(candidates)[1]


def fixture(estimator, mode='exact', shifted=False):
    count = 12
    matrix = np.zeros((count, count))
    for row in range(count - 1):  # row 11 is an island
        matrix[row, (row + 1) % count] = 0.5
        matrix[row, (row + 3) % count] = 0.25
    predictors = np.array([[-2, 0], [-1, 1], [0, -1], [1, 2], [2, -2], [2, 1],
                           [-2, 2], [-1, -1], [0, 0], [1, -2], [2, 1], [0, 2]], dtype=float)
    response = 1 + predictors @ [0.75, -0.5] + np.array(
        [0.5, 1, -0.25, 0.75, 1.5, -1, 0.25, -0.5, 1.25, -0.75, 0.5, -1.5])
    if shifted:
        predictors += [16, -8]
        response = response * 2 + 32
    design = np.column_stack([np.ones(count), predictors])
    initial = np.linalg.lstsq(design, response, rcond=None)[0]
    residuals = response - design @ initial
    bound = 0.99 / np.linalg.norm(matrix, ord=np.inf)
    gram = matrix.T @ matrix
    ratio = np.trace(gram) / count
    scale = 1 + ratio ** 2
    moments = [(gram - ratio * np.eye(count)) / scale, (matrix + matrix.T) / 2]
    variance_column = np.array([1, ratio, 0])

    def objective(values, weight=None):
        def evaluate(parameter):
            filtered = values - parameter * matrix @ values
            if estimator == 'homoskedastic':
                residual = np.array([filtered @ moment @ filtered / count for moment in moments])
                return residual @ (np.eye(2) if weight is None else weight) @ residual
            lagged = matrix @ filtered
            residual = np.array([filtered @ filtered, lagged @ lagged, filtered @ lagged]) / (values @ values)
            variance = max(0, variance_column @ residual / (variance_column @ variance_column))
            residual -= variance * variance_column
            return residual @ residual
        return evaluate

    first_objective = objective(residuals)
    first_parameter = minimize(first_objective, bound)
    filtered_design = design - first_parameter * matrix @ design
    coefficients = np.linalg.lstsq(filtered_design, response - first_parameter * matrix @ response, rcond=None)[0]
    residuals = response - design @ coefficients
    parameter = first_parameter
    lambda_error = np.nan
    final_objective = first_objective

    def trace(value):
        if mode == 'exact':
            return np.trace(value)
        probes = np.array([[probe_value(17, probe, row) for row in range(count)] for probe in range(64)])
        return np.mean(np.einsum('bi,ij,bj->b', probes, value, probes))

    # Exact parts of each estimator stay exact; only these three traces use probes.
    trace_terms = [(trace(gram @ gram) - count * ratio ** 2) / scale ** 2,
                   (2 * trace(gram @ matrix) - 2 * ratio * trace(matrix)) / scale,
                   trace(matrix @ matrix) + np.trace(gram),
                   np.diag(moments[0]) @ np.diag(moments[0])]

    def covariance(filtered):
        variance = np.mean(filtered ** 2)
        fourth = np.mean(filtered ** 4)
        return np.array([[2 * variance ** 2 * trace_terms[0] + (fourth - 3 * variance ** 2) * trace_terms[3],
                          variance ** 2 * trace_terms[1]],
                         [variance ** 2 * trace_terms[1], variance ** 2 * trace_terms[2]]]) / count

    if estimator == 'homoskedastic':
        weight = np.linalg.inv(covariance(residuals - first_parameter * matrix @ residuals))
        final_objective = objective(residuals, weight)
        parameter = minimize(final_objective, bound)
        filtered = residuals - parameter * matrix @ residuals
        jacobian = np.array([-2 * (matrix @ residuals) @ moment @ filtered / count for moment in moments])
        lambda_error = np.sqrt(1 / (jacobian @ np.linalg.solve(covariance(filtered), jacobian)) / count)
    filtered = residuals - parameter * matrix @ residuals
    variance = filtered @ filtered / count
    final_design = design - parameter * matrix @ design
    errors = np.r_[np.sqrt(np.diag(variance * np.linalg.inv(final_design.T @ final_design))), lambda_error]
    estimates = np.r_[coefficients, parameter]
    statistics = estimates / errors
    table = np.column_stack([estimates, errors, statistics, 2 * norm.sf(np.abs(statistics))]).ravel()
    normalized = final_design / np.linalg.norm(final_design, axis=0)
    centered_design = design.copy()
    centered_design[:, 1:] -= predictors.mean(axis=0)
    centered_design -= parameter * matrix @ centered_design
    centered_normalized = centered_design / np.linalg.norm(centered_design, axis=0)
    condition = np.linalg.cond(centered_normalized) ** 2
    summary = [count, parameter, variance, np.corrcoef(response, design @ coefficients)[0, 1] ** 2,
               residuals @ residuals, final_objective(parameter), np.linalg.cond(normalized), condition,
               np.diag(np.linalg.inv(centered_normalized.T @ centered_normalized))[1:].max(),
               float(condition > 1e4), bound, float(mode == 'stochastic'), float(abs(parameter) >= bound * 0.9999)]
    offsets, neighbors, weights = [0], [], []
    for row in matrix:
        columns = np.flatnonzero(row)
        neighbors.extend(columns.tolist())
        weights.extend(row[columns].tolist())
        offsets.append(len(neighbors))
    encode = lambda values: [float(value) if np.isfinite(value) else None for value in values]
    return dict(name=f'{estimator}-{mode}-' + ('shifted' if shifted else 'ordinary'),
                estimator=estimator, mode=mode, seed=17, probeCount=64,
                predictors=predictors.ravel().tolist(), response=response.tolist(),
                weights=dict(offsets=offsets, neighbors=neighbors, weights=weights),
                expected=dict(table=encode(table), summary=encode(summary), residuals=encode(residuals), status=[0]))


print(json.dumps([fixture(estimator, mode, shifted)
                  for estimator, mode in [('homoskedastic', 'exact'), ('homoskedastic', 'stochastic'),
                                          ('kelejian-prucha', 'exact')]
                  for shifted in [False, True]], indent=2))
