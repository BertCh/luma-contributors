# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/rr-weights-regression-review.oracle.py

Independent dense NumPy IV projection and Kelejian-Prucha moment minimization.
No implementation helper or captured output is read. The GM objective eliminates its
nonnegative variance by least squares; every local basin is bracketed before SciPy minimization.
"""
import json
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

predictors = np.array([0, 1, 4, 2, 5, 3, 7, 6], dtype=float)
response = np.array([1, 3, 2, 5, 4, 8, 7, 10], dtype=float)
matrix = np.roll(np.eye(8), 1, axis=1)
design = np.column_stack([np.ones(8), predictors])
instruments = np.column_stack([design, matrix @ predictors])
lag_design = np.column_stack([design, matrix @ response])
projector = instruments @ np.linalg.pinv(instruments)
twoStageCoefficients = np.linalg.solve(lag_design.T @ projector @ lag_design,
                     lag_design.T @ projector @ response)
ordinaryCoefficients = np.linalg.lstsq(design, response, rcond=None)[0]
residuals = response - design @ ordinaryCoefficients
variance_direction = np.array([1, np.trace(matrix.T @ matrix) / 8, 0])


def objective(parameter):
    filtered = residuals - parameter * matrix @ residuals
    lagged = matrix @ filtered
    moments = np.array([filtered @ filtered, lagged @ lagged, filtered @ lagged])
    variance = max(0, variance_direction @ moments / (variance_direction @ variance_direction))
    discrepancy = moments - variance * variance_direction
    return discrepancy @ discrepancy


grid = np.linspace(-0.99, 0.99, 4097)
values = np.array([objective(parameter) for parameter in grid])
candidates = [(values[0], grid[0]), (values[-1], grid[-1])]
for index in range(1, len(grid) - 1):
    if values[index] < min(values[index - 1], values[index + 1]):
        result = minimize_scalar(objective, bounds=(grid[index - 1], grid[index + 1]),
                                 method='bounded', options={'xatol': 1e-15})
        candidates.append((result.fun, result.x))
parameter = min(candidates)[1]
filtered_design = design - parameter * matrix @ design
errorCoefficients = np.linalg.lstsq(filtered_design, response - parameter * matrix @ response, rcond=None)[0]
variance = residuals @ residuals / 8
trace = np.trace(matrix.T @ matrix + matrix @ matrix)
fitted_lag = matrix @ (design @ ordinaryCoefficients)
projected_lag = fitted_lag - design @ np.linalg.lstsq(design, fitted_lag, rcond=None)[0]
information = projected_lag @ projected_lag / variance + trace
lag_moment = residuals @ matrix @ response / variance
error_moment = residuals @ matrix @ residuals / variance
statistics = [lag_moment ** 2 / information, error_moment ** 2 / trace,
              (lag_moment - error_moment) ** 2 / (information - trace),
              (error_moment - trace / information * lag_moment) ** 2 /
              (trace * (1 - trace / information))]
print(json.dumps({'twoStageCoefficients': twoStageCoefficients.tolist(),
                  'gmCoefficients': [*errorCoefficients.tolist(), parameter],
                  'twoStageResiduals': (response - lag_design @ twoStageCoefficients).tolist(),
                  'gmResiduals': (response - design @ errorCoefficients).tolist(),
                  'lmPValues': chi2.sf(statistics, 1).tolist()}))
