# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/rr-weights-regression-review2.oracle.py
Independent dense NumPy/SciPy diagnostics, IV sandwich, and profiled panel likelihood.
"""
import json
import numpy as np
from scipy.optimize import minimize_scalar

predictors = np.array([0, 1, 4, 2, 5, 3, 7, 6], dtype=float)
response = np.array([1, 3, 2, 5, 4, 8, 7, 10], dtype=float)
design = np.column_stack([np.ones(8), predictors])
matrix = np.roll(np.eye(8), 1, axis=1)
ordinary = np.linalg.lstsq(design, response, rcond=None)[0]
residuals = response - design @ ordinary
instruments = np.column_stack([design, matrix @ predictors])
lag_design = np.column_stack([design, matrix @ response])
projector = instruments @ np.linalg.pinv(instruments)
normal_inverse = np.linalg.inv(lag_design.T @ projector @ lag_design)
coefficients = normal_inverse @ lag_design.T @ projector @ response
scores = instruments * (response - lag_design @ coefficients)[:, None]
mapping = normal_inverse @ lag_design.T @ instruments @ np.linalg.inv(instruments.T @ instruments)
covariance = mapping @ scores.T @ scores @ mapping.T
# Local Moran uses sample scaling: I_i = (n - 1) c_i (Wc)_i / sum(c^2).
# OLS residuals here are exactly [-49, 1, -143, 51, -93, 143, -35, 125] / 42,
# have zero mean, and sum(c^2) = 850/21. These rational values independently pin the scale.
centered_residuals = residuals - residuals.mean()
local_i = centered_residuals * (matrix @ centered_residuals) / np.var(centered_residuals, ddof=1)
exact_local_i = np.array([-49/10200, -143/10200, -143/200, -93/200,
                          -4433/3400, -1001/2040, -175/408, -245/408])
np.testing.assert_allclose(local_i, exact_local_i, rtol=0, atol=1e-13)
entity_matrix = np.roll(np.eye(4), 1, axis=1)
panel_matrix = np.kron(np.eye(2), entity_matrix)
within_response = (response.reshape(2, 4) - response.reshape(2, 4).mean(axis=0)).ravel()
within_predictors = (predictors.reshape(2, 4) - predictors.reshape(2, 4).mean(axis=0)).ravel()
def profile(parameter):
    filtered_predictor = within_predictors - parameter * panel_matrix @ within_predictors
    filtered_response = within_response - parameter * panel_matrix @ within_response
    coefficient = filtered_predictor @ filtered_response / (filtered_predictor @ filtered_predictor)
    errors = filtered_response - coefficient * filtered_predictor
    return 4 * np.log(errors @ errors / 8) - 2 * np.linalg.slogdet(np.eye(4) - parameter * entity_matrix)[1]
grid = np.linspace(-0.99, 0.99, 4097)
values = np.array([profile(value) for value in grid])
candidates = [(values[0], grid[0]), (values[-1], grid[-1])]
for index in range(1, len(grid) - 1):
    if values[index] < min(values[index - 1], values[index + 1]):
        fit = minimize_scalar(profile, bounds=(grid[index - 1], grid[index + 1]), method='bounded', options={'xatol': 1e-14})
        candidates.append((fit.fun, fit.x))
parameter = min(candidates)[1]
filtered_predictor = within_predictors - parameter * panel_matrix @ within_predictors
filtered_response = within_response - parameter * panel_matrix @ within_response
coefficient = filtered_predictor @ filtered_response / (filtered_predictor @ filtered_predictor)
variance = residuals @ residuals / 6
panel_error = (residuals @ panel_matrix @ residuals / variance) ** 2 / np.trace(panel_matrix.T @ panel_matrix + panel_matrix @ panel_matrix)
print(json.dumps({'ordinaryResiduals': residuals.tolist(), 'hacCovariance': covariance.ravel().tolist(),
                  'localI': local_i.tolist(),
                  'panelCoefficients': [coefficient, parameter], 'panelLmError': panel_error}))
