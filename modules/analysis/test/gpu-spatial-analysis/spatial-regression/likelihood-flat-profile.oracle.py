# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/likelihood-flat-profile.oracle.py
Prints likelihood-flat-profile-fixtures.json. No production helpers are imported.

W=cI gives e=(1-c*lambda)(y-Xb). Its concentrated likelihood is identically
constant: -n/2*log((1-c*lambda)^2*SSE/n)+n*log|1-c*lambda|.
The declared lowest-parameter tie convention chooses -.99/|c|. The full
(lambda,sigma2) information matrix has determinant zero, so lambda inference
is undefined for EVERY Q. Coefficient covariance remains sigma0^2*(X'X)^-1.
NumPy least squares, SVD and covariance below independently supply all ports.
"""

import json
import numpy as np
from scipy.stats import norm


def describe(maximum_power, scalar_weight, transformed):
    row_count = 8
    predictor = np.arange(row_count, dtype=float)
    response = predictor ** 2
    if transformed:
        predictor += 16
        response = 4 * response + 32
    design = np.column_stack((np.ones(row_count), predictor))
    coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
    residuals = response - design @ coefficients
    residual_squares = residuals @ residuals
    parameter = -.99 / abs(scalar_weight)
    filtering_scale = 1 - parameter * scalar_weight
    variance = residual_squares / row_count * filtering_scale ** 2
    coefficient_errors = np.sqrt(residual_squares / row_count * np.diag(np.linalg.inv(design.T @ design)))
    statistics = coefficients / coefficient_errors
    table = np.column_stack((coefficients, coefficient_errors, statistics, 2 * norm.sf(np.abs(statistics)))).ravel().tolist()
    table += [parameter, None, None, None]
    normalized = design / np.linalg.norm(design, axis=0)
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    log_likelihood = -.5 * row_count * (np.log(residual_squares / row_count) + np.log(2 * np.pi) + 1)
    # The spatial/variance block is singular: I_ll*I_ss - I_ls^2 is exactly zero.
    summary = [row_count, parameter, variance, log_likelihood,
               4 - 2 * log_likelihood, 2 * np.log(row_count) - 2 * log_likelihood,
               np.corrcoef(response, design @ coefficients)[0, 1] ** 2, None,
               residual_squares, singular_values[0] / singular_values[-1], 1, 1, 0,
               .99 / abs(scalar_weight), 0, 0, 1, 0, 0, 0, 1,
               row_count * np.log(abs(filtering_scale))]
    return dict(maximumPower=maximum_power, scalarWeight=scalar_weight, transformed=transformed,
                predictors=predictor.tolist(), response=response.tolist(),
                expected=dict(table=table, summary=summary, residuals=residuals.tolist(),
                              powerTraceStandardErrors=[0.] * maximum_power, status=[0], converged=[0]))


fixtures = [describe(maximum_power, scalar_weight, transformed)
            for maximum_power in [3, 4, 17, 128]
            for scalar_weight, transformed in [(3., False), (6., True), (-3., False)]]
print(json.dumps(fixtures, indent=2, allow_nan=False))
