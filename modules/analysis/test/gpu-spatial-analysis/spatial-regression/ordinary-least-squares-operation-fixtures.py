# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Generate independent regression fixtures with SciPy QR/SVD and distribution functions.

Run with /usr/bin/python3 (without -I). Inputs are quantized to float32 before fitting.
No production twin or existing TypeScript test reference is used.
"""

import json
from pathlib import Path

import numpy
from scipy import linalg, stats


def create_fixture(name, ridge_lambda=0, offset=0):
    row = numpy.arange(32, dtype=float)
    predictors = numpy.column_stack((
        2 + numpy.sin(row * 1.7) + row / 20 + offset,
        -1 + numpy.cos(row * 0.7) + (row % 5) / 4 - offset / 2,
    )).astype(numpy.float32).astype(float)
    response = (3 + predictors @ numpy.array([0.8, -0.6])
                + (0.15 + row / 45) * (numpy.sin(row * 2.1) + 0.3 * numpy.cos(row * 0.4))
                + offset).astype(numpy.float32).astype(float)
    mask = numpy.where(row % 7 == 3, 0, 1)
    used = mask != 0
    design = numpy.column_stack((numpy.ones(used.sum()), predictors[used]))
    observed = response[used]
    augmented = numpy.vstack((design, [[0, numpy.sqrt(ridge_lambda), 0], [0, 0, numpy.sqrt(ridge_lambda)]]))
    coefficients = linalg.lstsq(augmented, numpy.r_[observed, 0, 0], lapack_driver='gelsy')[0]
    residual = observed - design @ coefficients
    fitted = design @ coefficients
    row_count, coefficient_count = design.shape
    residual_sum_squares = residual @ residual
    total_sum_squares = numpy.sum((observed - observed.mean()) ** 2)
    sigma_squared = residual_sum_squares / (row_count - coefficient_count)
    _, triangular = linalg.qr(augmented, mode='economic')
    inverse_factor = linalg.solve_triangular(triangular, numpy.eye(coefficient_count))
    covariance = inverse_factor @ inverse_factor.T
    standard_errors = numpy.sqrt(sigma_squared * numpy.diag(covariance))
    squared = residual ** 2
    auxiliary_coefficients = linalg.lstsq(design, squared, lapack_driver='gelsy')[0]
    auxiliary_residual = squared - design @ auxiliary_coefficients
    auxiliary_r_squared = 1 - (auxiliary_residual @ auxiliary_residual) / numpy.sum((squared - squared.mean()) ** 2)
    koenker_bassett = row_count * auxiliary_r_squared
    kurtosis = numpy.mean(residual ** 4) / numpy.mean(squared) ** 2
    skewness = numpy.mean(residual ** 3) / numpy.mean(squared) ** 1.5
    jarque_bera = row_count / 6 * (skewness ** 2 + (kurtosis - 3) ** 2 / 4)
    breusch_pagan = 0.5 * auxiliary_r_squared * row_count * (kurtosis - 1)
    log_likelihood = -0.5 * row_count * (numpy.log(2 * numpy.pi) + numpy.log(residual_sum_squares / row_count) + 1)
    r_squared = 1 - residual_sum_squares / total_sum_squares
    singular_values = linalg.svdvals(design / linalg.norm(design, axis=0))
    centered = predictors[used] - predictors[used].mean(axis=0)
    scales = linalg.norm(centered, axis=0)
    correlation = (centered.T @ centered + ridge_lambda * numpy.eye(2)) / numpy.outer(scales, scales)
    eigenvalues = linalg.eigvalsh(correlation)
    correlation_condition = eigenvalues[-1] / eigenvalues[0]
    summary = [
        row_count, r_squared, 1 - (1 - r_squared) * (row_count - 1) / (row_count - coefficient_count),
        sigma_squared, log_likelihood, 2 * coefficient_count - 2 * log_likelihood,
        coefficient_count * numpy.log(row_count) - 2 * log_likelihood,
        jarque_bera, stats.chi2.sf(jarque_bera, 2), koenker_bassett,
        stats.chi2.sf(koenker_bassett, 2), residual_sum_squares, total_sum_squares,
        skewness, kurtosis, ridge_lambda, breusch_pagan, stats.chi2.sf(breusch_pagan, 2),
        singular_values[0] / singular_values[-1], correlation_condition,
        numpy.max(numpy.diag(linalg.inv(correlation))), int(correlation_condition > 1e4)
    ]
    residuals = [None] * len(row)
    fitted_values = [None] * len(row)
    for position, original in enumerate(numpy.flatnonzero(used)):
        residuals[original] = float(residual[position])
        fitted_values[original] = float(fitted[position])
    return dict(name=name, ridgeLambda=ridge_lambda, predictors=predictors.ravel().tolist(),
                response=response.tolist(), mask=mask.tolist(), expected=dict(
                    coefficients=coefficients.tolist(), standardErrors=standard_errors.tolist(),
                    tStatistics=(coefficients / standard_errors).tolist(), summary=summary,
                    residuals=residuals, fitted=fitted_values))


fixtures = [create_fixture('ordinary'), create_fixture('ridge', ridge_lambda=2),
            create_fixture('large-mean', offset=4096)]
Path(__file__).with_name('ordinary-least-squares-operation-fixtures.json').write_text(
    json.dumps(fixtures, indent=2, allow_nan=False) + '\n')
