# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Independent oracle: dense SVD projector, normal inverse and sandwich matrices.

Run /usr/bin/python3 on this file. No first-party estimator is used.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2


def pack_weights(matrix):
    offsets = [0]
    neighbors = []
    weights = []
    for row in matrix:
        for column in np.flatnonzero(row):
            neighbors.append(int(column))
            weights.append(float(row[column]))
        offsets.append(len(neighbors))
    return {"offsets": offsets, "neighbors": neighbors, "weights": weights}


def conditioning(matrix, center):
    working = matrix[:, 1:] - matrix[:, 1:].mean(axis=0) if center else matrix
    unit = working / np.linalg.norm(working, axis=0)
    singular = np.linalg.svd(unit, compute_uv=False)
    return (singular[0] / singular[-1]) ** 2


def create_fixture(name, predictors, response, matrix, order):
    predictors = predictors.astype(np.float32).astype(float)
    response = response.astype(np.float32).astype(float)
    matrix = matrix.astype(np.float32).astype(float)
    row_count, predictor_count = predictors.shape
    instruments = np.column_stack([np.ones(row_count), predictors, matrix @ predictors])
    if order == 2:
        instruments = np.column_stack([instruments, matrix @ matrix @ predictors])
    design = np.column_stack([np.ones(row_count), predictors, matrix @ response])
    projector = instruments @ np.linalg.pinv(instruments)
    normal_inverse = np.linalg.inv(design.T @ projector @ design)
    coefficients = normal_inverse @ design.T @ projector @ response
    residuals = response - design @ coefficients
    residual_squares = residuals @ residuals
    sigma_squared = residual_squares / row_count
    errors = np.sqrt(sigma_squared * np.diag(normal_inverse))
    statistic = coefficients / errors
    table = np.column_stack([coefficients, errors, statistic, chi2.sf(statistic**2, 1)])
    total_weight = matrix.sum()
    trace = np.trace(matrix.T @ matrix + matrix @ matrix)
    product = residuals @ matrix @ residuals
    residual_design = residuals @ matrix @ design
    quadratic = residual_design @ normal_inverse @ residual_design
    anselin = (product / sigma_squared)**2 / (trace + 4 * quadratic / sigma_squared)
    projected = projector @ design
    projected_centered = projected[:, 1:] - projected[:, 1:].mean(axis=0)
    normalized = projected_centered / np.linalg.norm(projected_centered, axis=0)
    correlation_condition = conditioning(projected, True)
    instrument_condition = conditioning(instruments, True)
    inflation = np.diag(np.linalg.inv(normalized.T @ normalized)).max()
    full_residuals = design[:, -1] - projector @ design[:, -1]
    restricted = np.column_stack([np.ones(row_count), predictors])
    restricted_residuals = design[:, -1] - restricted @ np.linalg.lstsq(restricted, design[:, -1], rcond=None)[0]
    full_squares = full_residuals @ full_residuals
    restricted_squares = restricted_residuals @ restricted_residuals
    infinity_norm = abs(matrix).sum(axis=1).max()
    instrument_count = instruments.shape[1] - 1
    summary = [
        row_count, sigma_squared, residual_squares, np.corrcoef(response, response - residuals)[0, 1]**2,
        None if total_weight == 0 else row_count * product / (total_weight * residual_squares),
        anselin, chi2.sf(anselin, 1), np.sqrt(conditioning(projected, False)),
        correlation_condition, inflation, int(max(correlation_condition, instrument_condition) > 1e4),
        instrument_condition, int(abs(coefficients[-1]) * infinity_norm >= 1), infinity_norm,
        ((restricted_squares - full_squares) / (instrument_count - predictor_count)) / (full_squares / (row_count - instrument_count - 1)),
        (restricted_squares - full_squares) / restricted_squares,
    ]
    # Symmetric, unit-diagonal PSD circulant HAC kernel, independent of model weights.
    kernel = np.eye(row_count)
    for row in range(row_count):
        kernel[row, (row + 1) % row_count] = 0.125
        kernel[row, (row - 1) % row_count] = 0.125
    mapping = normal_inverse @ design.T @ instruments @ np.linalg.inv(instruments.T @ instruments)
    scores = instruments * residuals[:, None]
    robust = {}
    for method in ["HC0", "HC1", "HAC"]:
        meat = scores.T @ (kernel if method == "HAC" else np.eye(row_count)) @ scores
        covariance = mapping @ meat @ mapping.T
        if method == "HC1":
            covariance *= row_count / (row_count - design.shape[1])
        robust_errors = np.sqrt(np.diag(covariance))
        robust_statistic = coefficients / robust_errors
        robust[method] = {
            "covariance": covariance.ravel().tolist(),
            "table": np.column_stack([coefficients, robust_errors, robust_statistic, chi2.sf(robust_statistic**2, 1)]).ravel().tolist(),
        }
    return {
        "name": name, "predictorCount": predictor_count, "instrumentOrder": order,
        "predictors": predictors.ravel().tolist(), "response": response.tolist(),
        "weights": pack_weights(matrix), "kernel": pack_weights(kernel),
        "table": table.ravel().tolist(), "summary": summary, "residuals": residuals.tolist(), "robust": robust,
    }


random = np.random.default_rng(20261112)
row_count = 48
matrix = np.zeros((row_count, row_count))
for row in range(row_count):
    if row not in [3, 19]:
        matrix[row, (row + 1) % row_count] = 0.375
        matrix[row, (row + 5) % row_count] = 0.625 if row % 3 else 0.875
predictors = random.normal(size=(row_count, 2))
response = np.linalg.solve(np.eye(row_count) - 0.35 * matrix, 1 + predictors @ [1.2, -0.8] + random.normal(size=row_count))
fixtures = [create_fixture("ordinary", predictors, response, matrix, order) for order in [1, 2]]
signed_matrix = np.zeros((8, 8))
for pair, weight in enumerate([1, 1, -1, -1]):
    signed_matrix[2 * pair, 2 * pair + 1] = weight
    signed_matrix[2 * pair + 1, 2 * pair] = weight
fixtures.append(create_fixture("zero-total-weight", np.array([-2, -1, 0, 1, 2, 4, -3, 5])[:, None], np.array([1, 3, -1, 2, 7, 0, -2, 8]), signed_matrix, 1))
Path(__file__).with_suffix(".json").write_text(json.dumps(fixtures, indent=2, allow_nan=False) + "\n")
