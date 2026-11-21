# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/panel-regression-audit.oracle.py

Independent 2SLS via SVD projection, FE coefficients/covariance via full entity dummies.
Pooled sigma2=RSS/n; FE sigma2=RSS/(n-N-k-1). Both report normal two-sided p values.
Pooled LM tests use OLS RSS/(n-k-1), trace T tr(W'W+WW), and the residual
projection of W fitted on [1,X]. All output ordering is period-major.
"""
import json
import numpy as np
from scipy.stats import norm, chi2

PREDICTORS = np.array([-2, 1, 1, 3, 0, -1, 2, -2, 1, 3, -2, 0, 2, -2, 0, 1], dtype=float)
RESPONSE = np.array([1, 4, 2, 8, 3, -1, 6, 1, 4, 5, -2, 2, 7, -3, 4, 3], dtype=float)
WEIGHTS = np.array([[0, .5, 0, .5], [.5, 0, .5, 0], [0, .5, 0, .5], [.5, 0, .5, 0]])


def condition_correlation(design):
    centered = design - design.mean(axis=0)
    normalized = centered / np.linalg.norm(centered, axis=0)
    correlation = normalized.T @ normalized
    return np.linalg.cond(normalized)**2, np.diag(np.linalg.inv(correlation)).max()


def fit_iv(unit, island=False):
    weights = WEIGHTS.copy()
    if island:
        weights[0] = 0
    block = np.kron(np.eye(4), weights)
    dummies = np.tile(np.eye(4), (4, 1)) if unit else np.ones((16, 1))
    design = np.column_stack((dummies, PREDICTORS, block @ RESPONSE))
    instruments = np.column_stack((dummies, PREDICTORS, block @ PREDICTORS))
    projected = instruments @ np.linalg.lstsq(instruments, design, rcond=None)[0]
    estimate = np.linalg.lstsq(projected, RESPONSE, rcond=None)[0]
    residual = RESPONSE - design @ estimate
    squares = residual @ residual
    variance = squares / (10 if unit else 16)
    covariance = variance * np.linalg.inv(projected.T @ projected)
    standard_errors = np.sqrt(np.diag(covariance))
    statistics = estimate / standard_errors
    table = np.column_stack((estimate, standard_errors, statistics, 2 * norm.sf(np.abs(statistics))))
    if unit:
        table = np.vstack((np.full((1, 4), np.nan), table[-2:]))
    demean = np.eye(16) - np.kron(np.ones((4, 4)) / 4, np.eye(4)) if unit else np.eye(16)
    within_response = demean @ RESPONSE
    within_predictors = demean @ PREDICTORS
    regressors = np.column_stack((within_predictors, block @ within_response))
    within_instruments = np.column_stack((np.ones(16), within_predictors, block @ within_predictors))
    projected_regressors = within_instruments @ np.linalg.lstsq(within_instruments, regressors, rcond=None)[0]
    condition, inflation = condition_correlation(projected_regressors)
    raw_design = np.column_stack((np.ones(16), projected_regressors))
    belsley = np.linalg.cond(raw_design / np.linalg.norm(raw_design, axis=0))
    instrument_condition, _ = condition_correlation(within_instruments[:, 1:])
    first_target = block @ within_response
    first_full = first_target - within_instruments @ np.linalg.lstsq(within_instruments, first_target, rcond=None)[0]
    restricted = within_instruments[:, :2]
    first_restricted = first_target - restricted @ np.linalg.lstsq(restricted, first_target, rcond=None)[0]
    excluded_squares = first_restricted @ first_restricted - first_full @ first_full
    first_statistic = excluded_squares / ((first_full @ first_full) / (10 if unit else 13))
    partial = excluded_squares / (first_restricted @ first_restricted)
    if unit:
        moran, statistic, probability = [np.nan] * 3
    else:
        lag_product = residual @ block @ residual
        moran = 16 / block.sum() * lag_product / squares
        moment = residual @ block @ design
        adjustment = moment @ np.linalg.solve(projected.T @ projected, moment)
        trace = np.trace(block @ block) + np.sum(block**2)
        statistic = (lag_product / variance)**2 / (trace + 4 * adjustment / variance)
        probability = chi2.sf(statistic, 1)
    summary = [16, variance, squares, np.corrcoef(within_response, within_response-residual)[0, 1]**2,
               moran, statistic, probability, belsley, condition, inflation, 0, instrument_condition,
               int(abs(estimate[-1]) >= 1), 1, first_statistic, partial]
    return dict(table=table.ravel().tolist(), summary=summary, residuals=residual.tolist())


def fit_diagnostics(intermediates=False):
    block = np.kron(np.eye(4), WEIGHTS)
    design = np.column_stack((np.ones(16), PREDICTORS))
    fitted = design @ np.linalg.lstsq(design, RESPONSE, rcond=None)[0]
    residual = RESPONSE - fitted
    lag_prediction = block @ fitted
    projected = lag_prediction - design @ np.linalg.lstsq(design, lag_prediction, rcond=None)[0]
    variance = residual @ residual / 14
    if intermediates:
        return dict(fitted=fitted.tolist(), residuals=residual.tolist(), projected=projected.tolist(), variance=variance)
    trace = np.trace(block @ block) + np.sum(block**2)
    information = projected @ projected / variance + trace
    residual_lag = residual @ block @ residual / variance
    response_lag = residual @ block @ RESPONSE / variance
    statistics = [residual_lag**2/trace, (response_lag-residual_lag)**2/(information-trace),
                  (residual_lag-trace/information*response_lag)**2/(trace*(1-trace/information))]
    return [value for statistic in statistics for value in [statistic, chi2.sf(statistic, 1)]]


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return None if isinstance(value, float) and not np.isfinite(value) else value


if __name__ == '__main__':
    print(json.dumps(finite_json(dict(pooled=fit_iv(False), unit=fit_iv(True),
                                     islandPooled=fit_iv(False, True), islandUnit=fit_iv(True, True),
                                     diagnostics=fit_diagnostics(), diagnosticInputs=fit_diagnostics(True)))))
