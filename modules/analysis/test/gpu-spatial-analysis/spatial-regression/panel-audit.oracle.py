# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/panel-audit.oracle.py

Independent Gaussian likelihood: A=I-rho W, Q_phi=I-(1-phi)P_entity.
For each (rho,phi), project Q_phi A y on Q_phi X using NumPy SVD.
Maximize T log|A| + N log(phi) - NT/2 log(RSS/NT) with SciPy,
profiling phi by a separate scalar minimization (no alternating normal equations).
The published spreg convention omits N log(phi) and adds -NT/2 log(NT) for FE.
Conditional expected information is [X,Q_phi W A^-1 X beta]'[...] / sigma2,
with T*(tr((W A^-1)^2)+tr((W A^-1)' W A^-1)) added to its rho corner.
The tied spectrum of the four-cycle is [-1,0,0,1]. No mask is supported;
an island fails the required row-normalization condition instead of being dropped.
"""
import json
import numpy as np
from scipy.optimize import minimize_scalar, brentq
from scipy.stats import chi2

ENTITY_COUNT, PERIOD_COUNT = 4, 4
WEIGHTS = np.array([[0, .5, 0, .5], [.5, 0, .5, 0], [0, .5, 0, .5], [.5, 0, .5, 0]])
PREDICTORS = np.array([-2, 1, 1, 3, 0, -1, 2, -2, 1, 3, -2, 0, 2, -2, 0, 1], dtype=float)
RESPONSE = np.array([1, 4, 2, 8, 3, -1, 6, 1, 4, 5, -2, 2, 7, -3, 4, 3], dtype=float)


def fit_panel(unit, shift=0, scale=1):
    predictors = PREDICTORS + shift
    response = scale * RESPONSE + shift
    row_count = len(response)
    design = np.column_stack((np.ones(row_count), predictors))[:, int(unit):]
    lag = (response.reshape(PERIOD_COUNT, ENTITY_COUNT) @ WEIGHTS.T).ravel()
    projection = np.kron(np.ones((PERIOD_COUNT, PERIOD_COUNT)) / PERIOD_COUNT, np.eye(ENTITY_COUNT))

    def evaluate(rho, phi):
        transform = np.eye(row_count) - (1 - phi) * projection
        transformed_design = transform @ design
        transformed_target = transform @ (response - rho * lag)
        estimate = np.linalg.lstsq(transformed_design, transformed_target, rcond=None)[0]
        residuals = transformed_target - transformed_design @ estimate
        variance = residuals @ residuals / row_count
        determinant = np.linalg.slogdet(np.eye(ENTITY_COUNT) - rho * WEIGHTS)[1]
        score = PERIOD_COUNT * determinant - row_count / 2 * np.log(variance)
        if not unit:
            score += ENTITY_COUNT * np.log(phi)
        return score, estimate, variance, transform, transformed_design, residuals

    def choose_phi(rho):
        if unit:
            return 0
        optimum = minimize_scalar(lambda value: -evaluate(rho, np.exp(value))[0],
                                  bounds=(-18, 0), method='bounded', options={'xatol': 1e-12})
        return max([np.exp(optimum.x), 1], key=lambda value: evaluate(rho, value)[0])

    grid = np.linspace(-.999, .999, 101)
    best = np.argmax([evaluate(rho, choose_phi(rho))[0] for rho in grid])
    assert 0 < best < len(grid) - 1
    optimum = minimize_scalar(lambda rho: -evaluate(rho, choose_phi(rho))[0],
                              bounds=(grid[best - 1], grid[best + 1]), method='bounded',
                              options={'xatol': 1e-12})
    rho = optimum.x
    phi = choose_phi(rho)
    score, estimate, variance, transform, transformed_design, residuals = evaluate(rho, phi)
    product = WEIGHTS @ np.linalg.inv(np.eye(ENTITY_COUNT) - rho * WEIGHTS)
    lag_prediction = (transformed_design @ estimate).reshape(PERIOD_COUNT, ENTITY_COUNT) @ product.T
    information_design = np.column_stack((transformed_design, lag_prediction.ravel()))
    information = information_design.T @ information_design / variance
    information[-1, -1] += PERIOD_COUNT * (np.trace(product @ product) + np.sum(product**2))
    covariance = np.linalg.inv(information)
    if unit:
        expanded = np.full((3, 3), np.nan)
        expanded[1:, 1:] = covariance
        covariance = expanded
        estimate = np.r_[np.nan, estimate]
    likelihood = score - row_count / 2 * (1 + np.log(2 * np.pi))
    likelihood -= row_count / 2 * np.log(row_count) if unit else ENTITY_COUNT * np.log(phi)
    return dict(coefficients=np.r_[estimate, rho, phi].tolist(), covariance=covariance.ravel().tolist(),
                summary=[variance, likelihood], fitted=(transform @ response - residuals).tolist(),
                residuals=residuals.tolist())


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return None if isinstance(value, float) and not np.isfinite(value) else value


def profile_reference(phi):
    """SVD residual profile at a fixed phi, for the emitted WGSL search regression."""
    transform = np.eye(16) - (1-phi)*np.kron(np.ones((4,4))/4, np.eye(4))
    design = transform @ np.column_stack((np.ones(16), PREDICTORS))
    target = transform @ RESPONSE
    lag = transform @ (RESPONSE.reshape(4,4) @ WEIGHTS.T).ravel()
    first = target - design @ np.linalg.lstsq(design, target, rcond=None)[0]
    second = lag - design @ np.linalg.lstsq(design, lag, rcond=None)[0]
    squares = second @ second
    minimum = first @ second / squares
    remainder = first - minimum*second
    floor = remainder @ remainder
    # W has eigenvalues [-1,0,0,1], so log determinant is log(1-rho^2).
    root = brentq(lambda rho: 16*squares*(rho-minimum)/(floor+squares*(rho-minimum)**2) + 8*rho/(1-rho**2), -.5, 0)
    probes = []
    for rho in [-.5, 0, .5]:
        trace = 2*rho/(1-rho*rho)
        score = 16*(rho-minimum)/(floor/squares+(rho-minimum)**2) + 4*trace
        probes.append(dict(rho=rho, trace=trace, score=score))
    return dict(direct=[floor, squares, minimum, 0], rho=root, probes=probes)


if __name__ == '__main__':
    unit, random = fit_panel(True), fit_panel(False)
    difference = np.array(random['covariance']).reshape(3, 3)[1:, 1:] - np.array(unit['covariance']).reshape(3, 3)[1:, 1:]
    contrast = np.array(unit['coefficients'][1:3]) - np.array(random['coefficients'][1:3])
    assert np.linalg.eigvalsh(difference).min() > 0
    statistic = contrast @ np.linalg.solve(difference, contrast)
    random['hausman'] = [statistic, 3, chi2.sf(statistic, 3)]
    print(json.dumps(finite_json(dict(unit=unit, random=random, shiftedUnit=fit_panel(True, 16, 2),
                                     shiftedRandom=fit_panel(False, 16, 2),
                                     referenceProfile=profile_reference(random['coefficients'][3]))), indent=2))
