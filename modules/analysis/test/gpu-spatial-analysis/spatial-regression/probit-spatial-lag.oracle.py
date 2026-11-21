# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent SciPy/NumPy oracle for the probit linearization specified by ST05.

Run: OPENBLAS_NUM_THREADS=1 /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/probit-spatial-lag.oracle.py
Add --simulation for 50 seeds at each dependence level (n=2000, Python only).
Use --audit for the 20-row fixture, --ring for R3, and --boundary for R7's rounding case.
No TypeScript implementation, solver or fixture is read by this script.
"""
import json
import sys
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr
from scipy.stats import norm, chi2
from scipy import sparse
from scipy.sparse.linalg import spsolve


def fit_linearization(predictors, response, matrix, diagnostics=True):
    means = predictors.mean(axis=0)
    design = np.column_stack((np.ones(len(response)), predictors - means))
    sides = 2 * response - 1

    def objective(estimate):
        signed = sides * (design @ estimate)
        ratio = np.exp(norm.logpdf(signed) - log_ndtr(signed))
        return -np.mean(log_ndtr(signed)), -design.T @ (sides * ratio) / len(response)

    def hessian(estimate):
        signed = sides * (design @ estimate)
        ratio = np.exp(norm.logpdf(signed) - log_ndtr(signed))
        return design.T @ ((ratio * (ratio + signed))[:, None] * design) / len(response)

    fit = minimize(objective, np.zeros(design.shape[1]), jac=True, hess=hessian,
                   method='trust-exact', options={'gtol': 1e-11})
    assert np.max(np.abs(objective(fit.x)[1])) < 1e-7
    linear = design @ fit.x
    probabilities = ndtr(linear).clip(1e-7, 1 - 1e-7)
    density = norm.pdf(linear)
    variance = probabilities * (1 - probabilities)
    # d Phi((I-rho W)^-1 X beta)/d rho at zero is phi(X beta) W X beta.
    # The variance of the latent error has zero derivative at zero for zero-diagonal W.
    gradient = density[:, None] * np.column_stack((design, matrix @ linear))
    target = (response - probabilities + density * linear) / np.sqrt(variance)
    gradient /= np.sqrt(variance[:, None])
    instruments = np.column_stack((design, matrix @ design, matrix @ (matrix @ design)))
    instruments[:, 1:] -= instruments[:, 1:].mean(axis=0)
    # Unweighted Z; QR/SVD least squares, independent of the implementation's Gram solves.
    projected = instruments @ np.linalg.lstsq(instruments, gradient, rcond=1e-10)[0]
    estimate = np.linalg.lstsq(projected, target, rcond=1e-10)[0]
    residual = target - gradient @ estimate  # structural residual, not target - projected @ estimate
    bread = np.linalg.inv(projected.T @ projected)
    meat = projected.T @ ((residual**2)[:, None] * projected)
    errors = np.sqrt(np.diag(bread @ meat @ bread))
    transform = np.eye(design.shape[1] + 1)
    transform[0, 1:-1] = -means
    estimate = transform @ estimate
    errors = np.sqrt(np.diag(transform @ bread @ meat @ bread @ transform.T))
    if not diagnostics:
        return {'coefficients': estimate.tolist(), 'standardErrors': errors.tolist()}
    information = design.T @ ((density**2 / (ndtr(linear) * ndtr(-linear)))[:, None] * design)
    ordinary_transform = transform[:-1, :-1]
    ordinary_estimate = ordinary_transform @ fit.x
    effects = fit.x[1:] * np.mean(density)
    naive = response - probabilities
    generalized = naive * density / variance
    standardized = naive / np.sqrt(variance)
    trace = np.trace(matrix.T @ matrix + matrix @ matrix)
    variance_trace = np.sum((matrix**2 + matrix * matrix.T) * np.outer(variance, variance))
    pinkse = (generalized @ matrix @ generalized / np.mean(density**2 / variance))**2 / trace
    kelejian = naive @ matrix @ naive / np.sqrt(variance_trace)
    slade = (standardized @ matrix @ standardized)**2 / trace
    return {
        'coefficients': estimate.tolist(), 'standardErrors': errors.tolist(),
        'status': int(abs(np.float32(estimate[-1])) >= 1) * 4,
        'ordinaryCoefficients': ordinary_estimate.tolist(),
        'ordinaryStandardErrors': np.sqrt(np.diag(ordinary_transform @ np.linalg.inv(information) @ ordinary_transform.T)).tolist(),
        'fitted': probabilities.tolist(), 'marginalEffects': effects.tolist(),
        'summary': [len(response), -fit.fun * len(response), 2 * fit.fun * len(response)],
        'tests': [pinkse, chi2.sf(pinkse, 1), kelejian, chi2.sf(kelejian**2, 1), slade, chi2.sf(slade, 1)]
    }


def fixture(count=60):
    random = np.random.default_rng(20261118)
    predictors = random.normal(size=(count, 2)).astype(np.float32).astype(float)
    predictors[3] = predictors[2]  # a tied design row with independently drawn response
    matrix = np.zeros((count, count))
    for row in range(1, count):  # row zero is an island
        matrix[row, (row + 1) % count] = 0.5
        matrix[row, (row + 9) % count] = 0.5
    latent = np.linalg.solve(np.eye(count) - 0.25 * matrix,
                             0.25 + predictors @ [0.8, -0.45] + random.normal(size=count))
    response = (latent > 0).astype(float)
    mask = np.ones(count, dtype=int)
    mask[[4, 17]] = 0
    selected = mask != 0
    expected = fit_linearization(predictors[selected], response[selected], matrix[np.ix_(selected, selected)])
    expected['fitted'] = [float('nan')] * count
    retained_fit = fit_linearization(predictors[selected], response[selected], matrix[np.ix_(selected, selected)])['fitted']
    for row, value in zip(np.flatnonzero(selected), retained_fit):
        expected['fitted'][row] = value
    expected['fitted'] = [value if np.isfinite(value) else None for value in expected['fitted']]
    offsets, neighbors, weights = [0], [], []
    for row in matrix:
        columns = np.flatnonzero(row)
        neighbors.extend(columns.tolist())
        weights.extend(row[columns].tolist())
        offsets.append(len(neighbors))
    return {'predictors': predictors.ravel().tolist(), 'response': response.tolist(),
            'mask': mask.tolist(), 'offsets': offsets, 'neighbors': neighbors,
            'weights': weights, 'expected': expected}


def simulate():
    count = 2000
    positions = np.arange(count)
    matrix = sparse.csr_matrix((np.full(4 * count, 0.25),
                                (np.repeat(positions, 4),
                                 ((positions[:, None] + [-2, -1, 1, 2]) % count).ravel())),
                               shape=(count, count))
    cases = []
    for dependence in [0.2, 0.6]:
        estimates = []
        for seed in range(50):
            random = np.random.default_rng(seed)
            predictors = random.normal(size=(count, 2))
            latent = spsolve(sparse.eye(count, format='csr') - dependence * matrix,
                              0.25 + predictors @ [0.8, -0.45] + random.normal(size=count))
            estimates.append(fit_linearization(predictors, (latent > 0).astype(float), matrix, False)['coefficients'][-1])
        standard_deviation = float(np.std(estimates, ddof=1))
        cases.append({'rho': dependence, 'mean': float(np.mean(estimates)),
                      'standardDeviation': standard_deviation,
                      'meanStandardError': standard_deviation / np.sqrt(len(estimates)),
                      'minimum': float(np.min(estimates)), 'maximum': float(np.max(estimates)),
                      'estimatesBySeed': estimates})
    return {'oracle': 'SciPy trust-exact probit; NumPy SVD projection and least squares',
            'weighting': 'residual and gradients divided by sqrt(p(1-p)); unweighted instruments',
            'versions': {'numpy': np.__version__, 'scipy': scipy.__version__},
            'rowCount': count, 'seedCount': 50, 'firstSeed': 0,
            'intercept': 0.25, 'slopes': [0.8, -0.45],
            'neighborOffsets': [-2, -1, 1, 2], 'neighborWeight': 0.25,
            'cases': cases}


def ring_fixture():
    count = 24
    predictors = np.array([(7 * row % 13) - 6 for row in range(count)], dtype=float)[:, None]
    response = np.array([float((5 * row + 7) % 17 + 0.8 * predictors[row, 0] > 7.5) for row in range(count)])
    matrix = np.roll(np.eye(count), 1, axis=1)
    return {'predictors': predictors.ravel().tolist(), 'response': response.tolist(),
            'mask': [1] * count, 'offsets': list(range(count + 1)),
            'neighbors': [(row + 1) % count for row in range(count)], 'weights': [1] * count,
            'expected': fit_linearization(predictors, response, matrix)}


def boundary_fixture():
    result = fixture()
    scale = np.float32(result['expected']['coefficients'][-1])
    result['weights'] = [float(np.float32(value * scale)) for value in result['weights']]
    result['weights'][16] = float(np.nextafter(np.float32(result['weights'][16]), np.float32(np.inf)))
    matrix = np.zeros((60, 60))
    for row in range(60):
        for slot in range(result['offsets'][row], result['offsets'][row + 1]):
            matrix[row, result['neighbors'][slot]] = result['weights'][slot]
    selected = np.array(result['mask']) != 0
    expected = fit_linearization(np.array(result['predictors']).reshape(-1, 2)[selected],
                                 np.array(result['response'])[selected], matrix[np.ix_(selected, selected)])
    assert expected['coefficients'][-1] < 1 and np.float32(expected['coefficients'][-1]) == 1
    expected['fitted'] = result['expected']['fitted']
    result['expected'] = expected
    return result


if __name__ == '__main__':
    if '--simulation' in sys.argv:
        print(json.dumps(simulate(), indent=2, allow_nan=False))
    elif '--ring' in sys.argv:
        print(json.dumps(ring_fixture(), indent=2, allow_nan=False))
    elif '--boundary' in sys.argv:
        print(json.dumps(boundary_fixture(), indent=2, allow_nan=False))
    else:
        print(json.dumps(fixture(20 if '--audit' in sys.argv else 60), indent=2, allow_nan=False))
