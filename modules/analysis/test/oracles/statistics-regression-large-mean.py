# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Float64 references for regressions on a response whose mean is far above its residuals
(spreg, libpysal, NumPy), for the closing pass of the math correctness review (2026-11-11).

Writes fixtures/statistics-regression-large-mean.json next to this script, consumed by
test/gpu-spatial-analysis/spatial-regression/math-review-closing-residuals.spec.ts
(`GPUSpatialLagMaximumLikelihood`, `GPUSpatialTwoStageLeastSquares`, `GPUSpatialErrorGM`) and
test/gpu-spatial-analysis/spatial-regression/math-review-large-response-mean.spec.ts
(`GPUOrdinaryLeastSquares`). Run from anywhere with
    <venv>/bin/python -I modules/analysis/test/oracles/statistics-regression-large-mean.py
in the pinned environment of requirements.txt: CPython 3.14, NumPy 2.5.4, SciPy 1.16.3,
libpysal 4.15.0, spreg 1.9.1 (the versions are recorded in the fixture).

Scenes. 600 rows on a ring (two neighbours, weights 0.5). The inputs come from a 32-bit linear
congruential stream and IEEE double arithmetic only (no transcendental function), so the specs
rebuild bit-identical float32 inputs; `responseSum` (the float64 sum of the float32 responses)
lets a spec check that. `x` is uniform on [-2, 2), `e` is a sum of four uniforms scaled to unit
variance.
- `plain`: `y = 10000 + 2 x + 0.002 e`.
- `lag`: `y = (I - 0.4 W)^-1 (6000 + 2 x + 0.004 e)` rounded to multiples of 2^-9, so that the
  float32 ring lag `0.5 a + 0.5 b` is exact near 10,000.
- `lag-unrounded`: the same without that rounding (the float32 lag is then inexact).
- `error`: `y = 10000 + 2 x + (I - 0.5 W)^-1 0.004 e`.

References, all float64 on the float32 inputs:
- `ordinaryLeastSquares` (`plain`): `spreg.OLS(y, x, nonspat_diag=True)`. Koenker-Bassett and
  Breusch-Pagan are NOT spreg's (spreg regresses the squared residuals on the squared
  regressors): they follow the definition of `GPUOrdinaryLeastSquares`, with `numpy.linalg.lstsq`:
  Koenker-Bassett is `n R^2` of `e^2` on `[1, x]`, Breusch-Pagan half the explained sum of
  squares of `e^2 / (e'e / n) - 1` on `[1, x]`.
- Spatial estimators: spreg on `y - 10000`. On a row-standardized `W` the shift changes only the
  intercept (by `10000 (1 - rho)` for a lag model, by 10000 for an error model) and it removes
  the conditioning error of spreg's raw normal equations (the unshifted `GM_Lag` residuals have
  a mean of 2e-5). `shiftedIntercept` is the intercept of that fit; `standardErrors[0]` is the
  standard error of the intercept of the UNSHIFTED model, from the covariance of the shifted fit
  (`var(a) = var(a') - 2 c cov(a', rho) + c^2 var(rho)`, `c = 10000`, for the lag models).
  - `lag-maximum-likelihood`: `spreg.ML_Lag(method='full')`. A golden-section maximum of the
    concentrated likelihood with the ring eigenvalues must agree in `rho` to 1e-7 (asserted).
  - `two-stage`: `spreg.GM_Lag(w_lags=1, spat_diag=True)`, with its Anselin-Kelejian test.
  - `homoskedastic`: `spreg.GM_Error_Hom(A1='hom_sc', max_iter=1)`.
  - `kelejian-prucha`: the exact minimum of spreg's moment objective (`_momentsGM_Error`, grid
    and golden section, `sigma2 >= 0` eliminated by least squares) followed by spreg's filtered
    regression (`get_spFilter`). spreg's own L-BFGS-B search (`GM_Error`) stops within 2e-4 of
    that minimum and runs to -0.99 on the shifted `plain` scene, so it is not the reference; its
    `lambda` is stored as `spregOptimizerLambda` for the record. `lambda` has no standard error
    (null).
- `unroundedLag`: `sig2` of `ML_Lag` and `GM_Lag` on the `lag-unrounded` scene.
"""

import json
import math
import os
import sys
import warnings

import libpysal
import numpy as np
import spreg
from spreg import utils as spreg_utils
from spreg.error_sp import _momentsGM_Error

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE_PATH = os.path.join(HERE, 'fixtures', 'statistics-regression-large-mean.json')
ROW_COUNT = 600
RESPONSE_CENTER = 10000.0
GOLDEN_RATIO = (math.sqrt(5) - 1) / 2


def create_stream(seed):
    """Uniform stream in [0, 1): a 32-bit linear congruential state, top 24 bits."""
    state = [seed & 0xFFFFFFFF]

    def get_uniform():
        state[0] = (state[0] * 1664525 + 1013904223) & 0xFFFFFFFF
        return (state[0] >> 8) / 16777216.0

    return get_uniform


def round_to_float32(value):
    return float(np.float32(value))


def get_ring_lag(values):
    return [
        0.5 * values[(row + ROW_COUNT - 1) % ROW_COUNT] + 0.5 * values[(row + 1) % ROW_COUNT]
        for row in range(ROW_COUNT)
    ]


def get_inverse_series(values, parameter):
    """sum_k parameter^k W^k values, 80 terms in a fixed order."""
    total = list(values)
    term = list(values)
    for _ in range(1, 80):
        term = [parameter * value for value in get_ring_lag(term)]
        total = [total[row] + term[row] for row in range(ROW_COUNT)]
    return total


def create_scene(kind):
    uniform_predictor = create_stream(20261111)
    uniform_noise = create_stream(77003)
    predictor = [round_to_float32(4.0 * uniform_predictor() - 2.0) for _ in range(ROW_COUNT)]
    noise = []
    for _ in range(ROW_COUNT):
        total = uniform_noise() + uniform_noise() + uniform_noise() + uniform_noise()
        noise.append((total - 2.0) * 1.7320508075688772)
    if kind == 'plain':
        response = [
            round_to_float32(RESPONSE_CENTER + 2.0 * predictor[row] + 0.002 * noise[row])
            for row in range(ROW_COUNT)
        ]
    elif kind == 'error':
        error = get_inverse_series([0.004 * value for value in noise], 0.5)
        response = [
            round_to_float32(RESPONSE_CENTER + 2.0 * predictor[row] + error[row])
            for row in range(ROW_COUNT)
        ]
    else:
        reduced = [
            0.6 * RESPONSE_CENTER + 2.0 * predictor[row] + 0.004 * noise[row]
            for row in range(ROW_COUNT)
        ]
        series = get_inverse_series(reduced, 0.4)
        if kind == 'lag':
            response = [round_to_float32(math.floor(value * 512.0 + 0.5) / 512.0) for value in series]
        else:
            response = [round_to_float32(value) for value in series]
    return (
        np.array(predictor, dtype=np.float64).reshape(-1, 1),
        np.array(response, dtype=np.float64).reshape(-1, 1),
    )


def get_weights():
    neighbors = {row: [(row + ROW_COUNT - 1) % ROW_COUNT, (row + 1) % ROW_COUNT] for row in range(ROW_COUNT)}
    weights = {row: [0.5, 0.5] for row in range(ROW_COUNT)}
    return libpysal.weights.W(neighbors, weights, silence_warnings=True)


def minimize_by_golden_section(objective, low, high):
    for _ in range(200):
        left = high - GOLDEN_RATIO * (high - low)
        right = low + GOLDEN_RATIO * (high - low)
        if objective(left) <= objective(right):
            high = right
        else:
            low = left
    return (low + high) / 2


def get_concentrated_lag_parameter(predictor, shifted_response):
    """rho maximising the concentrated lag likelihood, with the eigenvalues of the ring."""
    design = np.column_stack([np.ones(ROW_COUNT), predictor[:, 0]])
    response = shifted_response[:, 0]
    lagged = np.array(get_ring_lag(list(response)))
    projector = np.linalg.pinv(design)
    response_residual = response - design @ (projector @ response)
    lagged_residual = lagged - design @ (projector @ lagged)
    eigenvalues = np.cos(2 * np.pi * np.arange(ROW_COUNT) / ROW_COUNT)

    def get_negative_likelihood(parameter):
        residual = response_residual - parameter * lagged_residual
        return ROW_COUNT / 2 * np.log(residual @ residual / ROW_COUNT) - np.sum(
            np.log(1 - parameter * eigenvalues)
        )

    return minimize_by_golden_section(get_negative_likelihood, -0.95, 0.95)


def get_unshifted_intercept_error(covariance):
    """Standard error of a' + c (1 - rho) from the covariance of (a', b, rho)."""
    last = covariance.shape[0] - 1
    variance = (
        covariance[0, 0]
        - 2 * RESPONSE_CENTER * covariance[0, last]
        + RESPONSE_CENTER**2 * covariance[last, last]
    )
    return float(math.sqrt(variance))


def fit_lag_maximum_likelihood(predictor, shifted_response, weights):
    model = spreg.ML_Lag(shifted_response, predictor, weights, method='full')
    coefficients = model.betas.flatten()
    golden_section_parameter = get_concentrated_lag_parameter(predictor, shifted_response)
    assert abs(golden_section_parameter - coefficients[2]) < 1e-7, (golden_section_parameter, coefficients[2])
    return {
        'shiftedIntercept': float(coefficients[0]),
        'slope': float(coefficients[1]),
        'parameter': float(coefficients[2]),
        'standardErrors': [
            get_unshifted_intercept_error(model.vm),
            float(model.std_err[1]),
            float(model.std_err[2]),
        ],
        'sigmaSquared': float(model.sig2),
        'residualSumOfSquares': float(model.utu),
        'logLikelihood': float(model.logll),
    }


def fit_two_stage(predictor, shifted_response, weights):
    model = spreg.GM_Lag(shifted_response, predictor, w=weights, w_lags=1, spat_diag=True)
    coefficients = model.betas.flatten()
    return {
        'shiftedIntercept': float(coefficients[0]),
        'slope': float(coefficients[1]),
        'parameter': float(coefficients[2]),
        'standardErrors': [
            get_unshifted_intercept_error(model.vm),
            float(model.std_err[1]),
            float(model.std_err[2]),
        ],
        'sigmaSquared': float(model.sig2),
        'residualSumOfSquares': float(model.utu),
        'anselinKelejian': float(model.ak_test[0]),
    }


def fit_kelejian_prucha(predictor, shifted_response, weights):
    first_stage = spreg.OLS(shifted_response, predictor)
    moments = _momentsGM_Error(weights, first_stage.u)
    moment_matrix = np.array(moments[0], dtype=float)
    moment_vector = np.array(moments[1], dtype=float).flatten()

    def get_objective(parameter):
        # Least squares in sigma2 >= 0 for a fixed lambda.
        remainder = moment_vector - moment_matrix[:, 0] * parameter - moment_matrix[:, 1] * parameter**2
        variance_column = moment_matrix[:, 2]
        variance = max(0.0, float(remainder @ variance_column) / float(variance_column @ variance_column))
        error = remainder - variance_column * variance
        return float(error @ error)

    grid = np.linspace(-0.99, 0.99, 3961)
    best = int(np.argmin([get_objective(parameter) for parameter in grid]))
    parameter = minimize_by_golden_section(
        get_objective, grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    )
    design = np.hstack([np.ones((ROW_COUNT, 1)), predictor])
    filtered_design = spreg_utils.get_spFilter(weights, parameter, design)
    filtered_response = spreg_utils.get_spFilter(weights, parameter, shifted_response)
    coefficients = np.linalg.lstsq(filtered_design, filtered_response, rcond=None)[0]
    filtered_residual = filtered_response - filtered_design @ coefficients
    residual = shifted_response - design @ coefficients
    sigma_squared = float((filtered_residual.T @ filtered_residual)[0, 0]) / ROW_COUNT
    covariance = sigma_squared * np.linalg.inv(filtered_design.T @ filtered_design)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        optimizer_model = spreg.GM_Error(shifted_response, predictor, weights)
    return {
        'shiftedIntercept': float(coefficients[0, 0]),
        'slope': float(coefficients[1, 0]),
        'parameter': float(parameter),
        'standardErrors': [float(math.sqrt(covariance[0, 0])), float(math.sqrt(covariance[1, 1])), None],
        'sigmaSquared': sigma_squared,
        'residualSumOfSquares': float((residual.T @ residual)[0, 0]),
        'spregOptimizerLambda': float(optimizer_model.betas.flatten()[2]),
    }


def fit_homoskedastic(predictor, shifted_response, weights):
    model = spreg.GM_Error_Hom(shifted_response, predictor, weights, A1='hom_sc', max_iter=1)
    coefficients = model.betas.flatten()
    return {
        'shiftedIntercept': float(coefficients[0]),
        'slope': float(coefficients[1]),
        'parameter': float(coefficients[2]),
        'standardErrors': [float(value) for value in model.std_err],
        'sigmaSquared': float(model.sig2),
        'residualSumOfSquares': float((model.u.T @ model.u)[0, 0]),
    }


def fit_ordinary_least_squares(predictor, response):
    model = spreg.OLS(response, predictor, nonspat_diag=True)
    design = np.column_stack([np.ones(ROW_COUNT), predictor[:, 0]])
    shifted = response[:, 0] - RESPONSE_CENTER
    residual = shifted - design @ np.linalg.lstsq(design, shifted, rcond=None)[0]
    squared = residual**2
    squared_residual = squared - design @ np.linalg.lstsq(design, squared, rcond=None)[0]
    auxiliary_r_squared = 1 - (squared_residual @ squared_residual) / ((squared - squared.mean()) ** 2).sum()
    scaled = squared / (residual @ residual / ROW_COUNT) - 1
    scaled_fit = design @ np.linalg.lstsq(design, scaled, rcond=None)[0]
    return {
        'coefficients': model.betas.flatten().tolist(),
        'standardErrors': [float(value) for value in model.std_err],
        'residualSumOfSquares': float(model.utu),
        'sigmaSquared': float(model.sig2),
        'rSquared': float(model.r2),
        'logLikelihood': float(model.logll),
        'jarqueBera': float(model.jarque_bera['jb']),
        'koenkerBassett': float(ROW_COUNT * auxiliary_r_squared),
        'breuschPagan': float(((scaled_fit - scaled_fit.mean()) ** 2).sum() / 2),
    }


def main():
    warnings.simplefilter('ignore')
    weights = get_weights()
    scenes = {kind: create_scene(kind) for kind in ['plain', 'lag', 'lag-unrounded', 'error']}
    references = []
    for estimator, fit, kinds in [
        ('lag-maximum-likelihood', fit_lag_maximum_likelihood, ['plain', 'lag']),
        ('two-stage', fit_two_stage, ['plain', 'lag']),
        ('kelejian-prucha', fit_kelejian_prucha, ['plain', 'error']),
        ('homoskedastic', fit_homoskedastic, ['plain', 'error']),
    ]:
        for kind in kinds:
            predictor, response = scenes[kind]
            references.append(
                {
                    'estimator': estimator,
                    'scene': kind,
                    'responseSum': float(np.sum(response)),
                    **fit(predictor, response - RESPONSE_CENTER, weights),
                }
            )
    predictor, response = scenes['plain']
    ordinary_least_squares = {
        'responseSum': float(np.sum(response)),
        **fit_ordinary_least_squares(predictor, response),
    }
    predictor, response = scenes['lag-unrounded']
    unrounded_lag = {
        'responseSum': float(np.sum(response)),
        'lagMaximumLikelihoodSigmaSquared': fit_lag_maximum_likelihood(
            predictor, response - RESPONSE_CENTER, weights
        )['sigmaSquared'],
        'twoStageSigmaSquared': fit_two_stage(predictor, response - RESPONSE_CENTER, weights)['sigmaSquared'],
    }
    fixture = {
        'generator': 'modules/analysis/test/oracles/statistics-regression-large-mean.py',
        'versions': {
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'scipy': __import__('scipy').__version__,
            'libpysal': libpysal.__version__,
            'spreg': spreg.__version__,
        },
        'rowCount': ROW_COUNT,
        'responseCenter': RESPONSE_CENTER,
        'ordinaryLeastSquares': ordinary_least_squares,
        'references': references,
        'unroundedLag': unrounded_lag,
    }
    with open(FIXTURE_PATH, 'w') as file:
        json.dump(fixture, file, indent=1)
        file.write('\n')


if __name__ == '__main__':
    main()
