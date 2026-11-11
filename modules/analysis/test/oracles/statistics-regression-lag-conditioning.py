# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Float64 references for the spatial lag estimators on inputs that stress float32 conditioning
(spreg, libpysal, NumPy), for the review fixes of the closing pass of the math correctness review
(2026-11-11, frc-regression-01, -02, -04 and -06).

Writes fixtures/statistics-regression-lag-conditioning.json next to this script, consumed by
test/gpu-spatial-analysis/spatial-regression/math-review-lag-conditioning.spec.ts
(`GPUSpatialLagMaximumLikelihood`, `GPUSpatialTwoStageLeastSquares`). Run from anywhere with
    <venv>/bin/python -I modules/analysis/test/oracles/statistics-regression-lag-conditioning.py
in the pinned environment of requirements.txt: CPython 3.14, NumPy 2.5.4, SciPy 1.16.3,
libpysal 4.15.0, spreg 1.9.1 (the versions are recorded in the fixture).

Scenes. 600 rows. Every input comes from a 32-bit linear congruential stream and IEEE double
arithmetic in a fixed order (no transcendental function, no linear solve), so the spec rebuilds
bit-identical float32 inputs from the scene parameters stored in the fixture; `responseSum` (the
float64 sum of the float32 responses) lets the spec check that.
- Predictors: `uniform` is uniform on [-2, 2) (stream 20261111); `grid` is 4096 plus a multiple
  of 2^-9 in [-2, 2) (stream 424242), a predictor whose mean is thousands of its deviations.
- Noise `e`: a sum of four uniforms scaled to unit variance (the stream of `noiseSeed`).
- Weights (CSR rows, neighbours ascending, float32 values):
  `ring` two neighbours at 0.5; `six` six neighbours at float32(1/6) (every row sums to
  1.0000000298, not 1); `binary` two neighbours at 1 (row sums 2, not row-standardized);
  `islands` the ring with rows 100, 250 and 477 without neighbours and the other rows
  row-standardized over the neighbours that remain; `mixed` rows `i - 1, i + 1` and, for every
  third row, `i + 2`, row-standardized in float32, so the exact row sums are 1 and 1.0000000298;
  `unequal` the neighbours of `mixed` at weight 1 (row sums 2 and 3, not row-standardized).
- Response: `y = (I - rho W)^-1 (constant + sum_j slope_j (x_j - offset_j) + deviation e)` by
  the power series with 400 terms in a fixed order, optionally rounded to a multiple of 2^-9,
  then to float32.

Scene families.
- `high-fit-*` (frc-regression-01): slopes 200 and 60 against residuals of 0.004, so the
  concentrated residual sum of squares is 1e-9 to 1e-7 of `e0'e0`; response centres 10,000
  and 0; eight noise streams each.
- `islands-*`: the lag process and an ordinary response on the island weights.
- `lag-*` (frc-regression-02): the six rows of the review's table (ring and six neighbours,
  `rho` 0.4 to 0.8, residual deviations 0.004 and 0.002 around 10,000, no rounding, so the
  float32 lag of the raw response is inexact), the mixed and the binary weights.
- `lag-unequal-*`: row sums of 2 and 3 with a response near 10,000, where the lag itself differs
  between rows by 10,000 and `rho` multiplies it: the limit float32 leaves (a float32 `rho` and
  a float32 lag column), measured and pinned, not removed.
- `two-predictors-*`: a uniform and a grid predictor on the six-neighbour weights.

References, all float64 on the float32 inputs.
- `lagMaximumLikelihood`: the maximum of the concentrated likelihood. `e0` and `e1` are the
  residuals of `y` and of the float64 lag `W y` on `[1, X]` (centered columns,
  `numpy.linalg.lstsq`), the log-determinant is `sum log|1 - rho lambda_i|` over
  `numpy.linalg.eigvals(W)`, and `rho` is a 1,801-point grid followed by a golden section. It is
  the estimator of `spreg.ML_Lag(method='full')`, whose own bounded search stops within 1e-7
  of this `rho`; on a high-fit scene that is a tenth of a standard error of `rho` and up to a
  percent of `sigma2`, so spreg's own numbers are stored under `spreg` and asserted here (`rho`
  within 5e-7, a log-likelihood not above this maximum and within 0.05 of it) instead of used
  as the pin. Standard errors are spreg's (`std_err`, with the intercept mapped back through
  `vm` where the fit was shifted).
- `twoStage`: `delta = (Z'PZ)^-1 Z'P y` with `Z = [1, X, W y]`, `H = [1, X, W X]`, by
  least squares on centered columns; `sigma2 = u'u / n`; standard errors from the QR factor of
  the projected regressors; the Anselin-Kelejian statistic by the formula of spreg `akTest`.
  On the scenes whose rows have equal sums, `spreg.GM_Lag(w_lags=1, spat_diag=True)` on the
  shifted response must agree (asserted: `rho` and the intercept within 1e-4 of their standard
  errors, `sigma2` 1e-8, the statistic 1e-6, standard errors 1e-7). spreg on an unshifted response near
  10,000 is itself 1e-4 off in `sigma2` (its raw normal equations), which is why the scenes with
  unequal row sums, where no shift is an intercept shift, use the centered float64 solve.
- Shift. Where every row of `W` has the same sum `s`, spreg runs on `y - c` and on
  `x - offset` (`c` is `responseShift`), which changes only the intercept: the stored intercept
  is `a' + c (1 - rho s) - sum_j offset_j b_j`. With islands or mixed row sums spreg runs on the
  raw inputs.

Edge cases (`edgeCases` of the fixture; the third round of review fixes, frc2-regression-01,
-02 and -07). Their inputs are stored in full: weights as CSR (`offsets`, `neighbors`,
`weights`), one `predictor`, the `response`.
- `zeroResidualAtOutsideVertex`: six rows on a ring; `e0 = y` and `e1 = y / 2` exactly, so
  `u'u = 12 (1 - rho / 2)^2`, zero only at `rho = 2`. Reference `lagMaximumLikelihood`:
  `spreg.ML_Lag(method='full', epsilon=1e-11)` (coefficients, `std_err`, `sig2`, `logll`).
- `zeroLagResidual`: eight rows in four reciprocal pairs of weight 1; `W y` is the predictor, so
  `e1 = 0`. The same spreg call; its `rho` and coefficients are within 1e-10 of zero (asserted)
  and are stored as exact zeros.
- `responseScale`: the second reviewer's input (NumPy `default_rng(20261111)`, 600 rows on the
  ring, `y = (I - 0.4 W)^-1 (1 + 2 x + e)` by `numpy.linalg.solve`). `baseResponse` is the
  float64 solve; each entry of `scales` is the fit of `float32(baseResponse * scale)`: the
  float64 maximum of the concentrated likelihood as above, held to spreg (`epsilon=1e-11`) at
  the scales 1e-10 and 1 (`rho` within 5e-9, `sig2` within 1e-8). At 1e10 spreg's power
  expansion of the fitted values, which has an absolute tolerance, does not converge.
  `standardErrors` are spreg's at scale 1, those of the intercept and the slope times the scale.
- `parameterBelowFloat32Spacing`: the second reviewer's input (`default_rng(73)`, 64 rows on the
  ring, an integer response and `x = float32(y - 0.4 W y)`). The pin is the minimizer of `u'u`
  in exact rational arithmetic on the float32 inputs (`fractions.Fraction`), which the
  likelihood maximum equals to 1e-14; spreg is 2.7e-9 away (it differences three float64 sums)
  and is stored under `spreg`, with its standard errors.
- `signedWeightsOfZeroTotal`: eight rows in four reciprocal pairs with weights +1, +1, -1, -1.
  `twoStage` is `spreg.GM_Lag(w_lags=1)` (coefficients, `std_err`, `sig2`); the Anselin-Kelejian
  statistic is spreg's `akTest` formula with the total weight cancelled,
  `(u'Wu / sig2n)^2 / (T + 4 a / sig2n)`, evaluated on spreg's `u`, `z` and `varb`, and asserted
  equal to spreg's own `ak_test` (1e-10) on the same pairs with all weights +1.
"""

import contextlib
import io
import json
import math
import os
import sys
import warnings
from fractions import Fraction

import libpysal
import numpy as np
import spreg

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE_PATH = os.path.join(HERE, 'fixtures', 'statistics-regression-lag-conditioning.json')
ROW_COUNT = 600
SERIES_TERM_COUNT = 400
ISLAND_ROWS = (100, 250, 477)
GRID_PREDICTOR_CENTER = 4096.0
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


def get_noise(seed):
    get_uniform = create_stream(seed)
    noise = []
    for _ in range(ROW_COUNT):
        total = get_uniform() + get_uniform() + get_uniform() + get_uniform()
        noise.append((total - 2.0) * 1.7320508075688772)
    return noise


def get_predictor(kind):
    if kind == 'uniform':
        get_uniform = create_stream(20261111)
        return [round_to_float32(4.0 * get_uniform() - 2.0) for _ in range(ROW_COUNT)]
    get_uniform = create_stream(424242)
    return [
        GRID_PREDICTOR_CENTER + math.floor(get_uniform() * 2048.0 - 1024.0) / 512.0
        for _ in range(ROW_COUNT)
    ]


def get_weight_rows(kind):
    """One (neighbours, float32 weights) pair per row, neighbours ascending."""
    rows = []
    for row in range(ROW_COUNT):
        if kind == 'six':
            neighbors = [(row + ROW_COUNT + step) % ROW_COUNT for step in (-3, -2, -1, 1, 2, 3)]
        elif kind in ('mixed', 'unequal'):
            neighbors = [(row + ROW_COUNT - 1) % ROW_COUNT, (row + 1) % ROW_COUNT]
            if row % 3 == 0:
                neighbors.append((row + 2) % ROW_COUNT)
        else:
            neighbors = [(row + ROW_COUNT - 1) % ROW_COUNT, (row + 1) % ROW_COUNT]
        if kind == 'islands':
            neighbors = [] if row in ISLAND_ROWS else [n for n in neighbors if n not in ISLAND_ROWS]
        neighbors = sorted(neighbors)
        if kind == 'ring':
            weight = 0.5
        elif kind in ('binary', 'unequal'):
            weight = 1.0
        else:
            weight = round_to_float32(1.0 / len(neighbors)) if neighbors else 0.0
        rows.append((neighbors, [weight] * len(neighbors)))
    return rows


class Weights:
    def __init__(self, kind):
        self.kind = kind
        self.rows = get_weight_rows(kind)
        width = max(len(neighbors) for neighbors, _ in self.rows)
        # Padded slots carry a zero weight: adding +0.0 leaves the running sum unchanged.
        self.index = np.zeros((ROW_COUNT, width), dtype=np.int64)
        self.value = np.zeros((ROW_COUNT, width))
        self.dense = np.zeros((ROW_COUNT, ROW_COUNT))
        for row, (neighbors, values) in enumerate(self.rows):
            for slot, (neighbor, value) in enumerate(zip(neighbors, values)):
                self.index[row, slot] = neighbor
                self.value[row, slot] = value
                self.dense[row, neighbor] = value
        self.row_sums = self.dense.sum(axis=1)
        self.has_equal_row_sums = bool(np.all(self.row_sums == self.row_sums[0]))
        self.norm = float(np.abs(self.dense).sum(axis=1).max())
        self.eigenvalues = np.linalg.eigvals(self.dense)
        self.library = libpysal.weights.W(
            {row: neighbors for row, (neighbors, _) in enumerate(self.rows)},
            {row: values for row, (_, values) in enumerate(self.rows)},
            silence_warnings=True,
        )

    def get_lag(self, values):
        """Row by row `((0 + w0 v0) + w1 v1) + ...` in slot order, as the spec does."""
        total = np.zeros(ROW_COUNT)
        for slot in range(self.index.shape[1]):
            total = total + self.value[:, slot] * values[self.index[:, slot]]
        return total


def create_response(scene, weights, predictors):
    reduced = np.full(ROW_COUNT, scene['constant'])
    for column, slope in enumerate(scene['slopes']):
        reduced = reduced + slope * (predictors[:, column] - scene['predictorOffsets'][column])
    reduced = reduced + scene['noiseDeviation'] * np.array(get_noise(scene['noiseSeed']))
    total = reduced.copy()
    term = reduced.copy()
    if scene['parameter'] != 0.0:
        for _ in range(1, SERIES_TERM_COUNT):
            term = scene['parameter'] * weights.get_lag(term)
            total = total + term
    if scene['roundToGrid']:
        total = np.floor(total * 512.0 + 0.5) / 512.0
    return total.astype(np.float32).astype(np.float64)


def minimize_by_golden_section(objective, low, high):
    for _ in range(200):
        left = high - GOLDEN_RATIO * (high - low)
        right = low + GOLDEN_RATIO * (high - low)
        if objective(left) <= objective(right):
            high = right
        else:
            low = left
    return (low + high) / 2


def solve_least_squares(design, response):
    return np.linalg.lstsq(design, response, rcond=None)[0]


def fit_lag_maximum_likelihood(predictors, response, weights):
    """Maximum of the concentrated lag likelihood, centered float64 regressions."""
    centered = predictors - predictors.mean(axis=0)
    lagged = weights.dense @ response
    centered_response = response - response.mean()
    centered_lagged = lagged - lagged.mean()
    response_slopes = solve_least_squares(centered, centered_response)
    lagged_slopes = solve_least_squares(centered, centered_lagged)
    response_residual = centered_response - centered @ response_slopes
    lagged_residual = centered_lagged - centered @ lagged_slopes

    def get_log_determinant(parameter):
        return float(np.sum(np.log(np.abs(1 - parameter * weights.eigenvalues))))

    def get_negative_likelihood(parameter):
        residual = response_residual - parameter * lagged_residual
        return ROW_COUNT / 2 * math.log(residual @ residual / ROW_COUNT) - get_log_determinant(parameter)

    bound = 0.9 / weights.norm
    grid = np.linspace(-bound, bound, 1801)
    best = int(np.argmin([get_negative_likelihood(parameter) for parameter in grid]))
    assert 0 < best < len(grid) - 1, 'the maximum must be interior'
    parameter = minimize_by_golden_section(get_negative_likelihood, grid[best - 1], grid[best + 1])
    residual = response_residual - parameter * lagged_residual
    slopes = response_slopes - parameter * lagged_slopes
    sigma_squared = float(residual @ residual) / ROW_COUNT
    return {
        'parameter': float(parameter),
        'slopes': [float(value) for value in slopes],
        'intercept': float(response.mean() - parameter * lagged.mean() - predictors.mean(axis=0) @ slopes),
        'sigmaSquared': sigma_squared,
        'logLikelihood': float(
            -ROW_COUNT / 2 * math.log(sigma_squared)
            + get_log_determinant(parameter)
            - ROW_COUNT / 2 * (math.log(2 * math.pi) + 1)
        ),
        # How far the concentrated sum of squares is below e0'e0: the cancellation the float32
        # form e0'e0 - 2 rho e0'e1 + rho^2 e1'e1 has to survive.
        'residualRatio': float((residual @ residual) / (response_residual @ response_residual)),
    }


def fit_two_stage(predictors, response, weights):
    """Spatial two-stage least squares with [1, X, W X] as instruments, centered float64."""
    lagged = weights.dense @ response
    lagged_predictors = weights.dense @ predictors
    regressors = np.column_stack([predictors, lagged])
    instruments = np.column_stack([predictors, lagged_predictors])
    regressor_means = regressors.mean(axis=0)
    centered_regressors = regressors - regressor_means
    centered_instruments = instruments - instruments.mean(axis=0)
    centered_response = response - response.mean()
    projected = centered_instruments @ solve_least_squares(centered_instruments, centered_regressors)
    delta = solve_least_squares(projected, centered_response)
    residual = centered_response - centered_regressors @ delta
    sigma_squared = float(residual @ residual) / ROW_COUNT
    # Full (uncentered) projected regressors, columns scaled to unit length before the QR.
    full_regressors = np.column_stack([np.ones(ROW_COUNT), regressors])
    full_instruments = np.column_stack([np.ones(ROW_COUNT), instruments])
    full_projected = full_instruments @ solve_least_squares(full_instruments, full_regressors)
    scale = np.linalg.norm(full_projected, axis=0)
    triangle = np.linalg.qr(full_projected / scale, mode='r')
    inverse_triangle = np.linalg.inv(triangle)
    inverse_normal = (inverse_triangle @ inverse_triangle.T) / np.outer(scale, scale)
    standard_errors = np.sqrt(sigma_squared * np.diag(inverse_normal))
    # Anselin-Kelejian, as spreg akTest: Moran's I of u with the estimation-variance correction.
    dense = weights.dense
    weights_total = dense.sum()
    trace = float((dense * dense).sum() + (dense * dense.T).sum())
    moran = (ROW_COUNT / weights_total) * float(residual @ dense @ residual) / float(residual @ residual)
    product = (dense.T @ residual) @ full_regressors
    estimation = float(product @ inverse_normal @ product)
    phi_squared = (trace + 4 * estimation / sigma_squared) / ((weights_total / ROW_COUNT) ** 2 * ROW_COUNT)
    return {
        'parameter': float(delta[-1]),
        'slopes': [float(value) for value in delta[:-1]],
        'intercept': float(response.mean() - regressor_means @ delta),
        'standardErrors': [float(value) for value in standard_errors],
        'sigmaSquared': sigma_squared,
        'anselinKelejian': float(ROW_COUNT * moran * moran / phi_squared),
    }


def get_shift_vector(scene, row_sum):
    """Gradient of the unshifted intercept in the coefficients of the shifted fit."""
    return np.array([1.0] + [-offset for offset in scene['predictorOffsets']] + [-scene['responseShift'] * row_sum])


def get_library_inputs(scene, weights, predictors, response):
    """The inputs spreg runs on: shifted where the shift is an intercept shift."""
    if weights.has_equal_row_sums:
        return predictors - np.array(scene['predictorOffsets']), (response - scene['responseShift']).reshape(-1, 1)
    return predictors, response.reshape(-1, 1)


def get_library_intercept(scene, weights, coefficients):
    if not weights.has_equal_row_sums:
        return float(coefficients[0])
    row_sum = float(weights.row_sums[0])
    return float(
        coefficients[0]
        + scene['responseShift'] * (1 - coefficients[-1] * row_sum)
        - np.array(scene['predictorOffsets']) @ coefficients[1:-1]
    )


def get_library_standard_errors(scene, weights, model):
    count = len(scene['slopes']) + 2
    errors = [float(value) for value in np.array(model.std_err).flatten()[:count]]
    if weights.has_equal_row_sums:
        gradient = get_shift_vector(scene, float(weights.row_sums[0]))
        covariance = np.array(model.vm)[:count, :count]
        errors[0] = float(math.sqrt(gradient @ covariance @ gradient))
    return errors


def get_scene_references(scene, weights):
    predictors = np.column_stack([get_predictor(kind) for kind in scene['predictors']])
    response = create_response(scene, weights, predictors)
    library_predictors, library_response = get_library_inputs(scene, weights, predictors, response)
    result = {'responseSum': float(np.sum(response))}

    lag = fit_lag_maximum_likelihood(predictors, response, weights)
    # spreg 1.9.1 prints the model name while fitting.
    with contextlib.redirect_stdout(io.StringIO()):
        lag_model = spreg.ML_Lag(library_response, library_predictors, weights.library, method='full')
        two_stage_model = spreg.GM_Lag(
            library_response, library_predictors, w=weights.library, w_lags=1, spat_diag=True
        )
    lag_coefficients = lag_model.betas.flatten()
    assert abs(lag_coefficients[-1] - lag['parameter']) < 5e-7, (scene['name'], lag_coefficients[-1], lag['parameter'])
    assert lag_model.logll <= lag['logLikelihood'] + 1e-6, (scene['name'], lag_model.logll, lag['logLikelihood'])
    # With row sums of 2 and 3 the standard error of rho is 5e-8, so spreg's search tolerance of
    # 1e-7 leaves it further below the maximum.
    gap_limit = 0.2 if scene['weights'] == 'unequal' else 0.05
    assert lag['logLikelihood'] - lag_model.logll < gap_limit, (scene['name'], lag_model.logll, lag['logLikelihood'])
    lag['standardErrors'] = get_library_standard_errors(scene, weights, lag_model)
    lag['spreg'] = {
        'parameter': float(lag_coefficients[-1]),
        'intercept': get_library_intercept(scene, weights, lag_coefficients),
        'sigmaSquared': float(lag_model.sig2),
        'logLikelihood': float(lag_model.logll),
    }
    result['lagMaximumLikelihood'] = lag

    two_stage = fit_two_stage(predictors, response, weights)
    two_stage_coefficients = two_stage_model.betas.flatten()
    library = {
        'parameter': float(two_stage_coefficients[-1]),
        'intercept': get_library_intercept(scene, weights, two_stage_coefficients),
        'sigmaSquared': float(two_stage_model.sig2),
        'anselinKelejian': float(two_stage_model.ak_test[0]),
        'standardErrors': get_library_standard_errors(scene, weights, two_stage_model),
    }
    if weights.has_equal_row_sums:
        errors = np.array(two_stage['standardErrors'])
        assert abs(library['parameter'] - two_stage['parameter']) < 1e-4 * errors[-1], (scene['name'], library, two_stage)
        assert abs(library['intercept'] - two_stage['intercept']) < 1e-4 * errors[0], (scene['name'], library, two_stage)
        assert abs(library['sigmaSquared'] / two_stage['sigmaSquared'] - 1) < 1e-8, (scene['name'], library, two_stage)
        assert abs(library['anselinKelejian'] / two_stage['anselinKelejian'] - 1) < 1e-6, (scene['name'], library, two_stage)
        assert np.max(np.abs(np.array(library['standardErrors']) / errors - 1)) < 1e-7, (scene['name'], library, two_stage)
    two_stage['spreg'] = library
    result['twoStage'] = two_stage
    return result


def get_scenes():
    scenes = []

    def add(name, weights, parameter, constant, slopes, deviation, seed, **options):
        scenes.append(
            {
                'name': name,
                'weights': weights,
                'parameter': parameter,
                'constant': constant,
                'slopes': slopes,
                'predictors': options.get('predictors', ['uniform']),
                'predictorOffsets': options.get('offsets', [0.0] * len(slopes)),
                'noiseDeviation': deviation,
                'noiseSeed': seed,
                'roundToGrid': options.get('round_to_grid', False),
                'responseShift': options.get('shift', 0.0),
            }
        )

    # frc-regression-01: u'u is 1e-9 to 1e-7 of e0'e0.
    for slope, center in [(200.0, 10000.0), (200.0, 0.0), (60.0, 10000.0)]:
        for seed in range(1001, 1009):
            add(
                f'high-fit-slope-{slope:g}-center-{center:g}-stream-{seed}',
                'ring', 0.4, 0.6 * center, [slope], 0.004, seed, round_to_grid=True, shift=center,
            )
    add('islands-lag', 'islands', 0.4, 6000.0, [2.0], 0.004, 77003)
    add('islands-ordinary', 'islands', 0.0, 10000.0, [2.0], 0.5, 77003)
    # frc-regression-02: an inexact float32 lag of the raw response.
    six_row_sum = 6 * round_to_float32(1.0 / 6.0)
    for weights, row_sum, parameter, deviation in [
        ('ring', 1.0, 0.4, 0.004),
        ('ring', 1.0, 0.8, 0.004),
        ('ring', 1.0, 0.8, 0.002),
        ('six', six_row_sum, 0.4, 0.004),
        ('six', six_row_sum, 0.7, 0.004),
        ('six', six_row_sum, 0.7, 0.002),
        ('binary', 2.0, 0.2, 0.004),
    ]:
        add(
            f'lag-{weights}-rho-{parameter:g}-deviation-{deviation:g}',
            weights, parameter, 10000.0 * (1 - parameter * row_sum), [2.0], deviation, 77003, shift=10000.0,
        )
    add('lag-mixed-rho-0.7-deviation-0.002', 'mixed', 0.7, 3000.0, [2.0], 0.002, 77003)
    add('lag-unequal-rho-0.15-deviation-0.004', 'unequal', 0.15, 6000.0, [2.0], 0.004, 77003)
    add(
        'two-predictors-six-rho-0.4', 'six', 0.4, 10000.0 * (1 - 0.4 * six_row_sum), [2.0, -1.5], 0.004, 77003,
        predictors=['uniform', 'grid'], offsets=[0.0, GRID_PREDICTOR_CENTER], shift=10000.0,
    )
    return scenes


def get_library_weights(rows):
    """libpysal weights of explicit `(neighbours, weights)` rows."""
    return libpysal.weights.W(
        {row: neighbors for row, (neighbors, _) in enumerate(rows)},
        {row: values for row, (_, values) in enumerate(rows)},
        silence_warnings=True,
    )


def get_ring_rows(row_count):
    return [
        (sorted([(row + row_count - 1) % row_count, (row + 1) % row_count]), [0.5, 0.5])
        for row in range(row_count)
    ]


def get_csr(rows):
    """The weights as the spec passes them to the kernel: offsets, neighbours, weights."""
    offsets = [0]
    for neighbors, _ in rows:
        offsets.append(offsets[-1] + len(neighbors))
    return {
        'offsets': offsets,
        'neighbors': [neighbor for neighbors, _ in rows for neighbor in neighbors],
        'weights': [value for _, values in rows for value in values],
    }


def fit_library_lag(response, predictor, rows):
    """`spreg.ML_Lag(method='full', epsilon=1e-11)` on one predictor."""
    with contextlib.redirect_stdout(io.StringIO()):
        model = spreg.ML_Lag(
            np.array(response, dtype=float).reshape(-1, 1),
            np.array(predictor, dtype=float).reshape(-1, 1),
            get_library_weights(rows),
            method='full',
            epsilon=1e-11,
        )
    coefficients = model.betas.flatten()
    return {
        'intercept': float(coefficients[0]) + 0.0,
        'slopes': [float(coefficients[1]) + 0.0],
        'parameter': float(coefficients[2]) + 0.0,
        'standardErrors': [float(value) for value in np.array(model.std_err).flatten()[:3]],
        'sigmaSquared': float(model.sig2),
        'logLikelihood': float(model.logll),
    }


def get_exact_vertex(response, predictor):
    """Minimizer and minimum of `(e0 - rho e1)'(e0 - rho e1)` on a ring of weights 0.5, in exact
    rational arithmetic on the float32 inputs."""
    count = len(response)
    exact_response = [Fraction(float(value)) for value in response]
    exact_predictor = [Fraction(float(value)) for value in predictor]
    lagged = [
        Fraction(1, 2) * (exact_response[(row + count - 1) % count] + exact_response[(row + 1) % count])
        for row in range(count)
    ]
    predictor_mean = sum(exact_predictor) / count
    predictor_squares = sum((value - predictor_mean) ** 2 for value in exact_predictor)

    def get_residual(values):
        mean = sum(values) / count
        slope = (
            sum((first - predictor_mean) * (second - mean) for first, second in zip(exact_predictor, values))
            / predictor_squares
        )
        return [(second - mean) - slope * (first - predictor_mean) for first, second in zip(exact_predictor, values)]

    response_residual = get_residual(exact_response)
    lagged_residual = get_residual(lagged)
    lagged_squares = sum(value * value for value in lagged_residual)
    cross = sum(first * second for first, second in zip(response_residual, lagged_residual))
    response_squares = sum(value * value for value in response_residual)
    return float(cross / lagged_squares), float(response_squares - cross * cross / lagged_squares)


def get_cancelled_anselin_kelejian(model, dense):
    """spreg `akTest` with the total weight `S0` cancelled:
    `AK = (u'Wu)^2 / (sig2^2 (T + 4 a / sig2))`, `T = tr(W'W + WW)`, `a = (u'WZ) varb (u'WZ)'`."""
    residual = np.asarray(model.u).flatten()
    product = (residual @ dense) @ np.asarray(model.z)
    estimation = float(product @ np.asarray(model.varb) @ product)
    trace = float((dense * dense).sum() + (dense * dense.T).sum())
    sigma_squared = float(model.sig2n)
    ratio = float(residual @ dense @ residual) / sigma_squared
    return ratio * ratio / (trace + 4 * estimation / sigma_squared)


def get_edge_cases():
    """Third round of review fixes (2026-11-11, frc2-regression-01, -02 and -07): inputs given
    in full, because they are a few rows or come from NumPy's generator and a linear solve."""
    cases = {}

    # frc2-regression-01, S = 0 with the minimizer outside the search interval: e0 = y,
    # e1 = y / 2, so u'u = 12 (1 - rho / 2)^2, zero at rho = 2 only.
    rows = get_ring_rows(6)
    response, predictor = [2.0, 1.0, -1.0, -2.0, -1.0, 1.0], [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    cases['zeroResidualAtOutsideVertex'] = {
        **get_csr(rows),
        'response': response,
        'predictor': predictor,
        'lagMaximumLikelihood': fit_library_lag(response, predictor, rows),
    }

    # frc2-regression-01, e1'e1 = 0: the lag of the response is the predictor, so u'u = e0'e0
    # for every rho and the likelihood is the log-determinant, largest at rho = 0.
    rows = [([(row + 4) % 8], [1.0]) for row in range(8)]
    response = [-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    predictor = [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, -1.0, 1.0]
    library = fit_library_lag(response, predictor, rows)
    # spreg's bounded search stops within 1e-11 of zero; the maximum is at zero exactly.
    assert abs(library['parameter']) < 1e-10 and abs(library['slopes'][0]) < 1e-10, library
    cases['zeroLagResidual'] = {
        **get_csr(rows),
        'response': response,
        'predictor': predictor,
        'lagMaximumLikelihood': {**library, 'parameter': 0.0, 'slopes': [0.0], 'intercept': 0.0},
    }

    # frc2-regression-02, the scale of the response: the reviewer's input (NumPy generator
    # 20261111, 600 rows on the ring), the float64 solve stored in full and scaled by 1e-10, 1
    # and 1e10 before the rounding to float32.
    generator = np.random.default_rng(20261111)
    predictor = generator.uniform(-2, 2, ROW_COUNT).astype(np.float32).astype(np.float64)
    noise = generator.normal(size=ROW_COUNT).astype(np.float32).astype(np.float64)
    ring = Weights('ring')
    base = np.linalg.solve(np.eye(ROW_COUNT) - 0.4 * ring.dense, 1 + 2 * predictor + noise)
    scales = []
    unit_library = None
    for scale in (1e-10, 1.0, 1e10):
        response = (base * scale).astype(np.float32).astype(np.float64)
        reference = fit_lag_maximum_likelihood(predictor.reshape(-1, 1), response, ring)
        if scale != 1e10:
            # spreg's power expansion of the fitted values has an absolute tolerance and does
            # not converge on a response of 1e10; the float64 maximum is held to it elsewhere.
            library = fit_library_lag(response, predictor, ring.rows)
            assert abs(library['parameter'] - reference['parameter']) < 5e-9, (scale, library, reference)
            assert abs(library['sigmaSquared'] / reference['sigmaSquared'] - 1) < 1e-8, (scale, library, reference)
            if scale == 1.0:
                unit_library = library
        scales.append({'scale': scale, 'responseSum': float(np.sum(response)), **reference})
    for entry in scales:
        # Standard errors of spreg at scale 1: those of the intercept and the slope scale with
        # the response, that of rho does not.
        entry['standardErrors'] = [
            unit_library['standardErrors'][0] * entry['scale'],
            unit_library['standardErrors'][1] * entry['scale'],
            unit_library['standardErrors'][2],
        ]
        assert abs(entry['parameter'] - scales[1]['parameter']) < 1e-7, entry
    cases['responseScale'] = {
        'predictor': [float(value) for value in predictor],
        'baseResponse': [float(value) for value in base],
        'scales': scales,
    }

    # frc2-regression-02, a fit whose standard error of rho (4.6e-9) is below the float32
    # spacing of rho (3e-8): the reviewer's input (NumPy generator 73, 64 rows on the ring).
    generator = np.random.default_rng(73)
    response = generator.integers(-10, 11, 64).astype(np.float64)
    response[32:] = -response[:32]
    rows = get_ring_rows(64)
    dense = np.zeros((64, 64))
    for row, (neighbors, values) in enumerate(rows):
        dense[row, neighbors] = values
    predictor = (response - 0.4 * (dense @ response)).astype(np.float32).astype(np.float64)
    library = fit_library_lag(response, predictor, rows)
    vertex, minimum = get_exact_vertex(response, predictor)
    # The likelihood maximum is within 1e-14 of the minimizer of u'u (the log-determinant moves
    # it by about S tr(W A) / (n e1'e1)), so the exact minimizer is the pin. spreg forms u'u as
    # e0'e0 - 2 rho e0'e1 + rho^2 e1'e1 in float64, 1e-15 of e0'e0 here, and lands 2.7e-9 away
    # (0.6 of its own standard error of rho): it is stored under `spreg` and held to 5e-9.
    assert abs(library['parameter'] - vertex) < 5e-9, (library, vertex)
    cases['parameterBelowFloat32Spacing'] = {
        **get_csr(rows),
        'response': [float(value) for value in response],
        'predictor': [float(value) for value in predictor],
        'lagMaximumLikelihood': {
            'parameter': vertex,
            'residualSumOfSquares': minimum,
            'nearestFloat32Parameter': round_to_float32(vertex),
            'standardErrors': library['standardErrors'],
            'spreg': library,
        },
    }

    # frc2-regression-07, signed weights whose total is zero: Moran's I is undefined, the
    # Anselin-Kelejian statistic is not (the total cancels from it).
    def get_pair_rows(signs):
        return [([row ^ 1], [signs[row // 2]]) for row in range(8)]

    predictor = [-2.0, -1.0, 0.0, 1.0, 2.0, 4.0, -3.0, 5.0]
    response = [1.0, 3.0, -1.0, 2.0, 7.0, 0.0, -2.0, 8.0]

    def fit_pairs(signs, with_diagnostics):
        rows = get_pair_rows(signs)
        dense = np.zeros((8, 8))
        for row, (neighbors, values) in enumerate(rows):
            dense[row, neighbors] = values
        with contextlib.redirect_stdout(io.StringIO()):
            model = spreg.GM_Lag(
                np.array(response).reshape(-1, 1),
                np.array(predictor).reshape(-1, 1),
                w=get_library_weights(rows),
                w_lags=1,
                spat_diag=with_diagnostics,
            )
        return rows, model, get_cancelled_anselin_kelejian(model, dense)

    # The cancelled form is spreg's own statistic where the total weight is not zero.
    _, positive_model, positive_statistic = fit_pairs([1.0, 1.0, 1.0, 1.0], True)
    assert abs(positive_statistic / float(positive_model.ak_test[0]) - 1) < 1e-10, positive_statistic
    rows, model, statistic = fit_pairs([1.0, 1.0, -1.0, -1.0], False)
    coefficients = model.betas.flatten()
    cases['signedWeightsOfZeroTotal'] = {
        **get_csr(rows),
        'response': response,
        'predictor': predictor,
        'twoStage': {
            'intercept': float(coefficients[0]),
            'slopes': [float(coefficients[1])],
            'parameter': float(coefficients[2]),
            'standardErrors': [float(value) for value in np.array(model.std_err).flatten()[:3]],
            'sigmaSquared': float(model.sig2),
            'anselinKelejian': statistic,
        },
    }
    return cases


def main():
    warnings.simplefilter('ignore')
    weights = {}
    scenes = []
    for scene in get_scenes():
        if scene['weights'] not in weights:
            weights[scene['weights']] = Weights(scene['weights'])
        scenes.append({**scene, **get_scene_references(scene, weights[scene['weights']])})
    fixture = {
        'generator': 'modules/analysis/test/oracles/statistics-regression-lag-conditioning.py',
        'versions': {
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'scipy': __import__('scipy').__version__,
            'libpysal': libpysal.__version__,
            'spreg': spreg.__version__,
        },
        'rowCount': ROW_COUNT,
        'seriesTermCount': SERIES_TERM_COUNT,
        'islandRows': list(ISLAND_ROWS),
        'gridPredictorCenter': GRID_PREDICTOR_CENTER,
        'scenes': scenes,
        'edgeCases': get_edge_cases(),
    }
    with open(FIXTURE_PATH, 'w') as file:
        json.dump(fixture, file, indent=1)
        file.write('\n')


if __name__ == '__main__':
    main()
