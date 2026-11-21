# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent dense NumPy OLS/sandwich oracle.

Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/robust-restriction.oracle.py
No implementation imports. Output is the literal fixture in robust-restriction-fixture.ts.
HC0: B X' diag(e**2) X B; HC1: n/(n-k) HC0; HAC: B X' diag(e) K diag(e) X B.
Chow F: ((SSE_r-SSE_u)/(k_u-k_r))/(SSE_u/(n-k_u)). Wald: d' (R V R')^-1 d.
"""
import json
import numpy as np
from scipy.stats import chi2

rows = 33
regimes = np.arange(rows) % 3
predictors = np.array([[(row * 7 % 13) - 6, (row * 5 % 11) - 5] for row in range(rows)], dtype=np.float32)
response = np.array([2 + regimes[row] * .75 + .5 * predictors[row, 0]
                     + (.25 + .125 * regimes[row]) * predictors[row, 1]
                     + [.5, 1.5, 4][regimes[row]] * ((row * 3 % 7) - 3) / 4
                     for row in range(rows)], dtype=np.float32)
mask = np.ones(rows, dtype=int)
mask[[4, 22]] = 0
regimes[32] = 99
used = (mask != 0) & (regimes < 3)
positions = np.arange(rows, dtype=float)
positions[31] = 1000  # An included island still contributes its unit-diagonal HC0 term.
kernel = np.maximum(0, 1 - np.abs(positions[:, None] - positions[None, :]) / 4)
offsets, neighbors, weights = [0], [], []
for row in range(rows):
    for neighbor in np.flatnonzero(kernel[row]):
        neighbors.append(int(neighbor))
        weights.append(float(kernel[row, neighbor]))
    offsets.append(len(neighbors))

def design(values, common):
    columns = []
    for column, shared in enumerate(common):
        value = np.ones(rows) if column == 0 else values[:, column - 1].astype(float)
        columns.extend([value] if shared else [value * (regimes == regime) for regime in range(3)])
    return np.array(columns).T[used]

def fit(values, common=(False, True, False)):
    restricted = design(values, common)
    unrestricted = design(values, [False, False, False])
    observed = response[used].astype(float)
    estimate = np.linalg.lstsq(restricted, observed, rcond=None)[0]
    full_estimate = np.linalg.lstsq(unrestricted, observed, rcond=None)[0]
    fitted = restricted @ estimate
    residual = observed - fitted
    squares = residual @ residual
    full_residual = observed - unrestricted @ full_estimate
    full_squares = full_residual @ full_residual
    count, size = restricted.shape
    # SVD bread avoids squaring the conditioning of the shifted copy.
    inverse = np.linalg.pinv(restricted)
    bread = inverse @ inverse.T
    covariance = bread * squares / (count - size)
    def tests(score_matrix, smoothing=None, scale=1):
        results = []
        offsets = np.cumsum([0] + [1 if shared else 3 for shared in common])
        for test in range(4):
            selected = [(offsets[column]+regime, offsets[column]) for column in range(3)
                        if not common[column] and (test == 3 or test == column) for regime in [1, 2]]
            if not selected:
                results.extend([None, 0, None])
                continue
            restriction = np.zeros((len(selected), size))
            for row, (positive, negative) in enumerate(selected):
                restriction[row, negative] = -1
                restriction[row, positive] = 1
            difference = restriction @ estimate
            projected = restriction @ score_matrix
            contrast_covariance = scale * (projected @ projected.T if smoothing is None else projected @ smoothing @ projected.T)
            statistic = float(difference @ np.linalg.solve(contrast_covariance, difference))
            results.extend([statistic, len(selected), float(chi2.sf(statistic, len(selected)))])
        return results
    fitted_all = np.full(rows, np.nan)
    fitted_all[used] = fitted
    result = dict(coefficients=estimate.tolist(), covariance=covariance.ravel().tolist(),
                  fitted=fitted_all.tolist(), chow=[float((squares-full_squares)/(9-size)/(full_squares/(count-9))), 9-size, count-9],
                  stability=tests(inverse, scale=squares/(count-size)), status=[0])
    scores = inverse * residual[None, :]
    for name, robust_covariance in [('white', scores @ scores.T),
                                   ('hac', scores @ kernel[np.ix_(used, used)] @ scores.T)]:
        # B X' = pinv(X). This is the stated sandwich formula without squaring
        # the conditioning of the large-mean common-intercept pivot fixture.
        if name == 'white':
            result['hc1'] = dict(covariance=(robust_covariance * count/(count-size)).ravel().tolist(),
                                 stability=tests(scores, scale=count/(count-size)))
            result['hc1']['chow'] = result['hc1']['stability'][-3:]
            result['hc1']['status'] = [0]
        result[name] = dict(covariance=robust_covariance.ravel().tolist(), stability=tests(scores, kernel[np.ix_(used, used)] if name == 'hac' else None), status=[0])
        result[name]['chow'] = result[name]['stability'][-3:]
    return result

spatial_coefficients, spatial_fitted, counts, moments = [], np.full(rows, np.nan), [], []
for regime in range(4):
    selected = used & (regimes == regime)
    count = int(selected.sum())
    counts.append(count)
    if not count:
        spatial_coefficients.extend([None]*3)
        moments.extend([0]*12)
        continue
    values = np.column_stack([predictors[selected], response[selected]]).astype(float)
    means = values.mean(axis=0).astype(np.float32).astype(float)
    deviations = values-means
    products = deviations.T @ deviations
    moments.extend([*means, *products[np.triu_indices(3)], *deviations.sum(axis=0)])
    matrix = np.column_stack([np.ones(count), predictors[selected]])
    coefficients = np.linalg.lstsq(matrix, response[selected], rcond=None)[0]
    spatial_coefficients.extend(coefficients.tolist())
    spatial_fitted[selected] = matrix @ coefficients

fixture = dict(predictors=predictors.ravel().tolist(), response=response.tolist(), regimes=regimes.tolist(), mask=mask.tolist(),
               offsets=offsets, neighbors=neighbors, weights=weights, expected=fit(predictors),
               commonIntercept=fit(predictors, [True, True, False]),
               pivot=fit(predictors + np.array([0, 1e6], dtype=np.float32), [True, True, False]),
               spatial=dict(coefficients=spatial_coefficients, fitted=spatial_fitted.tolist(), counts=counts, status=[0,0,0,1], moments=moments))
def finite(value):
    if isinstance(value, dict): return {key: finite(item) for key, item in value.items()}
    if isinstance(value, list): return [finite(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value
print('// SPDX-License-Identifier: MIT\n// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors')
print('// Literal independent expectations from robust-restriction.oracle.py.\nexport const ROBUST_RESTRICTION_FIXTURE = ' + json.dumps(finite(fixture), indent=2) + ';')
