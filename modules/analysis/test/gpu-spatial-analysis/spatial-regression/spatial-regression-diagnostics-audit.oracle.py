# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Independent dense annihilator definitions with SciPy SVD/p-values.
Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/spatial-regression-diagnostics-audit.oracle.py
Prints fixtures to stdout; no production code is read.
"""

import json

import numpy
from scipy import linalg, stats


def create_fixture(masked, restandardize, scaled):
    row_count = 16
    row = numpy.arange(row_count, dtype=float)
    predictors = numpy.column_stack((2 + numpy.sin(row * 1.2) + row / 10,
                                  -1 + numpy.cos(row * 0.7))).astype(numpy.float32).astype(float)
    predictors[8] = predictors[1]  # repeated predictor row
    response = (3 + predictors @ [0.7, -0.4] + 0.5 * numpy.sin(row * 1.8)
                + 0.3 * numpy.cos(row * 0.4)).astype(numpy.float32).astype(float)
    if scaled:
        response *= 2
        predictors += [16, -8]  # same design space with an implicit intercept
        predictors = predictors.astype(numpy.float32).astype(float)  # exact uploaded inputs
    mask = (row % 8 != 3).astype(int) if masked else numpy.ones(row_count, dtype=int)
    used = mask != 0
    dense = numpy.zeros((row_count, row_count))
    offsets = [0]
    neighbors = []
    weights = []
    for source in range(row_count):
        entries = [] if source in [0, 9] else sorted([((source + 1) % row_count, 0.2 + 0.01 * (source % 4)),
                          ((source + 4) % row_count, 0.5), ((source - 1) % row_count, 0.3)])
        for neighbor, weight in entries:
            neighbors.append(neighbor)
            weights.append(float(numpy.float32(weight)))
            dense[source, neighbor] = numpy.float32(weight)
        offsets.append(len(neighbors))
    restricted = dense[numpy.ix_(used, used)]
    if restandardize:
        sums = restricted.sum(axis=1)
        restricted = restricted / numpy.where(sums > 0, sums, 1)[:, None]
    design = numpy.column_stack((numpy.ones(used.sum()), predictors[used]))
    inverse_design = linalg.pinv(design)
    coefficients = inverse_design @ response[used]
    residuals = numpy.zeros(row_count)
    residuals[used] = (response[used] - design @ coefficients).astype(numpy.float32).astype(float)
    residual = residuals[used]
    annihilator = numpy.eye(used.sum()) - design @ inverse_design
    sigma_squared = residual @ residual / used.sum()
    trace = numpy.trace(restricted.T @ restricted + restricted @ restricted)
    weights_sum = restricted.sum()
    residual_lag = residual @ restricted @ residual
    response_lag = residual @ restricted @ response[used]
    fitted_lag = restricted @ (response[used] - residual)
    information = fitted_lag @ annihilator @ fitted_lag / sigma_squared + trace
    error_gradient = residual_lag / sigma_squared
    lag_gradient = response_lag / sigma_squared
    lag_statistic = lag_gradient ** 2 / information
    error_statistic = error_gradient ** 2 / trace
    robust_error = (error_gradient - trace * lag_gradient / information) ** 2 / (trace - trace ** 2 / information)
    robust_lag = (lag_gradient - error_gradient) ** 2 / (information - trace)
    scale = used.sum() / weights_sum
    degrees = used.sum() - design.shape[1]
    product = annihilator @ restricted
    moran_i = scale * residual_lag / (residual @ residual)
    moran_expectation = scale * numpy.trace(product) / degrees
    moran_variance = scale ** 2 * (numpy.trace(product @ annihilator @ restricted.T)
                                  + numpy.trace(product @ product) + numpy.trace(product) ** 2) / (degrees * (degrees + 2)) - moran_expectation ** 2
    moran_z = (moran_i - moran_expectation) / numpy.sqrt(moran_variance)
    moran_p = 2 * stats.norm.sf(abs(moran_z))
    tests = []
    for statistic, freedom in [(lag_statistic, 1), (error_statistic, 1), (robust_lag, 1),
                               (robust_error, 1), (lag_statistic + robust_error, 2)]:
        tests.extend([statistic, freedom, stats.chi2.sf(statistic, freedom)])
    tests.extend([moran_z, 0, moran_p])
    summary = [used.sum(), sigma_squared, trace, information, weights_sum, moran_i, moran_expectation,
               moran_variance, moran_z, moran_p, residual_lag, response_lag, row_count - used.sum()]
    return dict(name=f'masked-{masked}-restandardized-{restandardize}-scaled-{scaled}', rowCount=row_count, predictorCount=2,
                restandardizeRows=restandardize, weights=dict(offsets=offsets, neighbors=neighbors, weights=weights),
                predictors=predictors.ravel().tolist(), response=response.tolist(), residuals=residuals.tolist(), mask=mask.tolist(),
                expected=dict(tests=[float(value) for value in tests], summary=[float(value) for value in summary]))


print(json.dumps([create_fixture(True, restandardize, scaled)
                  for restandardize in [False, True] for scaled in [False, True]],
                 indent=2, allow_nan=False))
