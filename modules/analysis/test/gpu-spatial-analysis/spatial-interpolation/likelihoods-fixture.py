# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Generate independent dense Gaussian prefix oracles with NumPy and SciPy.

Run /usr/bin/python3 likelihoods-fixture.py. Full predecessor sets make Vecchia equal
the dense multivariate normal. Scores use central differences of scipy logpdf;
expected information uses 1/2 trace(C^-1 C_j C^-1 C_k). Prefix differences give
per-site scores/information without implementing conditional derivative formulas.
"""

import json
from pathlib import Path

import numpy
from scipy.linalg import solve
from scipy.stats import multivariate_normal


def as_float32(values):
    return numpy.asarray(values, dtype=numpy.float32).astype(float)


positions = as_float32([[0, 0], [0.4, 0.2], [1.1, -0.3], [0.7, 1.3], [1.8, 0.6]])
times = as_float32([0, 0.3, 0.9, 1.5, 2.1])
values = as_float32([0.8, -0.4, 1.6, 0.2, -1.1])
errors = as_float32([0.1, 0.25, 0, 0.05, 0.15])
observations = numpy.column_stack([values, errors, numpy.zeros(5), numpy.ones(5)])
spatial_lags = numpy.linalg.norm(positions[:, None] - positions[None, :], axis=2)
time_lags = numpy.abs(times[:, None] - times[None, :])
same_place = spatial_lags == 0
same_time = time_lags == 0


def correlation(code, scaled):
    if code == 0:
        return numpy.where(scaled >= 1, 0, 1 - 1.5 * scaled + 0.5 * scaled**3)
    if code == 1:
        return numpy.exp(-3 * scaled)
    if code == 2:
        return numpy.exp(-3 * scaled**2)
    argument = (4.749031386012701 if code == 3 else 5.924462618951797) * scaled
    return (1 + argument + (argument**2 / 3 if code == 4 else 0)) * numpy.exp(-argument)


def structure(parameters, lag, coincident):
    code, nugget, sill, effective_range = parameters
    return sill * correlation(code, lag / effective_range) + nugget * coincident


def moments(covariance):
    matrix = covariance + numpy.diag(errors)
    means = []
    mean_term_magnitudes = []
    variances = []
    terms = []
    for count in range(5):
        cross = covariance[count, :count]
        weights = solve(matrix[:count, :count], cross, assume_a="pos") if count else []
        means.append(float(numpy.dot(weights, values[:count])))
        mean_term_magnitudes.append(float(numpy.abs(weights * values[:count]).sum()))
        variances.append(float(covariance[count, count] - numpy.dot(weights, cross)))
        previous = multivariate_normal.logpdf(values[:count], cov=matrix[:count, :count]) if count else 0
        current = multivariate_normal.logpdf(values[:count+1], cov=matrix[:count+1, :count+1])
        terms.append(float(current - previous))
    return dict(means=means, meanTermMagnitudes=mean_term_magnitudes, variances=variances, siteTerms=terms, logLikelihood=sum(terms))


fixtures = dict(positions=positions.ravel().tolist(), times=times.tolist(),
                observations=observations.ravel().tolist(),
                neighborhoodOffsets=[0, 0, 1, 3, 6, 10], neighborhoods=[0, 0, 1, 0, 1, 2, 0, 1, 2, 3],
                spatial=[], spaceTime=[])

for code in [1, 3]:
    nugget, sill, effective_range = as_float32([0.25, 1.25, 2.7])
    packed = as_float32([nugget + sill, 0, 0, 0, code, nugget, sill, effective_range])

    def covariance_at(theta):
        current_sill, current_range, current_nugget = numpy.exp(theta)
        return current_sill * correlation(code, spatial_lags / current_range) + current_nugget * same_place

    theta = numpy.log([sill, effective_range, nugget])
    covariance = covariance_at(theta)
    result = moments(covariance)
    all_cells = []
    previous_cells = numpy.zeros(10)
    step = 1e-5
    for count in range(1, 6):
        matrix = covariance[:count, :count] + numpy.diag(errors[:count])
        score = []
        derivatives = []
        for parameter in range(3):
            change = numpy.eye(3)[parameter] * step
            plus = covariance_at(theta + change)[:count, :count]
            minus = covariance_at(theta - change)[:count, :count]
            score.append((multivariate_normal.logpdf(values[:count], cov=plus + numpy.diag(errors[:count])) -
                          multivariate_normal.logpdf(values[:count], cov=minus + numpy.diag(errors[:count]))) / (2 * step))
            derivatives.append(solve(matrix, (plus - minus) / (2 * step), assume_a="pos"))
        information = [float(numpy.trace(derivatives[first] @ derivatives[second]) / 2)
                       for first in range(3) for second in range(first, 3)]
        cells = numpy.asarray([multivariate_normal.logpdf(values[:count], cov=matrix), *score, *information])
        all_cells.extend((cells - previous_cells).tolist())
        previous_cells = cells
    fixtures["spatial"].append(dict(modelCode=code, parameters=packed.tolist(), **result,
                                   fisherSiteTerms=all_cells, fisherTotals=previous_cells.tolist()))

for code in range(5):
    packed = numpy.zeros(28)
    packed[1:4] = [code, 0.7, 1.5 if code == 0 else 0.25]
    packed[4:8] = [1, 0.125, 0.875, 2.4]
    packed[8:12] = [3, 0.25, 0.75, 2.8]
    packed[12:16] = [2, 0.125, 1.25, 2.2]
    packed[16:24] = [1.25, 0.25, 0.4, 0.7, 0.5, 0.5, 0.4, 1]
    packed[24:28] = [10, 10, 1, 0]
    packed = as_float32(packed)
    spatial = structure(packed[4:8], spatial_lags, same_place)
    temporal = structure(packed[8:12], time_lags, same_time)
    joint = structure(packed[12:16], numpy.sqrt(spatial_lags**2 + packed[2]**2 * time_lags**2), same_place & same_time)
    if code == 0:
        covariance = packed[3] * spatial * temporal
    elif code == 1:
        covariance = packed[3] * spatial * temporal + spatial + temporal
    elif code == 2:
        covariance = joint
    elif code == 3:
        covariance = spatial + temporal + joint
    else:
        sill, nugget, temporal_scale, spatial_scale, alpha, gamma, beta, tau = packed[16:24]
        psi = 1 + temporal_scale * time_lags**(2 * alpha)
        covariance = sill * psi**(-tau) * numpy.exp(-spatial_scale * spatial_lags**(2 * gamma) * psi**(-beta * gamma)) + nugget * (same_place & same_time)
    packed[0] = as_float32([covariance[0, 0]])[0]
    fixtures["spaceTime"].append(dict(typeCode=code, parameters=packed.tolist(), **moments(covariance)))

Path(__file__).with_suffix('.json').write_text(json.dumps(fixtures, indent=2) + '\n')
