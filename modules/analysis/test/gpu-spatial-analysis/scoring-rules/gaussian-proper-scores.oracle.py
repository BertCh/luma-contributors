# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 <this file>.

Independent SciPy normal density, survival probability and central quantiles.
Interval scoring uses explicit endpoints and strict outside-interval penalties,
independent of the implementation's centered expression. y>threshold is strict.
Inputs include repeated/tied values, point forecasts and invalid moments; no
mask/group/island applies to these independent per-row predictive distributions.
"""
import json
import numpy as np
from scipy.special import erf, erfc
from scipy.stats import norm

MEANS = np.array([0, 0, 2, -1, 3, 0, 2, 1, 0, 0, 1, np.nan], dtype=np.float32)
VARIANCES = np.array([1, 4, .25, 9, 1, 0, 0, 0, -1, np.nan, .01, 1], dtype=np.float32)
OBSERVATIONS = np.array([0, 3, 3, -5, 1, 0, 0, 1, 2, 0, 1, 2], dtype=np.float32)
ALPHA = float(np.float32(.1))


def scores(means, variances, observations, threshold):
    output = {name: [] for name in ['crps', 'logScore', 'brierScore', 'intervalScore']}
    for mean, variance, observed in zip(means.astype(float), variances.astype(float), observations.astype(float)):
        if not np.isfinite([mean, variance, observed]).all() or variance < 0:
            for values in output.values():
                values.append(None)
            continue
        if variance == 0:
            output['crps'].append(abs(observed - mean))
            output['logScore'].append(None)  # Dirac forecast has no Gaussian density.
            probability = float(mean > threshold)
            lower = upper = mean
        else:
            deviation = np.sqrt(variance)
            standardized = (observed - mean) / deviation
            output['crps'].append(float(deviation * (standardized * (2 * norm.cdf(standardized) - 1)
                + 2 * norm.pdf(standardized) - 1 / np.sqrt(np.pi))))
            output['logScore'].append(float(-norm.logpdf(observed, loc=mean, scale=deviation)))
            probability = norm.sf(threshold, loc=mean, scale=deviation)
            lower, upper = norm.ppf([ALPHA / 2, 1 - ALPHA / 2], loc=mean, scale=deviation)
        output['brierScore'].append(float((probability - (observed > threshold)) ** 2))
        interval = upper - lower
        if observed < lower:
            interval += 2 / ALPHA * (lower - observed)
        if observed > upper:
            interval += 2 / ALPHA * (observed - upper)
        output['intervalScore'].append(float(interval))
    return output


quantile_alphas = [2 ** -125, 1e-30, 1e-12, 1e-4, .01, .1, .5, .999]
erf_arguments = [0, .1, .5, 1, 2, 4, 5, 7]
result = {
    'base': scores(MEANS, VARIANCES, OBSERVATIONS, 1),
    'scaled': scores(2 * MEANS + 16, 4 * VARIANCES, 2 * OBSERVATIONS + 16, 18),
    'quantiles': [[float(np.float32(alpha)), float(-norm.ppf(float(np.float32(alpha)) / 2))] for alpha in quantile_alphas],
    'erf': [[argument, float(erf(argument))] for argument in erf_arguments]
}
tail_means = np.array([1e30, -1e30, 0, 0, 1e30, -1e30, 2, -2], dtype=np.float32).astype(float)
tail_variances = np.array([1e-20, 1e-20, 1e-20, 1e-20, 1, 1, 1, 1], dtype=np.float32).astype(float)
tail_observations = np.array([1, 1, 0, 1, -1, -1, 0, 0], dtype=float)
result['tailBrier'] = {
    'means': tail_means.tolist(), 'variances': tail_variances.tolist(), 'observations': tail_observations.tolist(),
    'expected': ((norm.sf(0, loc=tail_means, scale=np.sqrt(tail_variances)) - (tail_observations > 0)) ** 2).tolist()
}
result['representableTailBrier'] = [[magnitude, float(norm.sf(magnitude) ** 2)] for magnitude in [6, 8, 9]]
rounded_argument = float(np.float32(np.float32(8) * np.float32(1 / np.sqrt(2))))
result['brierFloat32Rounding'] = {
    'argument': rounded_argument,
    'squaredTail': float((.5 * erfc(rounded_argument)) ** 2),
    # A float32 exponent near -33 has ULP 2^-18; squaring exp doubles its effect.
    'oneExponentUlpRelativeEffect': float(np.expm1(2 * 2**-18))
}
print(json.dumps(result, indent=2, allow_nan=False))
