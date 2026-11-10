# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference proper scoring rules for GPUEnsembleScores and GPUGaussianPredictiveScores.

Writes fixtures/scoring-rules.json next to this script, consumed by
test/gpu-spatial-analysis/scoring-rules/gpu-ensemble-scores.spec.ts (GPU) and
scoring-rules.node.spec.ts (fixture sanity). Run from the repository root with
    <venv>/bin/python -I modules/analysis/test/oracles/scoring-rules.py
in a venv with numpy (version recorded).

EXTERNAL FIXTURE PENDING: scoringRules 1.1.3 (CRAN, GPL >= 2) is an R package and R is not
installed where this fixture was made. Every value here is a binary64 evaluation of the
scoringRules definitions, not a scoringRules output. When R is available, a sibling script should
call crps_sample(method = "edf"), es_sample, vs_sample(p = 0.5) and crps_norm on the same inputs
and compare with this file.

Definitions (members x_1 .. x_m, observation y; vectors for the energy and variogram scores):
- crps_sample(method = "edf"):  mean |x_i - y| - 1/(2 m^2) sum_ij |x_i - x_j|. Cross-checked below
  against the integral of (F_m(x) - 1{y <= x})^2 over the real line (exact, piecewise constant).
- es_sample:                    mean ||x_i - y|| - 1/(2 m^2) sum_ij ||x_i - x_j||.
- vs_sample(p):                 sum_ij w_ij (|y_i - y_j|^p - mean_k |x_ki - x_kj|^p)^2, full matrix.
- crps_norm:                    sd * (z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi)), z = (y - mean) / sd,
  with math.erf; cross-checked against numerical integration of (Phi - 1{y <= x})^2.
- fair variants (NOT in scoringRules 1.1.3; Ferro 2014): the pair term uses 1/(2 m (m - 1)).
Inputs are rounded to float32 first (the GPU reads float32).
"""

import json
import math
import os
import platform

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'scoring-rules.json')


def f32(values):
    return np.asarray(values, dtype=np.float32).astype(np.float64)


def crps_sample(members, y, fair=False):
    m = len(members)
    pair = sum(abs(a - b) for a in members for b in members)
    denominator = 2 * m * (m - 1 if fair else m)
    return sum(abs(x - y) for x in members) / m - pair / denominator


def crps_edf_integral(members, y):
    """Exact integral of (F_m - 1{y <= x})^2 dx for the empirical CDF."""
    points = sorted(set(list(members) + [y]))
    total = 0.0
    m = len(members)
    for left, right in zip(points, points[1:]):
        midpoint = 0.5 * (left + right)
        cdf = sum(1 for x in members if x <= midpoint) / m
        indicator = 1.0 if y <= midpoint else 0.0
        total += (cdf - indicator) ** 2 * (right - left)
    return total


def energy_score(members, y, fair=False):
    m = len(members)
    norm = lambda v: math.sqrt(sum(c * c for c in v))
    observed = sum(norm(x - y) for x in members) / m
    pair = sum(norm(a - b) for a in members for b in members)
    return observed - pair / (2 * m * (m - 1 if fair else m))


def variogram_score(members, y, pairs, order, weights=None):
    total = 0.0
    for index, (i, j) in enumerate(pairs):
        forecast = sum(abs(x[i] - x[j]) ** order for x in members) / len(members)
        weight = 1.0 if weights is None else weights[index]
        total += weight * (abs(y[i] - y[j]) ** order - forecast) ** 2
    return total


def crps_normal(mean, variance, y):
    if variance == 0:
        return abs(y - mean)
    sd = math.sqrt(variance)
    z = (y - mean) / sd
    phi = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return sd * (z * math.erf(z / math.sqrt(2)) + 2 * phi - 1 / math.sqrt(math.pi))


def crps_normal_integral(mean, sd, y):
    lower, upper, steps = mean - 12 * sd, mean + 12 * sd, 200000
    total = 0.0
    width = (upper - lower) / steps
    for k in range(steps):
        x = lower + (k + 0.5) * width
        cdf = 0.5 * (1 + math.erf((x - mean) / (sd * math.sqrt(2))))
        total += (cdf - (1.0 if y <= x else 0.0)) ** 2 * width
    return total


def ensemble_case(rng, name, members, dimension, spread, offset):
    ensemble = f32(rng.normal(0, spread, size=(members, dimension)))
    observation = f32(rng.normal(offset, spread, size=dimension))
    all_pairs = [[i, j] for i in range(dimension) for j in range(dimension) if i != j]
    neighbours = [[i, i + 1] for i in range(dimension - 1)]
    weights = [float(f32([0.5 + 0.25 * (k % 3)])[0]) for k in range(len(neighbours))]
    crps = [crps_sample(list(ensemble[:, s]), observation[s]) for s in range(dimension)]
    for s in range(dimension):
        integral = crps_edf_integral(list(ensemble[:, s]), observation[s])
        assert abs(integral - crps[s]) <= 1e-9 * max(1.0, abs(crps[s])), (integral, crps[s])
    return {
        'name': name,
        'memberCount': members,
        'dimension': dimension,
        'ensemble': ensemble.reshape(-1).tolist(),
        'observations': observation.tolist(),
        'crps': crps,
        'crpsFair': [crps_sample(list(ensemble[:, s]), observation[s], True) for s in range(dimension)],
        'energyScore': energy_score(list(ensemble), observation),
        'energyScoreFair': energy_score(list(ensemble), observation, True),
        'variogramOrder': 0.5,
        'variogramPairsAll': all_pairs,
        'variogramScoreAll': variogram_score(list(ensemble), observation, all_pairs, 0.5),
        'variogramPairsNeighbours': neighbours,
        'variogramWeights': weights,
        'variogramScoreNeighbours': variogram_score(list(ensemble), observation, neighbours, 0.5, weights),
        'variogramScoreNeighboursOrderOne': variogram_score(list(ensemble), observation, neighbours, 1.0, weights),
    }


def main():
    rng = np.random.default_rng(20261109)
    cases = [
        ensemble_case(rng, 'small', 4, 3, 1.0, 0.3),
        ensemble_case(rng, 'ten-members', 10, 6, 2.0, 0.0),
        ensemble_case(rng, 'offset-observation', 25, 5, 1.0, 4.0),
        ensemble_case(rng, 'many-members', 100, 8, 3.0, -1.0),
        # More members than lanes of the per-site workgroup (64): strided member loop.
        ensemble_case(rng, 'members-over-lanes', 150, 4, 1.5, 0.5),
    ]
    # Hand-checkable: members 0, 1, 2 and y = 0. mean|x - y| = 1; pair sum = 2*(1+2+1) = 8,
    # so CRPS = 1 - 8 / 18 = 5/9; the fair variant is 1 - 8 / 12 = 1/3.
    hand = {
        'members': [0.0, 1.0, 2.0],
        'observation': 0.0,
        'crps': 5 / 9,
        'crpsFair': 1 / 3,
    }
    assert abs(crps_sample([0.0, 1.0, 2.0], 0.0) - hand['crps']) < 1e-15
    assert abs(crps_sample([0.0, 1.0, 2.0], 0.0, True) - hand['crpsFair']) < 1e-15

    gaussian = []
    for mean, variance, y in [
        (0.0, 1.0, 0.0), (0.0, 1.0, 1.0), (0.0, 1.0, -2.5), (2.0, 0.25, 3.0), (10.0, 4.0, 4.0),
        (-3.0, 100.0, 7.5), (1.0, 0.0, 3.5), (0.0, 1e-4, 0.05), (5.0, 9.0, 5.0), (0.0, 1.0, 7.0),
        (0.0, 1.0, 0.3), (0.0, 1.0, 5.0),
    ]:
        value = crps_normal(mean, variance, y)
        if variance > 0:
            integral = crps_normal_integral(mean, math.sqrt(variance), y)
            assert abs(integral - value) <= 1e-3 * max(1.0, abs(value)), (mean, variance, y, integral, value)
        gaussian.append({'mean': mean, 'variance': variance, 'observation': y, 'crps': value})
    # crps_norm(0, mean = 0, sd = 1) = (sqrt(2) - 1) / sqrt(pi), a closed form to check by hand.
    assert abs(gaussian[0]['crps'] - (math.sqrt(2) - 1) / math.sqrt(math.pi)) < 1e-15

    fixture = {
        'generator': 'test/oracles/scoring-rules.py',
        'versions': {'python': platform.python_version(), 'numpy': np.__version__},
        'externalOracle': 'scoringRules 1.1.3 (R) pending: not installed here; values are binary64 evaluations of its documented definitions, cross-checked by numerical integration',
        'tolerances': {
            'crpsAbsolute': 2e-5,
            'crpsRelative': 2e-5,
            'energyRelative': 2e-5,
            'variogramRelative': 2e-5,
            'gaussianAbsolute': 2e-6,
            'gaussianRelative': 2e-6,
        },
        'hand': hand,
        'ensembles': cases,
        'gaussian': gaussian,
    }
    with open(OUTPUT, 'w') as file:
        json.dump(fixture, file, indent=1)
        file.write('\n')
    print(f'wrote {OUTPUT}: {len(cases)} ensemble cases and {len(gaussian)} Gaussian cases')


if __name__ == '__main__':
    main()
