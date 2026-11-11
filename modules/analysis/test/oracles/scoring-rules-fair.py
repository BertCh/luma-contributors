# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""NumPy and SciPy references for the FAIR ensemble scores of GPUEnsembleScores.

Math correctness review 2026-11-10, interpolation-density-12 (review follow-up). scoring-rules.py
evaluates the fair CRPS and the fair energy score with the same pairwise sum the kernel uses, so
the divisor 2 m (m - 1) had no check from another direction. This fixture checks it two ways that
do not share that sum:

1. Another formula for the same number, per ensemble.
   - Fair CRPS by order statistics (the probability-weighted-moment form, Zamo and Naveau 2018):
     mean |x_i - y| - sum_k (2 k - m - 1) x_(k) / (m (m - 1)), x_(1) <= ... <= x_(m), from
     numpy.sort. No pair loop.
   - Fair energy score as a U-statistic: mean ||x_i - y|| - (1/2) mean of the distances over
     the m (m - 1) / 2 distinct pairs, from scipy.spatial.distance.pdist.
2. The property that defines "fair" (Ferro 2014): the expectation over ensembles drawn from a
   distribution F equals the score of F itself, for any ensemble size. For standard normal
   members the scores of F are closed forms:
   - CRPS(N(0, 1), y) = y (2 Phi(y) - 1) + 2 phi(y) - 1 / sqrt(pi)        (scipy.stats.norm)
   - ES(N(0, I_d), y) = sqrt(2) G 1F1(-1/2; d/2; -|y|^2 / 2) - G, G = Gamma((d + 1) / 2) / Gamma(d / 2)
     (scipy.special.hyp1f1, gamma; E||X - X'|| = 2 G for independent X, X').
   The fixture records the mean over its ensembles and three standard errors of that mean. The
   standard (not fair) estimators are biased high by E||X - X'|| / (2 m), which is larger than
   that band here, so the check separates the two divisors (asserted below).

Run from the repository root with the pinned set of requirements.txt (NumPy and SciPy only):
    python modules/analysis/test/oracles/scoring-rules-fair.py
The first run (2026-11-10, NumPy 2.0.2 and SciPy 1.13.1 on the system Python 3.9.6) and the pinned
run (NumPy 2.5.4, SciPy 1.16.3) give the same values to the last digit.
Members and observations are rounded to float32 first (the GPU reads float32).
"""
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial.distance import pdist
from scipy.special import gamma, hyp1f1
from scipy.stats import norm

rng = np.random.default_rng(20261110)

# Fair CRPS: 512 sites, 5 members each, one observation value.
CRPS_SITES = 512
CRPS_MEMBERS = 5
crps_observation = float(np.float32(0.3))
crps_members = rng.standard_normal((CRPS_MEMBERS, CRPS_SITES)).astype(np.float32).astype(float)
ordered = np.sort(crps_members, axis=0)
ranks = np.arange(1, CRPS_MEMBERS + 1)[:, None]
pair_sum = np.sum((2 * ranks - CRPS_MEMBERS - 1) * ordered, axis=0)  # sum over i < j of |x_i - x_j|
absolute_error = np.mean(np.abs(crps_members - crps_observation), axis=0)
crps_fair = absolute_error - pair_sum / (CRPS_MEMBERS * (CRPS_MEMBERS - 1))
crps_standard = absolute_error - pair_sum / CRPS_MEMBERS**2
crps_closed = crps_observation * (2 * norm.cdf(crps_observation) - 1) + 2 * norm.pdf(crps_observation) - 1 / math.sqrt(math.pi)
crps_band = 3 * float(np.std(crps_fair, ddof=1)) / math.sqrt(CRPS_SITES)
assert abs(float(np.mean(crps_fair)) - crps_closed) < crps_band
assert abs(float(np.mean(crps_standard)) - crps_closed) > 2 * crps_band

# Fair energy score: 128 ensembles of 3 members in 3 dimensions, one observation vector.
ENERGY_ENSEMBLES = 128
ENERGY_MEMBERS = 3
ENERGY_DIMENSION = 3
energy_observation = np.array([0.5, -0.2, 0.1], dtype=np.float32).astype(float)
energy_members = rng.standard_normal((ENERGY_ENSEMBLES, ENERGY_MEMBERS, ENERGY_DIMENSION)).astype(np.float32).astype(float)
energy_fair = np.array([
    np.mean(np.linalg.norm(members - energy_observation, axis=1)) - 0.5 * np.mean(pdist(members))
    for members in energy_members
])
energy_standard = np.array([
    np.mean(np.linalg.norm(members - energy_observation, axis=1))
    - 0.5 * np.mean(pdist(members)) * (ENERGY_MEMBERS - 1) / ENERGY_MEMBERS
    for members in energy_members
])
gamma_ratio = gamma((ENERGY_DIMENSION + 1) / 2) / gamma(ENERGY_DIMENSION / 2)
energy_closed = float(
    math.sqrt(2) * gamma_ratio * hyp1f1(-0.5, ENERGY_DIMENSION / 2, -float(energy_observation @ energy_observation) / 2)
    - gamma_ratio
)
energy_band = 3 * float(np.std(energy_fair, ddof=1)) / math.sqrt(ENERGY_ENSEMBLES)
assert abs(float(np.mean(energy_fair)) - energy_closed) < energy_band
assert abs(float(np.mean(energy_standard)) - energy_closed) > 1.5 * energy_band

result = {
    'generator': 'scoring-rules-fair.py',
    'versions': {'numpy': np.__version__, 'scipy': scipy.__version__},
    'crps': {
        'memberCount': CRPS_MEMBERS,
        'siteCount': CRPS_SITES,
        'observation': crps_observation,
        # Member-major, as GPUEnsembleScores reads it: ensemble[member * siteCount + site].
        'ensemble': crps_members.ravel().tolist(),
        'fair': crps_fair.tolist(),
        'standard': crps_standard.tolist(),
        'closedForm': crps_closed,
        'meanBand': crps_band,
    },
    'energy': {
        'memberCount': ENERGY_MEMBERS,
        'dimension': ENERGY_DIMENSION,
        'observation': energy_observation.tolist(),
        # One row per ensemble, member-major inside: ensemble[member * dimension + component].
        'ensembles': [members.ravel().tolist() for members in energy_members],
        'fair': energy_fair.tolist(),
        'standard': energy_standard.tolist(),
        'closedForm': energy_closed,
        'meanBand': energy_band,
    },
}
output = Path(__file__).parent / 'fixtures/scoring-rules-fair.json'
output.write_text(json.dumps(result, allow_nan=False) + '\n')
print(json.dumps({
    'fixture': str(output), 'versions': result['versions'],
    'crps': [float(np.mean(crps_fair)), float(np.mean(crps_standard)), crps_closed, crps_band],
    'energy': [float(np.mean(energy_fair)), float(np.mean(energy_standard)), energy_closed, energy_band],
}))
