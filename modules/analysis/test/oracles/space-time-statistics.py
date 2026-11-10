# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference lagged and differential local Moran (esda) and per-location cross-correlation (numpy,
scipy) for STX-02.

Writes fixtures/space-time-statistics.json next to this script, consumed by
test/gpu-spatial-analysis/space-time-statistics/space-time-statistics.spec.ts. Run with
    <venv>/bin/python -I modules/analysis/test/oracles/space-time-statistics.py
in a venv with numpy, scipy, esda and libpysal (versions are recorded in the fixture).

Scene: 40 uniform points in [0, 100]^2 (RandomState 21), 5 nearest neighbors (ties by index),
row-standardized; a cube of 7 periods of float32 values (a smooth field that drifts plus noise).
The fixture stores the cube period-major (one list per period) and the neighbor lists.

Local Moran references (GeoDa definitions, computed by esda on the float32-rounded cube):
- lagged inward : Moran_Local_BV(x = y_t, y = y_{t-lag}), transformation 'r'  (GeoDa inward influence)
- lagged outward: Moran_Local_BV(x = y_{t-lag}, y = y_t)
- differential  : Moran_Local(y_t - y_{t-lag}), transformation 'r'  (GeoDa: the Moran statistic of
  the change variable)
with 9999 permutations, seed 12345, alternative 'two-sided'. esda's `Is` for the bivariate form is
(n - 1)/n times the GeoDa z-product (esda divides by sum z_x^2 = n); the GPU twin uses esda's scaling.

Cross-correlation reference: for each cell and lag l, the Pearson r of the overlapping pairs
(first[t], second[t + l]) from numpy.corrcoef; the exact t-test p-value from scipy.stats.pearsonr;
the Fisher z p-value 2 norm.sf(|atanh r| sqrt(n - 3)) the GPU contributor uses; and
scipy.stats.false_discovery_control(method='bh') over all (cell, lag) Fisher p-values.

Tolerances (as the spec uses them): statistics and correlations are f32 sums of a few terms,
1e-4 absolute + 1e-4 relative; Fisher p-values 1e-5 absolute + 1e-3 relative; permutation p-values
are Monte Carlo estimates from different generators and are bounded by 5 binomial standard errors,
5 sqrt(2 * 0.25 / permutations).
"""

import json
import math
import os
import platform
import warnings

import esda
import libpysal
import numpy as np
import scipy
from esda import Moran_Local, Moran_Local_BV
from libpysal.weights import W
from scipy import stats

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(HERE, 'fixtures', 'space-time-statistics.json')

rng = np.random.RandomState(21)
n = 40
k = 5
periods = 7
points = rng.rand(n, 2) * 100.0
field = np.sin(points[:, 0] / 20.0) + np.cos(points[:, 1] / 25.0)
cube = np.zeros((periods, n))
level = field.copy()
for t in range(periods):
    level = 0.8 * level + 0.3 * np.sin(points[:, 0] / 15.0 + t) + 0.1 * t
    cube[t] = level + 0.4 * rng.randn(n)
cube = cube.astype(np.float32).astype(np.float64)
# A second variable that follows the first by one period, plus noise, for the cross-correlation.
second = np.zeros((periods, n))
for t in range(periods):
    base = cube[t - 1] if t > 0 else cube[0]
    second[t] = 0.7 * base + 0.5 * rng.randn(n)
second = second.astype(np.float32).astype(np.float64)

neighbors = []
for i in range(n):
    distance = np.hypot(*(points - points[i]).T)
    order = [j for j in np.lexsort((np.arange(n), distance)) if j != i][:k]
    neighbors.append(sorted(int(j) for j in order))


def make_w():
    return W({i: list(v) for i, v in enumerate(neighbors)}, {i: [1.0 / k] * k for i in range(n)}, id_order=list(range(n)))


SEED = 12345
PERMUTATIONS = 9999
period = 4
result = {'n': n, 'k': k, 'periods': periods, 'period': period, 'cube': cube.tolist(), 'second': second.tolist(),
          'neighbors': neighbors, 'permutations': PERMUTATIONS, 'moran': {}}

for lag in (1, 2):
    now = cube[period]
    before = cube[period - lag]
    inward = Moran_Local_BV(now, before, make_w(), transformation='r', permutations=PERMUTATIONS, seed=SEED, alternative='two-sided')
    outward = Moran_Local_BV(before, now, make_w(), transformation='r', permutations=PERMUTATIONS, seed=SEED, alternative='two-sided')
    differential = Moran_Local(now - before, make_w(), transformation='r', permutations=PERMUTATIONS, seed=SEED, alternative='two-sided')
    result['moran'][str(lag)] = {
        'laggedInward': {'I': inward.Is.tolist(), 'pSim': [float(v) for v in inward.p_sim], 'quadrants': [int(v) for v in inward.q]},
        'laggedOutward': {'I': outward.Is.tolist(), 'pSim': [float(v) for v in outward.p_sim], 'quadrants': [int(v) for v in outward.q]},
        'differential': {'I': differential.Is.tolist(), 'pSim': [float(v) for v in differential.p_sim], 'quadrants': [int(v) for v in differential.q]},
    }

# --- Cross-correlation per cell over lags ---
lags = [-2, -1, 0, 1, 2, 3]
correlations, fisher, exact, pairs = [], [], [], []
for cell in range(n):
    for lag in lags:
        start = max(0, -lag)
        stop = min(periods, periods - lag)
        a = cube[start:stop, cell]
        b = second[start + lag:stop + lag, cell]
        count = stop - start
        r = float(np.corrcoef(a, b)[0, 1])
        z = abs(math.atanh(r)) * math.sqrt(count - 3)
        correlations.append(r)
        fisher.append(float(2 * stats.norm.sf(z)))
        exact.append(float(stats.pearsonr(a, b).pvalue))
        pairs.append(count)
adjusted = stats.false_discovery_control(np.array(fisher), method='bh')
result['crossCorrelation'] = {
    'lags': lags,
    'correlations': correlations,
    'fisherPValues': fisher,
    'exactPValues': exact,
    'pairCounts': pairs,
    'adjustedFisherBH': [float(v) for v in adjusted],
}


def encode(value):
    """JSON-safe copy: NaN -> null, infinities -> strings (decoded by test/oracles/oracle-fixture.ts)."""
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [encode(entry) for entry in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return None
        if math.isinf(value):
            return 'Infinity' if value > 0 else '-Infinity'
        return value
    return value


fixture = {
    'generator': 'test/oracles/space-time-statistics.py',
    'versions': {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'esda': esda.__version__,
        'libpysal': libpysal.__version__,
    },
    'seed': SEED,
    'tolerances': {
        'statistic': {'absolute': 1e-4, 'relative': 1e-4},
        'fisherPValue': {'absolute': 1e-5, 'relative': 1e-3},
        'permutationPValue': '5 binomial standard errors: 5 sqrt(2 * 0.25 / permutations)',
    },
    'reference': result,
}
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as handle:
    json.dump(encode(fixture), handle, separators=(',', ':'))
    handle.write('\n')
print('wrote', out_path)
