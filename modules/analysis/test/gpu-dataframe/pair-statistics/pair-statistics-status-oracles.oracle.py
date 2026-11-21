# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-dataframe/pair-statistics/pair-statistics-status-oracles.oracle.py
"""Independent NumPy enumeration of the R4 complete five-point fixture."""
import json
import numpy as np

positions = np.array([[.25, .25], [.75, .25], [.25, .75], [.75, .75], [.5, .5]])
for amplitude in [1., 2. ** -47]:
    values = np.array([0, 0, 0, 0, amplitude], dtype=np.float32).astype(float)
    centered = values - values.mean()
    semivariance_terms = []
    products = []
    for first in range(5):
        for second in range(first + 1, 5):
            if np.linalg.norm(positions[first] - positions[second]) <= 2:
                semivariance_terms.append((values[first] - values[second]) ** 2 / 2)
                products.extend([centered[first] * centered[second]] * 2)
    variance = np.var(values)
    unit = variance / 4096
    with np.errstate(over='ignore'):
        old_scale = np.float32(2. ** 24) / (np.sqrt(np.float32(unit)) * np.sqrt(np.float32(unit)))
    print(json.dumps(dict(amplitude=amplitude, unorderedPairs=len(semivariance_terms),
        semivariance=float(np.mean(semivariance_terms)),
        moran=float(5 / len(products) * np.sum(products) / np.sum(centered ** 2)),
        unit=unit, maximumTermInUnits=max(semivariance_terms) / unit,
        oldSpaceTimeScale='Infinity' if np.isinf(old_scale) else float(old_scale))))

values = 2. ** -66 * np.array([0., 1., 2., 3., 524288.])
variance = np.var(values)
spread = (np.quantile(values, .75) - np.quantile(values, .25)) / 1.3489795
unit = np.clip(spread ** 2, variance * 2. ** -32, variance)
gamma = np.mean([(first-second)**2/2 for index, first in enumerate(values) for second in values[index+1:]])
print(json.dumps(dict(fixture='review2-subnormal-unit', variance=variance, unit=unit,
    flushedUnit=0 if unit < 2.**-126 else unit, semivariance=gamma, pairs=10, moran=-.25)))

# Enumerate every location of the one nonzero mark on the five-node path.
weights = np.diag(np.ones(4), 1) + np.diag(np.ones(4), -1)
permutation_values = []
for endpoint in range(5):
    values = np.eye(5)[endpoint]
    centered = values - values.mean()
    permutation_values.append(5 / weights.sum() * (centered @ weights @ centered) / (centered @ centered))
permutation_variance = np.var(permutation_values)
print(json.dumps(dict(fixture='review2-randomization', morans=permutation_values,
    variance=permutation_variance, z=(permutation_values[0] - np.mean(permutation_values)) / np.sqrt(permutation_variance))))
centered = (np.array([1.,0,0,0,0]) - .2).astype(np.float32) * np.float32(2.**-47)
second = np.sum(centered * centered, dtype=np.float32)
fourth = np.sum((centered * centered)**2, dtype=np.float32)
print(json.dumps(dict(oldSecondSquared=float(np.float32(second*second)), oldFourth=float(fourth), oldKurtosis='0/0')))
