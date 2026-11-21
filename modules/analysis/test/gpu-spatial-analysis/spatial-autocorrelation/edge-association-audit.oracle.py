# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-autocorrelation/edge-association-audit.oracle.py

The contract is a weighted edge covariance/correlation or mean difference, not
esda Gamma's uncentered sum and not esda Spatial_Pearson (Lee's L).
Expand integer edge weights into repeated observations, use numpy.cov and
numpy.corrcoef; no analytic WGSL reduction or spatial helper is reused.
Local Geary uses esda Geary_Local's population-standardized squared distance.
"""
import importlib.util
from pathlib import Path
import numpy as np

source = Path(__file__).with_name('local-autocorrelation-audit.oracle.py')
specification = importlib.util.spec_from_file_location('fixture', source)
fixture = importlib.util.module_from_spec(specification)
specification.loader.exec_module(fixture)
second_values = np.array([2, 5, 1, 7, 4, 3, 11, 999, 9, np.nan])

for second in [fixture.VALUES, second_values]:
    pairs = []
    for row, edges in enumerate(fixture.EDGES):
        for neighbor, weight in edges:
            if (row != neighbor and weight > 0 and fixture.MASK[row]
                    and fixture.MASK[neighbor] and np.isfinite(fixture.VALUES[row])
                    and np.isfinite(second[neighbor])):
                pairs += [(fixture.VALUES[row], second[neighbor])] * weight
    pairs = np.array(pairs)
    print('same variable' if second is fixture.VALUES else 'second variable')
    print('cross-product', np.cov(pairs.T, bias=True)[0, 1])
    print('pearson', np.corrcoef(pairs.T)[0, 1])
    print('squared-difference', np.mean(np.diff(pairs, axis=1)**2))
    print('absolute-difference', np.mean(np.abs(np.diff(pairs, axis=1))))

for keep in [False, True]:
    active = np.arange(7) if not keep else np.array([0, 1, 2, 3, 4, 5, 6, 8])
    standardized = (fixture.VALUES[active] - fixture.VALUES[active].mean()) / fixture.VALUES[active].std()
    lookup = dict(zip(active, standardized))
    for row_standardized in [False, True]:
        output = [None] * 10
        for row in active:
            terms = [(neighbor, weight) for neighbor, weight in fixture.EDGES[row]
                     if neighbor != row and weight > 0 and neighbor in lookup]
            total = sum(weight for _, weight in terms)
            output[row] = sum(weight*(lookup[row]-lookup[neighbor])**2 for neighbor, weight in terms)
            if row_standardized and total:
                output[row] /= total
        print('geary', 'keep' if keep else 'exclude', 'row' if row_standardized else 'raw', output)

# A perfect linear relationship at each of eight directed edges must have r=1,
# whatever the units of measurement. All individual squares fit in f32 at 2**32;
# multiplying two sums of squares overflows f32, producing r=0 in the old kernel.
large_x = np.arange(1, 9, dtype=float) * 2**32
large_y = np.roll(large_x, 1)  # neighbor of i is (i+1)%8, so y_neighbor == x_i
print('large-unit edge Pearson', np.corrcoef(large_x, np.roll(large_y, -1))[0, 1])
