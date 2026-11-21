# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-autocorrelation/quantile-lisa-audit.oracle.py

NumPy Hazen/type-5 percentiles; exhaustive conditional combinations, with exact
GeoDa strict-complement tail. Eleven rows: an eight-ring of constant observations,
one island, one masked row and one nonfinite row. All labels in the final quantile
are 1, so every conditional sample of two donors has statistic 2. For P=9,
g=P, min(g,P-g)=0, p=(0+1)/(9+1)=.1; alpha=.2 gives significant=1.
First tied quantile is empty; zero observed join counts are not tested.
"""
import itertools
import numpy as np

values = np.array([5]*10 + [np.nan])
mask = np.array([1]*9 + [0, 1], dtype=bool)
valid = np.isfinite(values) & mask
print('sourceCount', len(values), 'validCount', int(valid.sum()))
print('interior tied example', np.quantile([1, 1, 2, 3, 4, 5, 6, 7, 8], [1/3, 2/3], method='hazen').tolist())
for selected in [1, 4]:
    bounds = np.quantile(values[valid], [(selected-1)/4, selected/4], method='hazen')
    bounds[0] = -np.inf if selected == 1 else bounds[0]
    bounds[1] = np.inf if selected == 4 else bounds[1]
    labels = (valid & (values >= bounds[0]) & (values < bounds[1])).astype(int)
    print('boundaries', bounds.tolist(), 'membership', labels.tolist())
    print('analysisCount', 8, 'islandCount', 1)
    if selected == 4:
        for focus in range(8):
            donors = [row for row in range(8) if row != focus]
            observed = 2
            draws = list(itertools.combinations(donors, 2))
            assert all(sum(labels[list(draw)]) == observed for draw in draws)
        print('ring observed=2, exceedances=0, p=.1, significant/classes=1; other rows untested')
    else:
        assert not labels.any()
        print('ring observed=0, untested, classes=0; island class=3, excluded classes=2')
