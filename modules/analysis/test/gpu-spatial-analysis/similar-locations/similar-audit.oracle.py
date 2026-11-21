# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/similar-locations/similar-audit.oracle.py
"""Independent SciPy zscore/rankdata, NumPy weighted norms and stable lexsort.
Population ddof=0; ties use average percentile rank; columns are standardized
over all valid/unmasked rows, including reference rows. Ranking ties prefer row ID.
"""
import json
import numpy as np
from scipy.stats import zscore, rankdata

values = np.array([[0,2,7],[2,0,7],[1,1,7],[1,1,7],[4,3,7],[99,99,7],[np.nan,2,7],[-2,-1,7]], dtype=float)
valid = np.isfinite(values).all(axis=1)
valid[5] = False
selected = np.array([True,True,False,False,False,False,False,False])
for mode in ['zscore', 'rank']:
    standardized = np.full_like(values, np.nan)
    if mode == 'zscore':
        standardized[valid, :2] = zscore(values[valid, :2], axis=0, ddof=0)
        standardized[valid, 2] = 0
    else:
        standardized[valid] = (rankdata(values[valid], method='average', axis=0)-1)/(sum(valid)-1)
    reference = standardized[selected & valid].mean(axis=0)
    distances = np.linalg.norm((standardized-reference)*np.sqrt([2,.5,0]), axis=1)
    for exclude in [True,False]:
        eligible = valid & (~selected if exclude else True)
        for least in [False,True]:
            rows = np.flatnonzero(eligible)
            # Symmetric reference deltas have exact mathematical ties; discard
            # ~1e-16 norm noise before lexsort so row ID resolves those ties.
            order = rows[np.lexsort((rows, (-1 if least else 1)*np.round(distances[rows], 14)))]
            ranks = np.full(8, 0xffffffff)
            ranks[order] = np.arange(len(order))
            shown = distances.copy()
            shown[~eligible] = np.nan
            top = np.full(8, 0xffffffff)
            top[:len(order)] = order
            print(mode, 'exclude', exclude, 'least', least, json.dumps(dict(distances=shown.tolist(),ranks=ranks.tolist(),topIds=top.tolist(),count=len(order))))
