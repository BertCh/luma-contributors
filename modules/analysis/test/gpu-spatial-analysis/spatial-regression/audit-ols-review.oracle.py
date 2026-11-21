# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/audit-ols-review.oracle.py
Independent review expectations; no production formulas or TypeScript imports.
"""
import numpy as np
from scipy import linalg, stats

for value, freedom in [(10000,10000),(100000,100000),(200000000,200000000),
                       (200020000,200000000),(199980000,200000000),
                       (20000000000,20000000000)]:
    print('R7 chi-square',value,freedom,repr(stats.chi2.sf(value,freedom)))
print('R9 small t',repr(2*stats.t.sf(1e-6,65534)))
print('R3 singular values',linalg.svdvals([[1,0,1],[0,1,1],[0,0,0]]),linalg.svdvals([[1,1]]))
for scale in [1.,2.**32,2.**-40]:
    x=np.array([-1,-1,1,1],float)
    y=scale*np.array([-1,1,-2,2],float)
    design=np.column_stack([np.ones(4),x])
    residual=y-design@linalg.lstsq(design,y)[0]
    standardized=residual/scale
    squared=standardized**2
    projected=design@linalg.lstsq(design,squared)[0]
    explained=np.sum((projected-squared.mean())**2)
    print('R8',scale,'kurtosis',stats.kurtosis(standardized,fisher=False,bias=True),
          'JB',stats.jarque_bera(standardized).statistic,
          'KB',4*explained/np.sum((squared-squared.mean())**2),
          'BP',explained/(2*squared.mean()**2))
