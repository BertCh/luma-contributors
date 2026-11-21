# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/audit-ols-review2.oracle.py
Exact rational auxiliary moments and independent SciPy tails/SVD for review 2.
"""
from fractions import Fraction
import numpy as np
from scipy import linalg, stats

difference = Fraction(1, 2**23)
first = Fraction(11, 8) + 3*difference
predictors = [1, 1, -2, -2, 1, 1]
response = [-first, first, -(first+difference), first+difference,
            -(first+2*difference), first+2*difference]
assert sum(response) == 0
assert sum(value*predictor for value, predictor in zip(response,predictors)) == 0
print('R10 exact coefficients',[0,0], 'response float32 step',float(np.spacing(np.float32(first))))
levels = [first**2, (first+difference)**2, (first+2*difference)**2]
mean = sum(levels)/3
cross = 64*2*(levels[0]-2*levels[1]+levels[2])
total = 64*2*sum((value-mean)**2 for value in levels)
koenker = Fraction(384)*cross**2/(768*total)
print('R10 KB,p',repr(float(koenker)),repr(float(stats.chi2.sf(float(koenker),1))))
scale = first+2*difference
print('R10 centered normalized levels',[float(value/scale**2-1) for value in levels])
print('R11 BP,p',2.88,repr(float(stats.chi2.sf(2.88,1))))
print('R12 singular values',linalg.svdvals([[1,1e-180],[0,1e-180]]))
for statistic,freedom in [(12,1e20),(2,1e15),(12,1e10),(35,1e10),(0.1,1e10)]:
    print('R14 Student t',statistic,freedom,repr(float(2*stats.t.sf(statistic,freedom))))
