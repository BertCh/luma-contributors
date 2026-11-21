# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import math, runpy
import numpy as np
reference = runpy.run_path('modules/analysis/test/gpu-spatial-analysis/permutation-inference/permutation-mathematical-audit.oracle.py')

def order(permutation):
    result = list(range(4))
    for round_index in range(18):
        words = reference['stream'](7, permutation, round_index, 0x5A0B5A0B)
        pivot = reference['below'](words, 4)
        salt = next(words)
        for row, position in enumerate(result):
            partner = (pivot - position) % 4
            if reference['hash_word'](max(position, partner) ^ salt) >> 31:
                result[row] = partner
    return result

weights = np.array([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]])
weights=weights/weights.sum(axis=1)[:,None]
x = np.array([1.,2.,4.,8.]); z=x-x.mean(); ss=z@z
for statistic in ['bivariateMoran', 'leesL']:
    def value(indices):
        y=z[indices]
        return 4*(z@weights@y)/(weights.sum()*ss) if statistic=='bivariateMoran' else 4*((weights@z[indices])@(weights@y))/(np.sum(weights.sum(axis=1)**2)*ss)
    observed=value(list(range(4))); simulated=[value(order(p)) for p in range(1,10)]
    print(statistic,observed,simulated,'upper',sum(v>=observed for v in simulated),'lower',sum(v<=observed for v in simulated))
# Local bivariate Moran uses sample-normalized factor (n-1)/sqrt(SSx SSy).
for y in [x,x[::-1]]:
    zy=y-y.mean()
    print('localBV',(3*z*(weights@zy)/math.sqrt(ss*(zy@zy))).tolist())
