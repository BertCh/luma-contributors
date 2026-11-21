# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/gwr-review.oracle.py
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/gwr-review.oracle.py
# Independent NumPy weighted raw-design SVD, with fixed assignments from the public shuffle.
import numpy as np
from scipy.stats import t

assignments = [
    [7,4,13,10,3,15,1,9,0,2,5,12,6,11,14,8],
    [6,5,4,12,3,15,9,11,2,7,10,14,0,8,13,1],
    [3,14,10,2,4,1,7,0,6,11,12,15,9,13,5,8],
    [14,8,11,6,7,2,0,10,1,5,4,15,13,12,3,9],
    [2,8,7,4,1,11,0,12,14,3,5,13,6,15,10,9]
]
rows=np.arange(16)
design=np.column_stack([np.ones(16), rows%2])
response=(rows//4)**2+rows%2
def fit(assignment,bandwidth):
    x=design[assignment]; y=response[assignment]
    coefficients=[]; residuals=[]; trace=0
    for focal in rows:
        weights=np.exp(-.5*((rows-focal)/bandwidth)**2)
        mapping=np.linalg.pinv(np.sqrt(weights)[:,None]*x)*np.sqrt(weights)
        beta=mapping@y
        coefficients.append(beta)
        residuals.append(y[focal]-x[focal]@beta)
        trace+=x[focal]@mapping[:,focal]
    rss=np.sum(np.array(residuals)**2)
    score=16*np.log(rss/16)+16*np.log(2*np.pi)+16*(16+trace)/(16-2-trace)
    return np.array(coefficients),score
observed,scores=fit(rows,2)[0],[fit(rows,b)[1] for b in [2,8]]
print('scores:',scores)
spreads=np.array([np.std(fit(a,2)[0],axis=0) for a in [rows,*assignments]])
print('spreads:',spreads.ravel().tolist())
g=np.sum(spreads[1:]>=spreads[0],axis=0)
mean=spreads[1:].mean(axis=0);sd=spreads[1:].std(axis=0)
print('table:',np.column_stack([spreads[0],(g+1)/6,g,mean,sd,(spreads[0]-mean)/sd]).ravel().tolist())
print('tail:',2*t.sf(1.96,200000))
print('tail range:',[(value,degrees,float(2*t.sf(value,degrees))) for degrees in [4096,20000,200000] for value in [.5,1.96,3,6,8]])
print('constant population variance:',np.var(np.full(10,np.float32(.1)).astype(float)))
print('large-offset population SD:',np.std([100000000.,100000001.]))
