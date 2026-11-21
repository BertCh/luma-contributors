# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/global-spatial-statistics/global-mathematical-audit.oracle.py

Independent definitions: dense quadratic forms and exhaustive row permutations
for randomization means/variances (population divisor, not ddof=1). Bivariate
Moran permutes y only. Binary joins enumerate every placement of the fixed black
population, count directed edges and divide by two (asymmetric half joins).
Normality moments come from the uniform centered Gaussian sphere identity:
Var(u'A u)=2/(d(d+2))*(tr(A^2)-tr(A)^2/d), d=n-1, A projected off the mean.
This does not reuse the implementation's S1/S2/kurtosis formulas or its centered
General-G variance identity. p uses libm erfc and the documented 2^-126 floor.
"""
import itertools
import json
import math
import numpy as np

VALUES = [0,1,1,3,0,5,8,999,2,float('nan')]
SECOND = [9,4,4,1,5,2,8,999,7,2]
MASK = [1,1,1,1,1,1,1,0,1,1]
EDGES = [[(0,99),(1,1),(2,2),(7,4)],[(0,1),(1,99),(3,1)],[(1,1),(2,99),(3,2),(4,0)],[(2,1),(3,99),(4,3)],[(3,2),(4,99),(5,1)],[(4,1),(5,99),(6,2)],[(0,2),(5,1),(6,99)],[(0,1)],[(8,99)],[(0,1)]]

def probability(score, tail):
    if not np.isfinite(score): return None
    value = math.erfc(abs(score)/math.sqrt(2)) if tail == 'two-sided' else .5*math.erfc((score if tail == 'greater' else -score)/math.sqrt(2))
    return max(2**-126,min(1,value))

def block(observed, expected, normal, random, tail):
    result = [observed,expected]
    for variance in [normal,random]:
        score = (observed-expected)/math.sqrt(variance) if variance is not None and variance>0 else float('nan')
        result += [variance,score,probability(score,tail)]
    return result

def compute(keep, standardize, tail='two-sided'):
    matrix = np.zeros((10,10))
    for row, edges in enumerate(EDGES):
        for neighbor,weight in edges:
            if row != neighbor and weight>0: matrix[row,neighbor]=weight
    included = np.flatnonzero(np.array(MASK,dtype=bool) & np.isfinite(VALUES) & np.isfinite(SECOND))
    islands = [row for row in included if matrix[row,included].sum()==0]
    active = np.array([row for row in included if keep or row not in islands])
    weights = matrix[np.ix_(active,active)]
    if standardize:
        totals = weights.sum(axis=1,keepdims=True)
        weights = np.divide(weights,totals,out=np.zeros_like(weights),where=totals!=0)
    binary = (weights>0).astype(float)
    x = np.array(VALUES)[active]; y = np.array(SECOND)[active]; n=len(x)
    centered = x-x.mean(); second=y-y.mean(); squares=centered@centered
    s0=weights.sum(); s1=((weights+weights.T)**2).sum()/2
    degree=weights.sum(axis=0)+weights.sum(axis=1); s2=degree@degree
    permutations=np.array(list(itertools.permutations(x)))
    deviations=permutations-x.mean()
    quadratic=lambda arrays,matrix: np.einsum('bi,ij,bj->b',arrays,matrix,arrays)
    moran=quadratic(deviations,weights)*n/s0/squares
    laplacian=np.diag(degree)-weights-weights.T
    geary=quadratic(permutations,laplacian)*(n-1)/(2*s0*squares)
    general=quadratic(permutations,weights)/(x.sum()**2-x@x)
    permuted_y=np.array(list(itertools.permutations(y)))-y.mean()
    bivariate=(permuted_y @ (weights.T@centered))*n/s0/math.sqrt(squares*(second@second))
    projection=np.eye(n)-np.ones((n,n))/n
    def normal_variance(matrix,scale):
        projected=projection@matrix@projection
        return scale**2*2/((n-1)*(n+1))*(np.trace(projected@projected)-np.trace(projected)**2/(n-1))
    summary=[n,s0,s1,s2,x.mean(),x.var(),int(np.count_nonzero(x)),len(islands)]
    summary += block(moran[0],moran.mean(),normal_variance((weights+weights.T)/2,n/s0),moran.var(),tail)
    summary += block(geary[0],geary.mean(),normal_variance(laplacian,(n-1)/(2*s0)),geary.var(),tail)
    summary += block(general[0],general.mean(),general.var(),general.var(),tail)
    summary += block(bivariate[0],0,None,bivariate.var(),tail)
    black=(x!=0).astype(float); placements=[]
    for locations in itertools.combinations(range(n),int(black.sum())):
        labels=np.zeros(n); labels[list(locations)]=1; placements.append(labels)
    labels=np.array(placements)
    bb=quadratic(labels,binary)/2
    binary_laplacian=np.diag(binary.sum(axis=0)+binary.sum(axis=1))-binary-binary.T
    bw=quadratic(labels,binary_laplacian)/2
    observed_bb=black@binary@black/2
    observed_bw=black@binary_laplacian@black/2
    observed_ww=(1-black)@binary@(1-black)/2
    def join_inference(observed,distribution):
        variance=distribution.var(); score=(observed-distribution.mean())/math.sqrt(variance)
        return [distribution.mean(),variance,score,probability(score,tail)]
    summary += [observed_bb,observed_bw,observed_ww,binary.sum()/2] + join_inference(observed_bb,bb)+join_inference(observed_bw,bw)
    clean=[None if value is None or not np.isfinite(value) else float(value) for value in summary]
    return {'summary':clean,'counts':[n,len(islands),int(black.sum())], 'joinCounts':[int(2*observed_bb),int(2*observed_bw),int(2*observed_ww)]}

if __name__=='__main__':
    for keep,standardize in [(False,False),(False,True),(True,False),(True,True)]:
        print('keep' if keep else 'exclude','row' if standardize else 'raw',json.dumps(compute(keep,standardize)))
