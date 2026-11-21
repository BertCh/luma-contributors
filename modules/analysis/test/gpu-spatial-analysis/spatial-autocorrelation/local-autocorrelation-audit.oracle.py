# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-autocorrelation/local-autocorrelation-audit.oracle.py

Definition oracle, independent of the TypeScript/WGSL formulas:
Moran_Local I=(n-1)c_i sum(w_ij c_j)/sum(c^2), esda scaling.
Its conditional null fixes c_i and exhaustively permutes the other values;
z=(observed I - empirical null mean)/population null standard deviation.
Gi uses the same conditional permutations of sum(w_ij x_j), without sign(c_i).
Gi* permutes ALL values, including the focus, in sum(w_ij x_j)+selfWeight*x_i.
These exhaustive distributions avoid using the analytic variance under audit.
All diagonals are ignored; positive off-diagonal weights only; masked/nonfinite
rows are omitted before a single island pass. Row standardization follows that
selection. Normal p=erfc(abs(z)/sqrt(2)), two-sided, not permutation p-values.
Ties are retained as distinct observations in the permutation population.
"""

import itertools
import json
import math
import numpy as np
from scipy.stats import norm

VALUES = np.array([1, 1, 2, 4, 7, 10, 13, 999, 5, np.nan], dtype=float)
MASK = np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 1], dtype=bool)
EDGES = [
    [(0, 99), (1, 1), (2, 2), (7, 4)],
    [(0, 1), (1, 99), (3, 1)],
    [(1, 1), (2, 99), (3, 2), (4, 0)],
    [(2, 1), (3, 99), (4, 3)],
    [(3, 2), (4, 99), (5, 1)],
    [(4, 1), (5, 99), (6, 2)],
    [(0, 2), (5, 1), (6, 99)],
    [(0, 1)], [(8, 99)], [(0, 1)],
]


def compute(keep, standardize):
    included = np.flatnonzero(MASK & np.isfinite(VALUES))
    matrix = np.zeros((len(VALUES), len(VALUES)))
    for row, edges in enumerate(EDGES):
        for neighbor, weight in edges:
            if row != neighbor and weight > 0:
                matrix[row, neighbor] = weight
    islands = [row for row in included if matrix[row, included].sum() == 0]
    active = np.array([row for row in included if keep or row not in islands])
    weights = matrix[np.ix_(active, active)]
    if standardize:
        totals = weights.sum(axis=1, keepdims=True)
        weights = np.divide(weights, totals, out=np.zeros_like(weights), where=totals != 0)
    values = VALUES[active]
    centered = values - values.mean()
    permutations = np.array(list(itertools.permutations(values)))
    result = {
        'globalStatistics': [len(active), values.mean(), values.var(), values.std(), len(islands)],
        'globalCounts': [len(active), len(islands)],
        'neighborCounts': [0] * len(VALUES),
    }
    for name in ['localI', 'spatialLag', 'moranZ', 'moranP', 'giZ', 'giP', 'starZ', 'starP']:
        result[name] = [None] * len(VALUES)
    result['quadrants'] = [0] * len(VALUES)
    result['ungatedQuadrants'] = [0] * len(VALUES)
    result['giBins'] = [0] * len(VALUES)
    result['starBins'] = [0] * len(VALUES)
    for index, row in enumerate(active):
        row_weights = weights[index]
        count = np.count_nonzero(row_weights)
        result['neighborCounts'][row] = int(count)
        lag = float(row_weights @ centered)
        observed = (len(active)-1) * centered[index] * lag / (centered @ centered)
        result['localI'][row] = observed
        result['spatialLag'][row] = lag
        if count == 0:
            continue
        quadrant = (1 if lag > 0 else 4) if centered[index] > 0 else (2 if lag > 0 else 3)
        if lag == 0 or centered[index] == 0:
            quadrant = 0
        result['ungatedQuadrants'][row] = quadrant
        others = np.delete(values, index)
        other_weights = np.delete(row_weights, index)
        conditional = np.array(list(itertools.permutations(others))) @ other_weights
        null_i = ((len(active)-1) * centered[index] / (centered @ centered)) * (
            conditional - values.mean() * other_weights.sum())
        moran_z = (observed - null_i.mean()) / null_i.std()
        gi_observed = row_weights @ values
        gi_z = (gi_observed - conditional.mean()) / conditional.std()
        star_weights = row_weights.copy()
        star_weights[index] = 1
        null_star = permutations @ star_weights
        star_z = (values @ star_weights - null_star.mean()) / null_star.std()
        for name, score in [('moran', moran_z), ('gi', gi_z), ('star', star_z)]:
            result[name+'Z'][row] = float(score)
            result[name+'P'][row] = math.erfc(abs(score)/math.sqrt(2))
        result['quadrants'][row] = quadrant if result['moranP'][row] <= 0.2 else 0
        for name in ['gi', 'star']:
            level = sum(result[name+'P'][row] <= alpha for alpha in [0.1, 0.05, 0.01])
            result[name+'Bins'][row] = level if result[name+'Z'][row] > 0 else -level
    # Positive affine copies leave I, z, p and class codes invariant; lag scales by 2.
    shifted = 2 * values + 32
    assert np.allclose(shifted-shifted.mean(), 2*centered)
    return result


if __name__ == '__main__':
    print('critical z', [float(norm.isf(alpha/2)) for alpha in [0.1, 0.05, 0.01]])
    for keep, standardize in [(False, False), (False, True), (True, False), (True, True)]:
        print(('keep' if keep else 'exclude') + ('Row' if standardize else 'Raw'))
        print(json.dumps(compute(keep, standardize), allow_nan=False))
    # An external population summarized by (count,mean=4,variance=25), observed through a small
    # resident subset. Conditional lag moments depend only on these fixed population moments.
    for population in [2**24+3, 2**32-1]:
        included = set(np.flatnonzero(MASK & np.isfinite(VALUES)))
        scores=[]; local=[]; gi=[]; star=[]
        for row,value in enumerate(VALUES):
            if row not in included:
                scores.append(None); local.append(None); gi.append(None); star.append(None); continue
            centered=value-4
            pairs=[(neighbor,weight) for neighbor,weight in EDGES[row] if neighbor!=row and neighbor in included and weight>0]
            lag=sum(weight*(VALUES[neighbor]-4) for neighbor,weight in pairs)
            local.append(float((population-1)*centered*lag/(25*population)))
            others=population-1; other_mean=-centered/others
            variance=((25*population-centered**2)/others-other_mean**2)*(others*sum(weight**2 for _,weight in pairs)-sum(weight for _,weight in pairs)**2)/(others-1)
            score=(1 if centered>0 else -1)*(lag-sum(weight for _,weight in pairs)*other_mean)/math.sqrt(variance) if centered!=0 and variance>0 else None
            scores.append(score)
            gi.append((lag-sum(weight for _,weight in pairs)*other_mean)/math.sqrt(variance) if variance>0 else None)
            star_variance=25*(population*(1+sum(weight**2 for _,weight in pairs))-(1+sum(weight for _,weight in pairs))**2)/(population-1)
            star.append((lag+centered)/math.sqrt(star_variance) if pairs and star_variance>0 else None)
        print('fixed',population,json.dumps({'localI':local,'zScores':scores,'gi':gi,'star':star,'summary':[float(np.float32(population)),4,25,5,1]}))
