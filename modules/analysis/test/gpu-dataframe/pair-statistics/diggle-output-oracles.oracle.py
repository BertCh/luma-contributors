# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Reproduce: /usr/bin/python3 modules/analysis/test/gpu-dataframe/pair-statistics/diggle-output-oracles.oracle.py
# Independent integer Philox (Random123 4x32-10), permutation matrices for the swap-or-not
# construction (Hoang/Morris/Rogaway 2012), and SciPy distance matrices; no repository imports.
import json
import numpy as np
from scipy.spatial.distance import cdist

MASK = (1 << 32)-1
def philox(counter, key):
    a,b,c,d = counter
    x,y = key
    for _ in range(10):
        first, second = a*0xd2511f53, c*0xcd9e8d57
        a,b,c,d = ((second >> 32)^b^x)&MASK, second&MASK, ((first >> 32)^d^y)&MASK, first&MASK
        x,y = (x+0x9e3779b9)&MASK, (y+0xbb67ae85)&MASK
    return [a,b,c,d]
assert philox([0,0,0,0],[0,0]) == [0x6627e8d5,0xe169c58d,0xbc57ac4c,0x9b00dbd8]

def bit(value):
    value = ((value^(value>>16))*0x7feb352d)&MASK
    value = ((value^(value>>15))*0x846ca68b)&MASK
    return (value^(value>>16)) >> 31

def permutation(slot):
    matrix = np.eye(4, dtype=int)
    for round_index in range(18):
        words = philox([0,slot,round_index,0x44494747],[21,256])
        # Lemire's bounded draw for n=4 is exactly the top two bits; no rejection.
        pivot,salt = words[0]>>30,words[1]
        transform = np.eye(4,dtype=int)
        for first in range(4):
            second = (pivot-first)%4
            if first < second and bit(second^salt):
                transform[[first,second]] = transform[[second,first]]
        matrix = transform @ matrix
    return np.argmax(matrix,axis=0)

positions = np.array([[1,1],[2,1],[5,1],[6,1]])
times = np.array([0,1,3,7])
distance = cdist(positions,positions)
off_diagonal = ~np.eye(4,dtype=bool)
spatial_masks = np.array([(distance<=bound)&(distance<4)&off_diagonal for bound in [2,4]])
temporal_masks = lambda values: np.array([(np.abs(values[:,None]-values)<=bound)&(np.abs(values[:,None]-values)<8)&off_diagonal for bound in [4,8]])
spatial = spatial_masks.sum(axis=(1,2))*24/12
temporal = temporal_masks(times).sum(axis=(1,2))*8/12
curves = []
permutations = [np.arange(4)]+[permutation(slot) for slot in range(1,6)]
for order in permutations:
    joint = np.einsum('aij,bij->ab',spatial_masks.astype(int),temporal_masks(times[order]).astype(int))*192/12
    curves.append(joint-spatial[:,None]*temporal)
curves = np.array(curves)
statistics = curves.sum(axis=(1,2))
mean = statistics[1:].mean()
variance = statistics[1:].var(ddof=1)
greater,lesser = int(sum(statistics[1:]>=statistics[0])),int(sum(statistics[1:]<=statistics[0]))
probability_greater,probability_lesser = (greater+1)/6,(lesser+1)/6
result = dict(permutations=[order.tolist() for order in permutations], d=curves[0].ravel().tolist(),
              k=(curves[0]+spatial[:,None]*temporal).ravel().tolist(), spatialK=spatial.tolist(),temporalK=temporal.tolist(),
              statistics=statistics.tolist(),permutationMean=curves[1:].mean(axis=0).ravel().tolist(),
              permutationStandardDeviation=curves[1:].std(axis=0,ddof=1).ravel().tolist(),
              summary=[statistics[0],mean,variance,greater,lesser,probability_greater,probability_lesser,min(1,2*min(probability_greater,probability_lesser)),(statistics[0]-mean)/np.sqrt(variance),4,4], status=[0])
print(json.dumps(result,indent=2))
