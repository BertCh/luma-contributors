# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/panel-lag-poles.oracle.py

Independent Fraction arithmetic on original values; no fitted residual or floating rank test.
For unit effects, subtract each entity's first period. For random effects the same witness
gives residual variance tending to zero within each entity, a singular covariance boundary.
If y-rho*Wy-X*beta is constant per entity at admissible rho, likelihood is unbounded as
the within variance tends to zero. Status 2 and withheld floats are therefore the contract.
"""
import json
import struct
from fractions import Fraction
from pathlib import Path


def rank_fraction(matrix):
    matrix = [[Fraction(value) for value in row] for row in matrix]
    rank = 0
    for column in range(len(matrix[0])):
        pivot = next((row for row in range(rank, len(matrix)) if matrix[row][column]), None)
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        divisor = matrix[rank][column]
        matrix[rank] = [value / divisor for value in matrix[rank]]
        for row in range(rank + 1, len(matrix)):
            multiplier = matrix[row][column]
            matrix[row] = [value - multiplier * reference for value, reference in zip(matrix[row], matrix[rank])]
        rank += 1
        if rank == len(matrix):
            break
    return rank


def verify(scene, slopes, rho):
    response = list(map(Fraction, scene['response']))
    predictors = list(map(Fraction, scene['predictors']))
    columns = scene['predictorCount']
    graph = scene['weights']
    entities = len(graph['offsets']) - 1
    lag = []
    for row in range(len(response)):
        entity = row % entities
        lag.append(sum(Fraction(graph['weights'][slot]) * response[row-entity+graph['neighbors'][slot]]
                       for slot in range(graph['offsets'][entity], graph['offsets'][entity+1])))
    residual = [response[row] - rho * lag[row] - sum(slopes[column] * predictors[row*columns+column] for column in range(columns)) for row in range(len(response))]
    assert all(value == residual[row % entities] for row, value in enumerate(residual))
    # A one-ulp-sized perturbation must not be certified: augmented within rank increases.
    perturbed = response[:]
    encoded = struct.unpack('I', struct.pack('f', float(perturbed[0])))[0]
    encoded += -1 if perturbed[0] < 0 else 1
    perturbed[0] = Fraction(struct.unpack('f', struct.pack('I', encoded))[0])
    perturbed_lag = []
    for row in range(len(response)):
        entity = row % entities
        perturbed_lag.append(sum(Fraction(graph['weights'][slot]) * perturbed[row-entity+graph['neighbors'][slot]]
                                 for slot in range(graph['offsets'][entity], graph['offsets'][entity+1])))
    design, target = [], []
    for row in range(entities, len(response)):
        entity = row % entities
        design.append([predictors[row*columns+column]-predictors[entity*columns+column] for column in range(columns)] + [perturbed_lag[row]-perturbed_lag[entity]])
        target.append(perturbed[row]-perturbed[entity])
    assert rank_fraction([row+[value] for row, value in zip(design, target)]) > rank_fraction(design)
    return {'status': 2, 'floatOutputs': 'all NaN', 'hausmanStatus': 1, 'hausmanTests': 'all NaN', 'perturbationHasPole': False}


if __name__ == '__main__':
    edges = json.loads(Path(__file__).with_name('spatial-panel-lag-maximum-likelihood-operation-edge-fixtures.json').read_text())
    scenes = [scene for scene in edges['reviewCases'] if scene['name'] in ['responseCopy', 'originalPole']]
    weighted = dict(name='weightedPole', entityCount=4, periodCount=2, predictorCount=1,
                    response=[-1, 1, 2, -2, 1, -1, -2, 2],
                    weights=dict(offsets=[0,2,4,6,8], neighbors=[1,3,0,2,1,3,0,2], weights=[.375,.625,.375,.625,.625,.375,.625,.375]))
    response = list(map(Fraction, weighted['response']))
    weighted['predictors'] = []
    for row, value in enumerate(response):
        entity = row % 4
        lag = sum(Fraction(weighted['weights']['weights'][slot]) * response[row-entity+weighted['weights']['neighbors'][slot]] for slot in range(entity*2, entity*2+2))
        weighted['predictors'].append(float(value-Fraction(1,2)*lag))
    scenes.append(weighted)
    scenes.append(dict(weighted, name='weightedRationalPole',predictors=[3*value for value in weighted['predictors']]))
    scenes.append(dict(name='rationalSlopePole', predictorCount=1,
                       response=[5,1,-5,-1,-3,-6,3,6], predictors=[15,3,-15,-3,-9,-18,9,18],
                       weights=dict(offsets=[0,2,4,6,8], neighbors=[1,3,0,2,1,3,0,2], weights=[.5]*8)))
    for scene in scenes:
        slopes = [Fraction(1,3) if scene['name'] in ['rationalSlopePole','weightedRationalPole'] else Fraction(1)] + [Fraction(0)]*(scene['predictorCount']-1)
        rho = Fraction(0) if scene['name'] in ['responseCopy', 'rationalSlopePole'] else Fraction(1,2)
        print(scene['name'], verify(scene, slopes, rho))
        affine = json.loads(json.dumps(scene))
        affine['predictors'] = [value + 8 for value in scene['predictors']]
        affine['response'] = [2*value + 16*(row % 4) for row, value in enumerate(scene['response'])]
        print(scene['name']+' affine', verify(affine, [2*value for value in slopes], rho))
    print(json.dumps(weighted))
