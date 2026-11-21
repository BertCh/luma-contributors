# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent precision review probes; no implementation imports.

Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/distribution-dynamics/audit-review.oracle.py
R1/R2/R3/R4/R5/R10 literals are asserted in the owned audit specs.
"""
import itertools
import json
import math

import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2, norm


def variance_partition(values):
    """Enumerate every feasible partition, holding the true whole-series mean fixed."""
    center = values.mean()
    candidates = []
    for selected in itertools.product((False, True), repeat=len(values) - 1):
        cuts = [index + 1 for index, take in enumerate(selected) if take]
        boundaries = [0, *cuts, len(values)]
        segments = [values[start:end] for start, end in zip(boundaries, boundaries[1:])]
        if any(len(segment) < 2 for segment in segments):
            continue
        cost = sum(len(segment) * (math.log(2 * math.pi)
                   + math.log(np.mean((segment - center) ** 2)) + 1)
                   for segment in segments) + 3 * len(cuts)
        candidates.append((float(cost), cuts))
    return min(candidates)


def old_tree_squared(values):
    """Emulate only the old float32 reduction arithmetic, not an expectation oracle."""
    def reduce_tree(terms):
        scratch = np.zeros(256, dtype=np.float32)
        scratch[:len(terms)] = terms
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            scratch[:stride] += scratch[stride:2 * stride]
        return scratch[0]
    mean = np.float32(reduce_tree(values) / np.float32(len(values)))
    return float(reduce_tree((values - mean) ** 2))


result = {}
for first, second in ((4143, 4141), (574, 528)):
    observed = np.array([[first, second], [second, first]], dtype=np.float64)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    ratio = float(2 * xlogy(observed, observed / expected).sum())
    result[f'R1-{first}-{second}'] = dict(likelihoodRatio=ratio, degreesOfFreedom=1,
                                        tail=float(chi2.sf(ratio, 1)))

values = np.array([0] * 4 + [0.0625] * 4)
result['R2'] = dict(translated=variance_partition(values + 1000000),
                    centered=variance_partition(values))
rounded_center = float(np.float32((values + 1000000).mean()))
old_squares = (values + 1000000 - rounded_center) ** 2
result['R2']['oldRoundedMean'] = rounded_center
result['R2']['oldSquaredDeviations'] = old_squares.tolist()
old_floor = float(old_squares.mean()) * 1e-11
result['R2']['oldPredictedSplitAtFourCost'] = float(3 + sum(
    4 * (math.log(2 * math.pi) + math.log(max(float(segment.mean()), old_floor)) + 1)
    for segment in (old_squares[:4], old_squares[4:])))

weights = np.array([1 + 2 ** -12] + [1] * 7)
old_weights = weights.astype(np.float32)
old_spread = np.float32(np.float32(8) * np.sum(old_weights ** 2, dtype=np.float32)
                        - np.sum(old_weights, dtype=np.float32) ** 2)
scores = (np.arange(8) - 3.5) / math.sqrt(5.25)
result['R3'] = dict(exactSpread=float(8 * np.sum((weights - weights.mean()) ** 2) / 7),
                    oldSpread=float(old_spread), scores=scores.tolist(),
                    tails=(2 * norm.sf(np.abs(scores))).tolist())
result['R4'] = {str(value): float(2 * norm.sf(float(np.float32(value))))
                for value in (1, 0.75000001)}

for offset in (0, 1000000):
    first = np.array([offset, offset, offset + .0625, offset + .0625])
    second = np.full(4, offset + .0625)
    groups = [first, second, np.concatenate((first, second))]
    squares = [float(np.sum((group - group.mean()) ** 2)) for group in groups]
    result[f'R5-{offset}'] = dict(squaredDeviations=squares,
                                 gvf=1 - sum(squares[:2]) / squares[2],
                                 oldSquaredDeviations=[old_tree_squared(group.astype(np.float32))
                                                       for group in groups])

count = 63
statistic = count * (count - 1) // 2
variance = count * (count - 1) * (2 * count + 5) / 18
score = (statistic - 1) / math.sqrt(variance)
result['R10'] = dict(statistic=statistic, variance=variance, score=score,
                     tail=float(2 * norm.sf(abs(score))), senSlope=1,
                     detrendedVariance=0, correctionFactor=1)
# Second review: expectations use Decimal/closed forms, never the float32 probes below.
from decimal import Decimal, getcontext
getcontext().prec = 80
rows = [[4721, 4722], [4720, 4721]]
total = sum(map(sum, rows))
ratio = Decimal(0)
for row in rows:
    for column, observed in enumerate(row):
        expected = Decimal(sum(row)) * sum(other[column] for other in rows) / total
        ratio += Decimal(observed) * (Decimal(observed) / expected).ln() - observed + expected
ratio *= 2
result['review2-R1'] = dict(likelihoodRatio=str(ratio), tail=float(chi2.sf(float(ratio), 1)))

deviations = np.r_[np.full(1600, np.float32(-.8)), np.full(6400, np.float32(.2))]
relative_weight = np.float32(1 - (1 + 2**-12))
numerator = np.float32(0)
for value in deviations[1:]:
    numerator = np.float32(numerator + np.float32(relative_weight * value))
moment = np.zeros(2, dtype=np.float32)
for index, weight in enumerate(np.r_[np.float32(0), np.full(7999, relative_weight)]):
    difference = np.float32(weight - moment[0])
    mean = np.float32(moment[0] + np.float32(difference / np.float32(index+1)))
    moment = np.array([mean, np.float32(moment[1] + np.float32(difference * np.float32(weight - mean)))], dtype=np.float32)
spread = np.float32(moment[1] * np.float32(8000/7999))
old_score = float(np.float32(numerator / np.float32(np.float32(.4) * np.sqrt(spread))))
result['review2-R3'] = dict(score=-2, tail=float(2*norm.sf(2)), oldScore=old_score, oldTail=float(2*norm.sf(abs(old_score))))

length = 2**22
outlier = float(np.float32(-1.1))
result['review2-R11'] = dict(overallSquaredDeviations=outlier**2*(1-1/length), withinSquaredDeviations=outlier**2*(1-2/length), gvf=1/(length-1))
def tree_total(terms):
    for stride in (128,64,32,16,8,4,2,1):
        terms[:stride] += terms[stride:2*stride]
    return terms[0]
def old_skew_squared(count):
    sums = np.zeros(256, dtype=np.float32)
    for index in range(count//256):
        increments = np.full(256, -np.float32(outlier))
        if index == 0:
            increments[0] = 0
        sums += increments
    center = np.float32(tree_total(sums) / np.float32(count))
    terms = np.zeros(256, dtype=np.float32)
    for index in range(count//256):
        offsets = np.full(256, -np.float32(outlier))
        if index == 0:
            offsets[0] = 0
        terms += np.float32(offsets-center)**2
    return float(tree_total(terms))
old_within, old_overall = old_skew_squared(length//2), old_skew_squared(length)
result['review2-R11'].update(oldOverall=old_overall, oldGVF=float(np.float32(1)-np.float32(old_within/old_overall)))

series = list(map(Decimal, ['10000000','-10000000','1','1','1.25','1.25','1','1']))
partitions = []
for choices in itertools.product((False,True),repeat=7):
    cuts = [index+1 for index,take in enumerate(choices) if take]
    boundaries = [0,*cuts,8]
    score = Decimal(len(cuts))/64
    for start,end in zip(boundaries,boundaries[1:]):
        segment=series[start:end]
        mean=sum(segment)/len(segment)
        score += sum((value-mean)**2 for value in segment)
    partitions.append((score,cuts))
score,cuts=min(partitions)
result['review2-R12'] = dict(cost=str(score), cuts=cuts, oldOffsets=[float(np.float32(np.float32(value)-np.float32(1e7))) for value in (1,1.25)])
tiny=np.float32(2**-66)
result['review2-R13'] = dict(oldSquaredDifference=float(np.float32(np.float32(tiny*np.float32(2**-12))**2)), expectedScores=scores.tolist())
print(json.dumps(result, indent=2))
