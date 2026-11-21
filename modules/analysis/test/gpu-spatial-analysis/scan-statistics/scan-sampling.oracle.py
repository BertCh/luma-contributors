# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/scan-statistics/scan-sampling.oracle.py

Independent Poisson-law bounds from SciPy, with 512 fixed-seed draws per mean.
The sum is Poisson(512*mean); each CDF-bin count is Binomial(512, F(threshold)).
Use two-sided six-sigma tail probability (about 2e-9). The unbiased sample
variance has variance mean/n + 2*mean**2/(n-1); its six-standard-error interval
rejects constant/rounded-mean draws. These are distributional checks, not proof
of independence. The >=1e7 route is documented as a rounded normal approximation.
"""
import json
import math
from scipy.stats import binom, poisson
from itertools import product
from math import comb, factorial
from collections import defaultdict

sample_count = 512
tail = 1e-9
cases = []
for mean in [0, 0.5, 2, 29, 30, 100, 1000000, 10000000]:
    variance_error = 6 * math.sqrt(mean / sample_count + 2 * mean * mean / (sample_count - 1))
    threshold = int(poisson.ppf(0.5, mean))
    probability = poisson.cdf(threshold, mean)
    cases.append(dict(mean=mean,
                      sumMinimum=int(poisson.ppf(tail, sample_count * mean)),
                      sumMaximum=int(poisson.ppf(1 - tail, sample_count * mean)),
                      varianceMinimum=max(0, mean - variance_error),
                      varianceMaximum=mean + variance_error,
                      threshold=threshold,
                      belowMinimum=int(binom.ppf(tail, sample_count, probability)),
                      belowMaximum=int(binom.ppf(1 - tail, sample_count, probability))))

def likelihood(observed, expected, total, population=None, inside=None):
    if observed <= expected:
        return 0.0
    actual = [observed, total - observed]
    null = [expected, total - expected]
    if population is not None:
        actual += [inside - observed, population - inside - total + observed]
        null += [inside - expected, population - inside - total + expected]
    return sum(value * math.log(value / reference) for value, reference in zip(actual, null) if value)


conditional = []
for model in ["poisson", "bernoulli", "bernoulli-complement", "space-time-permutation"]:
    distribution = defaultdict(float)
    if model == "poisson":
        for allocation in product(range(6), repeat=8):
            if sum(allocation) != 5:
                continue
            score = max(likelihood(value, 5 / 8, 5) for value in allocation)
            probability = factorial(5) / math.prod(factorial(value) for value in allocation) / 8 ** 5
            distribution[round(score, 12)] += probability
    elif model.startswith("bernoulli"):
        total_cases = 9 if model == "bernoulli-complement" else 5
        population = [3, 2, 1, 2, 1, 2, 1, 2]
        for allocation in product(*(range(value + 1) for value in population)):
            if sum(allocation) != total_cases:
                continue
            score = max(likelihood(value, size * total_cases / 14, total_cases, 14, size) for value, size in zip(allocation, population))
            probability = math.prod(comb(size, value) for size, value in zip(population, allocation)) / comb(14, total_cases)
            distribution[round(score, 12)] += probability
    else:
        # Each row has two cases; times are chosen uniformly without replacement
        # from eight bucket-0 and eight bucket-1 labels. Weight of a table is the
        # product of binomial(2, rowBucket0), divided by binomial(16,8).
        for allocation in product(range(3), repeat=8):
            if sum(allocation) != 8:
                continue
            score = max(likelihood(max(value, 2 - value), 1, 16) for value in allocation)
            distribution[round(score, 12)] += math.prod(comb(2, value) for value in allocation) / comb(16, 8)
    cumulative = 0
    thresholds = []
    for score, probability in sorted(distribution.items()):
        cumulative += probability
        probability = 1 if cumulative > 1 - 1e-12 else cumulative
        thresholds.append(dict(score=score, probability=probability,
                               minimum=int(binom.ppf(1e-7, 1024, probability)),
                               maximum=int(binom.ppf(1 - 1e-7, 1024, probability))))
    assert abs(cumulative - 1) < 1e-12
    conditional.append(dict(model=model, thresholds=thresholds))
print(json.dumps(dict(sampleCount=sample_count, cases=cases, conditionalSampleCount=1024, conditional=conditional), indent=2))
