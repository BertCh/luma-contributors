# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-validation/conformal-validation-audit.oracle.py

Direct CV+ candidate enumeration (Barber et al.) and weighted empirical CDF,
independent of the fold-segment/binary-search implementation. All data are dyadic.
"""
from fractions import Fraction
from math import ceil, floor, inf

observations = [1, 4, 2, 6, 5, 3, 7, 8, 9, 10]
folds = [0, 1, 0, 1, 0, 1, 0, 1, 0, 2]
scales = [1, 1, 2, 1, 1, 2, 1, 1, 1, 1]
mask = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
scores = [(abs(Fraction(value - fold, scales[row])), fold)
          for row, (value, fold) in enumerate(zip(observations, folds))
          if mask[row] and fold < 2]
alpha = Fraction(1, 4)
lower_rank = floor(alpha * (len(scores) + 1))
upper_rank = ceil((1 - alpha) * (len(scores) + 1))
for shift, factor in [(0, 1), (32, 4)]:
    intervals = []
    for predictions, scale in [([10, 20], 2), ([-2, -6], Fraction(1, 2))]:
        lower = sorted(shift + factor * (predictions[fold] - scale * score)
                       for score, fold in scores)[lower_rank - 1]
        upper = sorted(shift + factor * (predictions[fold] + scale * score)
                       for score, fold in scores)[upper_rank - 1]
        intervals += [float(lower), float(upper)]
    print('CV+', {'intervals': intervals, 'status': [0, 0, 4], 'count': len(scores)})

# A weighted tie, one zero-weight masked entry, and an empty neighbor group.
weighted_scores = [1, 3, 1, 5, 5, 1, 7, 7, 999]
weights = [1, 2, 1, 1, 2, 1, 1, 1, 0]
level = (1 - alpha) * (sum(weights) + 1)
quantile = next((score for score in sorted(set(weighted_scores))
                 if sum(weight for other, weight in zip(weighted_scores, weights)
                        if other <= score) >= level), inf)
print('weighted', {'quantile': float(quantile), 'intervals': [10 - 2 * quantile, 10 + 2 * quantile]})

# R6: enumerate every possible query of the eight groups; random row order cannot
# restore the CV+ theorem's data-independent fold assumption.
covered = 0
for query in range(32):
    candidates = sorted(int(row // 4 != query // 4) for row in range(32) if row != query)
    lower, upper = candidates[7], candidates[23]
    covered += lower <= 0 <= upper
print('R6', {'covered': covered, 'queries': 32, 'ranks': [8, 24], 'interval': [lower, upper]})

# R9: exact common-scale cancellation, including query mass at infinity.
for factor in [1, 2 ** 126]:
    masses = [factor] * 7 + [2 * factor]
    for self_mass in [factor, 0]:
        level = Fraction(3, 4) * (sum(masses) + self_mass)
        quantile = next(score for score in range(1, 9) if sum(masses[:score]) >= level)
        print('R9', factor, self_mass, [-quantile, quantile], 'status', 0)

# Review 2 R5: all masses remain exact and positive, even at minimum float32 subnormal.
for factor in [Fraction(1), Fraction(1, 2**149), Fraction(1, 2**100), Fraction(2**126)]:
    masses = [factor] * 6 + [2 * factor] * 2
    level = Fraction(3, 4) * (sum(masses) + factor)
    quantile = next(score for score in range(1, 9) if sum(masses[:score]) >= level)
    assert level == Fraction(33, 4) * factor and sum(masses[:7]) == 8 * factor
    print('review 2 R5', str(factor), [-quantile, quantile], 'status', 0)
