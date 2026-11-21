# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/categorical-association/categorical-association-audit.oracle.py

Exact Fraction co-location shares/chance correction; SciPy binomial.sf for local
composition and entropy for endpoint marginals. Invalid category=4, category 3 empty.
"""
from fractions import Fraction
import numpy as np
from scipy.stats import binom, entropy

categories = [0, 0, 1, 1, 2, 2, 0, 4]
neighbors = [[(0, 1), (1, 1), (2, 3), (7, 2)], [(0, 2), (3, 2)], [(1, 1), (4, 1), (3, 2)], [(2, 1), (4, 0)], [(0, 3), (5, 1)], [], [], [(0, 1)]]
counts = [categories.count(category) for category in range(4)]
total = sum(counts)
pairs = np.zeros((4, 4), dtype=int)
local = []
dependence = []
shares_by_category = [[] for _ in range(4)]
for row, own in enumerate(categories):
    links = [(other, weight) for other, weight in neighbors[row] if other != row and weight > 0 and categories[other] < 4] if own < 4 else []
    if not links:
        local.extend([float('nan')] * 4)
        dependence.extend([0, 0] + [float('nan')] * 4)
        continue
    for other, _ in links:
        pairs[own, categories[other]] += 1
    shares = [Fraction(sum(weight for other, weight in links if categories[other] == category), sum(weight for _, weight in links)) for category in range(4)]
    shares_by_category[own].append(shares)
    local.extend([float(shares[category] / Fraction(counts[category] - (own == category), total - 1)) if counts[category] > (own == category) else float('nan') for category in range(4)])
    like, degree = sum(categories[other] == own for other, _ in links), len(links)
    probability = counts[own] / total
    dependence.extend([like, degree, like / degree, probability, binom.sf(like - 1, degree, probability), (like - degree * probability) / np.sqrt(degree * probability * (1 - probability))])
global_values = []
for own, rows in enumerate(shares_by_category):
    for category in range(4):
        chance = Fraction(counts[category] - (own == category), total - 1)
        global_values.append(float(sum(row[category] for row in rows) / len(rows) / chance) if rows and chance > 0 else float('nan'))
origin, destination = entropy(pairs.sum(axis=1)), entropy(pairs.sum(axis=0))
joint = entropy(pairs.ravel())
mutual = origin + destination - joint
print('rowCounts', counts, 'pairCounts', pairs.ravel().tolist())
print('entropy', [entropy(counts), origin, destination, joint, mutual, mutual / destination, int(pairs.sum()), total])
print('global', global_values)
print('local', local)
print('dependence', [float(value) for value in dependence])
print('R7 focal statistics', [1134, 2190, 1134 / 2190, .5, float(binom.sf(1133, 2190, .5)), (1134 - 2190 * .5) / np.sqrt(2190 * .25)])
