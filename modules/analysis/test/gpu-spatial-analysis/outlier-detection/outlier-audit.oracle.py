# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/outlier-detection/outlier-audit.oracle.py
"""Exact-rational Breunig reachability definition on an explicit eight-row CSR.
Tied nearest neighbors, coincident cluster, isolated/masked row, invalid neighbor.
Invalid IDs are excluded from every stage, per the contributor's documented contract.
"""
from fractions import Fraction
import json

neighbors = [[(1, 1), (2, 1), (99, 1000)], [(0, 1), (2, 2)], [(0, 1), (1, 2)], [(4, 0), (5, 0)], [(3, 0), (5, 0)], [(3, 0), (4, 0)], [(0, 8), (1, 9)], []]
active = [[(row, Fraction(distance)) for row, distance in entries if row < len(neighbors)] for entries in neighbors]
k_distance = [max([distance for _, distance in entries], default=Fraction(0)) for entries in active]
density = [1/(sum(max(distance, k_distance[row]) for row, distance in entries)/len(entries) + Fraction(1, 2)) if entries else Fraction(0) for entries in active]
lof = [sum(density[row] for row, _ in entries)/len(entries)/density[index] if entries else Fraction(1) for index, entries in enumerate(active)]
print(json.dumps(dict(kDistance=list(map(float, k_distance)), localReachabilityDensity=list(map(float, density)), lof=list(map(float, lof)))))
for threshold in [Fraction(5, 4), Fraction(1, 2)]:
    flags = [int(bool(entries) and score > threshold) for entries, score in zip(active, lof)]
    print('threshold', float(threshold), 'outlier', flags, 'outlierCount', sum(flags))
print('Scaling all distances and densityFloor by 2 doubles kDistance, halves density and preserves LOF/flags.')
