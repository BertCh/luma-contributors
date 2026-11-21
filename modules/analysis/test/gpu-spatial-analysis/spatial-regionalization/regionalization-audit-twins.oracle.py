# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regionalization/regionalization-audit-twins.oracle.py
"""Direct NumPy SSD differences for tree cutting and adjacent Ward merging."""
import numpy as np

edges = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)]
slots = [0, 2, 4, 6, 8, 10]
skater = np.array([[0, 0], [0, 0], [10, 2], [10, 2], [1, -1], [1, -1],
                   [5, 3], [5, 3], [99, 7]], dtype=float)
ward = np.array([0, 2, 10, 14, 1, 3, 20, 26, 99], dtype=float)[:, None]


def ssd(table, members):
    return np.sum((table[members] - table[members].mean(axis=0)) ** 2)


for name, table in [('raw', skater), ('affine', 3 * skater + 64), ('tie', skater * 0)]:
    live = list(edges)
    groups = [[0, 1, 2, 3], [4, 5, 6, 7], [8]]
    cuts = []
    while len(groups) < 6:
        candidates = []
        for edge in live:
            group = next(group for group in groups if edge[0] in group)
            reached = {edge[0]}
            while True:
                expanded = reached | {second for first, second in live + [(b, a) for a, b in live]
                                      if (first, second) not in [edge, edge[::-1]] and first in reached}
                if expanded == reached:
                    break
                reached = expanded
            side = sorted(reached)
            remainder = sorted(set(group) - reached)
            if min(len(side), len(remainder)) < 2:
                continue
            gain = ssd(table, group) - ssd(table, side) - ssd(table, remainder)
            candidates.append((-gain, slots[edges.index(edge)], edge, side, remainder, group))
        if not candidates:
            break
        negative_gain, slot, edge, side, remainder, group = min(candidates)
        cuts.append((slot, -negative_gain))
        live.remove(edge)
        groups.remove(group)
        groups.extend([side, remainder])
    print('SKATER', name, sorted(groups), cuts)

for name, table in [('raw', ward), ('affine', 3 * ward + 64),
                    ('standardized', (ward - ward.mean(axis=0)) / ward.std(axis=0))]:
    groups = [[row] for row in range(9)]
    objective = 0
    for step in range(4):
        candidates = []
        for first_index, first in enumerate(groups):
            for second in groups[first_index + 1:]:
                if not any((a in first and b in second) or (b in first and a in second) for a, b in edges):
                    continue
                increase = ssd(table, first + second) - ssd(table, first) - ssd(table, second)
                candidates.append((increase, first, second))
        increase, first, second = min(candidates)
        groups.remove(first)
        groups.remove(second)
        groups.append(sorted(first + second))
        objective += increase
    print('Ward', name, sorted(groups), 'SSD', objective)
