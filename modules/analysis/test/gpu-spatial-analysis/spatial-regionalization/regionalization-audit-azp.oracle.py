# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regionalization/regionalization-audit-azp.oracle.py
"""Independent direct-SSD and graph-connectivity oracle, without moment/gain identities."""
import numpy as np

values = np.array([0, 2, 8, 10, 12, 14, 30, 32, 100], dtype=float)
labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3])
edges = [(row, row + 1) for row in range(7)]  # row 8 is an island


def objective(table, partition):
    return sum(np.sum((table[partition == region] - np.mean(table[partition == region])) ** 2)
               for region in np.unique(partition))


def connected(partition):
    for region in np.unique(partition):
        members = set(np.flatnonzero(partition == region))
        reached = {min(members)}
        while True:
            expanded = reached | {second for first, second in edges + [(b, a) for a, b in edges]
                                  if first in reached and second in members}
            if expanded == reached:
                break
            reached = expanded
        if reached != members:
            return False
    return True


for name, table in [('raw', values), ('shifted', values + 64), ('scaled', values * 3),
                    ('standardized', (values - values.mean()) / values.std()),
                    ('affine-standardized', ((values * 3 + 64) - (values * 3 + 64).mean())
                     / (values * 3 + 64).std())]:
    before = objective(table, labels)
    improving = []
    for row in range(len(labels)):
        for destination in range(5):  # includes empty region slot four
            if destination == labels[row]:
                continue
            if not any((first == row and labels[second] == destination) or
                       (second == row and labels[first] == destination) for first, second in edges):
                continue
            candidate = labels.copy()
            candidate[row] = destination
            if np.sum(candidate == labels[row]) < 1 or not connected(candidate):
                continue
            after = objective(table, candidate)
            if after < before:
                improving.append((row, destination, after))
    assert len(improving) == 1 and improving[0][:2] == (2, 1)
    print(name, 'before', before, 'only improving move', improving[0])
print('final labels', [0, 0, 1, 1, 1, 1, 2, 2, 3])
print('population variance', values.var())
for exponent in [-80, 64, 120]:
    scaled = (values * 2.0 ** exponent).astype(np.float32).astype(float)
    standardized = (scaled - scaled.mean()) / scaled.std()
    final_labels = np.array([0, 0, 1, 1, 1, 1, 2, 2, 3])
    print('power-of-two scale', exponent, 'standardized final SSD',
          objective(standardized, final_labels))
print('floor 2 allows the move; floor 3 blocks it; disconnected island adds one component')
