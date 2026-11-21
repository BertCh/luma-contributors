# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-weights/weights-summary-audit.oracle.py

Dense-matrix definition of libpysal W.s0/s1/s2. Counts follow the explicitly
documented active-positive-nonself convention, and asymmetry counts listed
slots only (including a zero-weight slot with no reverse). No implementation
or existing TypeScript oracle is imported. Summary has no mask property.
"""
import numpy as np

entries = [
    {0: 0.5, 1: 2, 2: 1},
    {0: 1, 2: 3},
    {0: 1, 1: 3, 3: 4},
    {2: 0},
    {5: 2},
    {4: 2},
    {7: 0},
    {},
]
for scale in (1, 2):
    matrix = np.zeros((8, 8))
    for row, neighbors in enumerate(entries):
        for column, value in neighbors.items():
            matrix[row, column] = value * scale
    active = matrix > 0
    np.fill_diagonal(active, False)
    cardinality = active.sum(axis=1)
    asymmetric = sum(
        row not in entries[column] or value != entries[column][row]
        for row, neighbors in enumerate(entries)
        for column, value in neighbors.items()
    )
    print("scale", scale)
    print("statistics", [matrix.sum(), ((matrix + matrix.T) ** 2).sum() / 2,
                         ((matrix.sum(axis=1) + matrix.sum(axis=0)) ** 2).sum()])
    print("counts", [sum(map(len, entries)), asymmetric, sum(cardinality == 0),
                     min(cardinality), max(cardinality), sum(not row for row in entries)])
    print("cardinality", cardinality.tolist())
