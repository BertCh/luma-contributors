# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-weights/audit-weights-a-algebra.oracle.py

Dense support masks (stored zero is supported) and SciPy shortest paths, independent of
the contributor's sorted merges, frontier generation, scans and bounded CSR emission.
"""

import json
import numpy as np
from scipy.sparse.csgraph import shortest_path

left = {(0, 0): 9, (0, 1): 2, (0, 3): 4, (1, 0): 3, (1, 2): 5,
        (2, 3): 6, (2, 4): 7, (3, 0): 8, (4, 5): 2, (5, 4): 3}
right = {(0, 1): 7, (0, 2): 11, (1, 2): 13, (2, 4): 17,
         (3, 0): 19, (3, 5): 23, (5, 6): 29, (6, 5): 31}
left_distances = dict(zip(left, [0, 10, 20, 10, 12, 14, 16, 20, 10, 10]))
right_distances = dict(zip(right, [99, 98, 97, 96, 95, 94, 93, 92]))


def emit(name, edges, distances=None, rows=8):
    ordered = sorted(edges)
    counts = [sum(row == first for first, _ in ordered) for row in range(rows)]
    result = {"name": name, "offsets": [0] + np.cumsum(counts).tolist(),
              "neighbors": [column for _, column in ordered],
              "weights": [edges[edge] for edge in ordered],
              "requiredCount": len(ordered), "overflow": 0, "candidateOverflow": 0}
    if distances is not None:
        result["distances"] = [distances[edge] for edge in ordered]
    print(json.dumps(result))


for operation, support in [("union", left.keys() | right.keys()),
                           ("intersection", left.keys() & right.keys()),
                           ("difference", left.keys() - right.keys()),
                           ("symmetric-difference", left.keys() ^ right.keys())]:
    emit(operation, {edge: left.get(edge, 0) + right.get(edge, 0) for edge in support},
         {edge: left_distances[edge] if edge in left else right_distances[edge] for edge in support})

adjacency = np.zeros((8, 8))
for (row, column) in left:
    if row != column:
        adjacency[row, column] = 1
paths = shortest_path(adjacency, directed=True, unweighted=True)
for cumulative in [False, True]:
    emit("cumulative2" if cumulative else "exact2",
         {(row, column): 1 for row in range(8) for column in range(8)
          if row != column and (paths[row, column] <= 2 if cumulative else paths[row, column] == 2)})
mask = [1, 0, 2, 1, 1, 1, 1, 1]
emit("subgraph", {edge: value for edge, value in left.items() if mask[edge[0]] and mask[edge[1]]}, left_distances)
self_edges = left | {(row, row): (row + 1) / 2 for row in range(8)}
emit("selfWeight", self_edges, left_distances | {(row, row): 0 for row in range(8)})
groups = [0, 1, 0, 1, 2, 2, 99, 3]
emit("block", {(row, column): 1 for row in range(8) for column in range(8)
               if row != column and groups[row] < 4 and groups[row] == groups[column]})

# Space-time: two periods, contemporary 2W and previous-period 0.5(W with unit diagonal).
matrix = np.zeros((8, 8))
support = np.zeros((8, 8), dtype=bool)
for edge, value in left.items():
    matrix[edge] = value
    support[edge] = True
with_identity = matrix.copy()
np.fill_diagonal(with_identity, 1)
temporal = np.array([[0, 0], [0.5, 0]])
cube = np.kron(np.eye(2) * 2, matrix) + np.kron(temporal, with_identity)
cube_support = np.kron(np.eye(2, dtype=bool), support) | np.kron(temporal != 0, support | np.eye(8, dtype=bool))
for layout in ["period-major", "cell-major"]:
    permutation = list(range(16)) if layout == "period-major" else [period * 8 + cell for cell in range(8) for period in range(2)]
    permuted = cube[np.ix_(permutation, permutation)]
    permuted_support = cube_support[np.ix_(permutation, permutation)]
    emit(layout, {(row, column): permuted[row, column] for row in range(16) for column in range(16)
                  if permuted_support[row, column]}, rows=16)
