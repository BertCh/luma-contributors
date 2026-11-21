# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-weights/audit-weights-a-helpers.oracle.py"""

import numpy as np

counts = np.array([0, 1, 16777215, 16777216, 16777217, 16777219, 2147483647, 4294967295], dtype=np.uint32)
print("float32", counts.astype(np.float32).tolist())

# Independent row predicates: offsets, range, order, self, finite/nonnegative weight.
offsets = [0, 3, 4, 5, 6, 6, 8, 9, 9]
neighbors = [0, 1, 1, 2, 9, 3, 2, 1, 0]
weights = [1, 1, np.nan, 0, -1, 1, 1, 1, np.inf]
flags = []
for row in range(8):
    begin, end = offsets[row:row + 2]
    columns = neighbors[begin:end]
    values = weights[begin:end]
    predicates = [
        (row == 0 and begin != 0) or begin > end or (row == 7 and end > len(neighbors)),
        any(column >= 8 for column in columns),
        any(first >= second for first, second in zip(columns, columns[1:])),
        row in columns,
        any(not np.isfinite(value) or value < 0 for value in values)
    ]
    flags.append(sum(bit for bit, violated in zip([1, 2, 4, 8, 16], predicates) if violated))
print("flags", flags, "invalidCount", sum(word != 0 for word in flags))

# Whole and chunked transforms: direct dense rows, with a zero row and two empty rows.
offsets = [0, 3, 5, 7, 8, 9, 10, 10, 10]
weights = np.array([9, 2, 4, 3, 5, 6, 7, 8, 2, 3], dtype=float)
neighbors = [0, 1, 3, 0, 2, 3, 4, 0, 5, 4]
distances = np.array([0, 10, 20, 10, 12, 14, 16, 20, 10, 10], dtype=float)
matrix = np.zeros((8, 8))
for row in range(8):
    for slot in range(offsets[row], offsets[row + 1]):
        matrix[row, neighbors[slot]] = weights[slot]
row_sums = matrix.sum(axis=1)
norms = np.linalg.norm(matrix, axis=1)
standard = np.divide(matrix, norms[:, None], out=np.zeros_like(matrix), where=norms[:, None] != 0)
for operation in ["row", "binary", "double", "doubleRows", "variance", "average-reciprocal"]:
    result = {"row": np.divide(matrix, row_sums[:, None], out=np.zeros_like(matrix), where=row_sums[:, None] != 0),
              "binary": (matrix > 0).astype(float), "double": matrix / matrix.sum(),
              "doubleRows": matrix * 8 / matrix.sum(), "variance": standard * 8 / standard.sum(),
              "average-reciprocal": (matrix + matrix.T) / 2}[operation]
    print(operation, [result[row, neighbors[slot]] for row in range(8) for slot in range(offsets[row], offsets[row + 1])])
for adaptive in [False, True]:
    bandwidths = np.full(10, 20.)
    if adaptive:
        for row in range(8):
            begin, end = offsets[row:row + 2]
            if begin < end:
                bandwidths[begin:end] = distances[begin:end].max() * 1.0000001
    relative = distances / bandwidths
    profiles = {"gaussian": np.exp(-relative ** 2 / 2) / np.sqrt(2 * np.pi),
                "triangular": np.maximum(1 - relative, 0),
                "epanechnikov": .75 * np.maximum(1 - relative ** 2, 0),
                "bisquare": .9375 * np.maximum(1 - relative ** 2, 0) ** 2,
                "uniform": np.where(relative <= 1, .5, 0)}
    for name, values in profiles.items():
        print(("adaptive-" if adaptive else "fixed-") + name, values.tolist())

# Nearest original empty rows: simultaneous exact squared distances, ties to the lowest row.
positions = np.array([[0, 0], [2, 0], [1, 0], [5, 0], [5, 0], [10, 0], [np.nan, 0], [30, 0]])
original = {(0, 1): 2, (1, 0): 3}
attachments = []
for row, point in enumerate(positions):
    candidates = [(np.dot(point - other, point - other), column) for column, other in enumerate(positions)
                  if column != row and np.isfinite(point).all() and np.isfinite(other).all()]
    attachments.append(min(candidates)[1] if candidates and row not in [0, 1] else 4294967295)
print("attachments", attachments)
for reciprocal in [False, True]:
    edges = dict(original)
    for row, column in enumerate(attachments):
        if column != 4294967295:
            edges.setdefault((row, column), 1)
            if reciprocal:
                edges.setdefault((column, row), 1)
    ordered = sorted(edges)
    print("islands", reciprocal, "offsets", [0] + np.cumsum([sum(first == row for first, _ in ordered) for row in range(8)]).tolist(),
          "neighbors", [column for _, column in ordered], "weights", [edges[edge] for edge in ordered],
          "added", [int(edge not in original) for edge in ordered])

# 4x2 masked lattice, planar cell dimensions (3,4): diagonal distance is exactly 5.
mask = [1, 1, 0, 1, 1, 1, 0, 1]
for criterion in ["queen", "rook"]:
    pairs = [(row, column) for row in range(8) for column in range(8) if row != column and mask[row] and mask[column]
             and (max(abs(row % 4 - column % 4), abs(row // 4 - column // 4)) <= 1 if criterion == "queen"
                  else abs(row % 4 - column % 4) + abs(row // 4 - column // 4) <= 1)]
    print("lattice", criterion, "offsets", [0] + np.cumsum([sum(first == row for first, _ in pairs) for row in range(8)]).tolist(),
          "neighbors", [column for _, column in pairs],
          "distances", [np.hypot((row % 4 - column % 4) * 3, (row // 4 - column // 4) * 4).item() for row, column in pairs])

# Two-column spatial lag, dense multiplication with a masked row/column and structural self slot.
values = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]])
mask = np.array([1, 0, 2, 1, 1, 1, 1, 1], dtype=bool)
masked = matrix * mask[:, None] * mask[None, :]
for normalize in [False, True]:
    result = masked @ values
    if normalize:
        denominator = masked.sum(axis=1)[:, None]
        result = np.divide(result, denominator, out=np.zeros_like(result), where=denominator != 0)
    print("lag", normalize, result.ravel().tolist())

# Shapely uses filled polygons, so overlap and exact contact have zero separation.
from shapely.geometry import box
geometries = [box(x, y, x + 1, y + 1) for x, y in [(0, 0), (1, 0), (2.25, 0), (10, 0), (4, 4), (4.5, 4)]]
pairs = [(row, column) for row in range(6) for column in range(6)
         if row != column and geometries[row].distance(geometries[column]) <= .25]
distances = [geometries[row].distance(geometries[column]) for row, column in pairs]
print("fuzzy", "offsets", [0] + np.cumsum([sum(first == row for first, _ in pairs) for row in range(8)]).tolist(),
      "neighbors", [column for _, column in pairs], "distances", distances, "added", [int(distance > 0) for distance in distances],
      "featureStatus", [0, 0, 0, 0, 0, 0, 1, 2])
