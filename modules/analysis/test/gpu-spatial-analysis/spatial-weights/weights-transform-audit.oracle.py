# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-weights/weights-transform-audit.oracle.py

Definitions: libpysal W transformations R, B, D and V (Tiefelsdorf et al.).
The norm is evaluated in float64. Scaling the input by a positive constant
does not change R, B, D or V. Empty and all-zero rows stay zero.
"""
import numpy as np

offsets = np.array([0, 2, 4, 7, 8, 9, 10, 11, 11])
neighbors = np.array([1, 2, 0, 2, 0, 1, 3, 2, 5, 4, 7])
weights = np.array([3, 4, 1, 2, 0, 2, 1, 0, 2, 2, 0], dtype=float)
matrix = np.zeros((8, 8))
for row in range(8):
    matrix[row, neighbors[offsets[row]:offsets[row + 1]]] = weights[offsets[row]:offsets[row + 1]]

def slots(matrix):
    return [matrix[row, column] for row in range(8)
            for column in neighbors[offsets[row]:offsets[row + 1]]]

row_sums = matrix.sum(axis=1, keepdims=True)
norms = np.linalg.norm(matrix, axis=1, keepdims=True)
unit = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
outputs = {
    "row": np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0),
    "binary": (matrix > 0).astype(float),
    "doubleOne": matrix / matrix.sum(),
    "doubleRows": matrix * 8 / matrix.sum(),
    "variance": unit * 8 / unit.sum(),
    "average-reciprocal": (matrix + matrix.T) / 2,
}
for name, result in outputs.items():
    print(name, slots(result))

z = np.array([0, 0.25, 0.5, 1, 1.5, 2, -0.5, np.nan, np.inf, -np.inf, 0])
profiles = {
    "triangular": np.maximum(1 - z, 0),
    "epanechnikov": 0.75 * np.maximum(1 - z**2, 0),
    "bisquare": 0.9375 * np.maximum(1 - z**2, 0)**2,
    "uniform": np.where(z <= 1, 0.5, 0),
    "gaussian": np.exp(-z**2 / 2) / np.sqrt(2 * np.pi),
}
for name, values in profiles.items():
    values[~np.isfinite(z) | (z < 0)] = 0
    print(name, values.tolist())

# Mean of the actually stored binary32 words, rounded once to binary32.
for pair in [[2e38, 2e38], [2e38, 3e38], [2e38, 0.0]]:
    stored = np.asarray(pair, dtype=np.float32).astype(np.float64)
    print('large average', stored.tolist(), float(np.float32(stored.mean())))
print('finite scale boundary', float(np.float32(8 / float(np.float32(1.5 * 2**-125)))))
