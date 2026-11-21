# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-weights/weights-transpose-audit.oracle.py

SciPy sparse transpose preserves explicitly stored zero weights. Symmetry
counts positions in the union of listed W and W.T, with absolute tolerance
inclusive at equality. Neighbors outside the declared columns are dropped.
"""
import numpy as np
from scipy.sparse import csr_matrix

offsets = [0, 3, 5, 8, 9, 10, 11, 12, 13]
neighbors = [0, 1, 2, 0, 2, 0, 1, 3, 2, 5, 4, 7, 9]
weights = [0.5, 2, 1, 1, 3, 1, 3, 4, 0, 2, 2, 0, 99]
distances = np.arange(13) + 10.5
for columns in (8, 5):
    rows = np.repeat(np.arange(8), np.diff(offsets))
    kept = np.array(neighbors) < columns
    row_indices, column_indices = rows[kept], np.array(neighbors)[kept]
    matrix = csr_matrix((np.array(weights)[kept], (row_indices, column_indices)), shape=(8, columns))
    distance_matrix = csr_matrix((distances[kept], (row_indices, column_indices)), shape=(8, columns))
    transpose = matrix.T.tocsr()
    transpose.sort_indices()
    print("columns", columns)
    print("offsets", transpose.indptr.tolist())
    print("neighbors", transpose.indices.tolist())
    print("weights", transpose.data.tolist())
    print("distances", distance_matrix.T.tocsr().data.tolist())
    if columns == 8:
        present = set(zip(row_indices.tolist(), column_indices.tolist()))
        union = present | {(column, row) for row, column in present}
        for tolerance in (0, 1):
            count = sum((row, column) not in present or (column, row) not in present
                        or abs(matrix[row, column] - matrix[column, row]) > tolerance
                        for row, column in union)
            print("asymmetricSlots", tolerance, count)
