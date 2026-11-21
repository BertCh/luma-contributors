# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-weights/audit-weights-a-lag.oracle.py

Dense matrix multiplication is independent of CSR traversal, chunk offsets and merge-path sums.
Stored zero rows and empty rows have a zero normalized lag. Mask removes both endpoints.
"""

import numpy as np

edges = {(0, 0): 9, (0, 1): 2, (0, 3): 4, (1, 0): 3, (1, 2): 5,
         (2, 3): 6, (2, 4): 7, (3, 0): 8, (4, 5): 2, (5, 4): 3}
values = np.arange(1, 17, dtype=float).reshape(8, 2)
mask = np.array([1, 0, 2, 1, 1, 1, 1, 1], dtype=bool)
matrix = np.zeros((8, 8))
for edge, weight in edges.items():
    matrix[edge] = weight
matrix *= mask[:, None] * mask[None, :]

for name in ["masked", "shifted", "zeroWeight", "rectangular"]:
    current_matrix = matrix.copy()
    current_values = values.copy()
    if name == "shifted":
        current_values = 3 * values + 5
    if name == "zeroWeight":
        current_matrix[5, 4] = 0
    if name == "rectangular":
        # Eight focus rows and ten value rows. An additional neighbor ID 11 is out of range.
        current_matrix = np.pad(current_matrix, ((0, 0), (0, 2)))
        current_matrix[0, 9] = 2
        current_values = np.arange(1, 21, dtype=float).reshape(10, 2)
    result = current_matrix @ current_values
    sums = current_matrix.sum(axis=1)[:, None]
    normalized = np.divide(result, sums, out=np.zeros_like(result), where=sums > 0)
    print(name, "sum", result.ravel().tolist())
    print(name, "normalized", normalized.ravel().tolist())
