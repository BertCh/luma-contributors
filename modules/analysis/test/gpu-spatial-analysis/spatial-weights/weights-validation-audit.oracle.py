# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-weights/weights-validation-audit.oracle.py

Definition: the documented CSR invariants, evaluated with NumPy predicates
on each bounded slice. Bit values are literal, independent of TypeScript.
"""
import numpy as np

offsets = [1, 3, 5, 7, 9, 9, 11, 10, 15]
neighbors = np.array([6, 0, 1, 4, 4, 9, 0, 3, 1, 6, 7, 0, 8, 7])
weights = np.array([1, 1, np.nan, 1, np.inf, -1, 1, 1, 1, 0, 2, 1, 1, np.nan])
flags = []
for row, (begin, end) in enumerate(zip(offsets, offsets[1:])):
    columns = neighbors[begin:min(end, len(neighbors))]
    values = weights[begin:min(end, len(neighbors))]
    flag = int((row == 0 and begin != 0) or begin > end
               or (row == 7 and end > len(neighbors)))
    flag |= 2 * int(np.any(columns >= 8))
    flag |= 4 * int(np.any(np.diff(columns) <= 0))
    flag |= 8 * int(np.any(columns == row))
    flag |= 16 * int(np.any(~np.isfinite(values) | (values < 0)))
    flags.append(flag)
print("flags", flags)
print("invalidCount", np.count_nonzero(flags))
