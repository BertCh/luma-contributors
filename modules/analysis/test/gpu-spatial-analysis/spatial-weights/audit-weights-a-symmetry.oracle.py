# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-weights/audit-weights-a-symmetry.oracle.py

Independent support-union definition, with explicit zero distinct from absent support.
No sorting, scans, chunk traversal or TypeScript implementation is used.
"""

edges = {(0, 1): 2, (1, 0): 2.5, (2, 0): 0, (3, 4): 3,
         (4, 3): 1, (5, 5): 5, (6, 3): 1, (3, 6): 1}
support = set(edges) | {(column, row) for row, column in edges}
for tolerance in [0, 0.5]:
    mismatches = sum(
        (row, column) not in edges or (column, row) not in edges
        or abs(edges[row, column] - edges[column, row]) > tolerance
        for row, column in support
    )
    print({"tolerance": tolerance, "asymmetricSlots": mismatches})
