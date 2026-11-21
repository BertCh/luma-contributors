# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-weights/audit-weights-a-contiguity.oracle.py

Pairwise intersections of point/undirected-edge sets; independent of key grouping/CSR scans.
The last feature has an empty polygon group. Feature 6 is a geometric island.
"""

origins = [(0, 0), (1, 0), (0, 1), (1, 1), (3, 0), (4, 0), (10, 10)]
for scale, shift in [(1, 0), (8, 1048576)]:
    rings = [[(shift + scale * (x + dx), -shift + scale * (y + dy))
              for dx, dy in [(0, 0), (1, 0), (1, 1), (0, 1)]] for x, y in origins] + [[]]
    for criterion in ["queen", "rook"]:
        keys = [set(ring) if criterion == "queen" else
                {frozenset((ring[index], ring[(index + 1) % len(ring)]))
                 for index in range(len(ring))} for ring in rings]
        rows = [[column for column in range(8) if row != column and keys[row] & keys[column]]
                for row in range(8)]
        offsets = [0]
        for row in rows:
            offsets.append(offsets[-1] + len(row))
        degrees = sorted(map(len, rows))
        print({"scale": scale, "criterion": criterion, "offsets": offsets,
               "neighbors": sum(rows, []), "requiredCount": offsets[-1],
               "overflow": 0, "candidateOverflow": 0,
               "cardinality": list(map(len, rows)),
               "counts": [offsets[-1], 0, degrees.count(0), min(degrees), max(degrees), degrees.count(0)],
               "median": (degrees[3] + degrees[4]) / 2, "validity": 1})
