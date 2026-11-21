# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/region-statistics/region-definition.oracle.py

Shapely covers() supplies the inclusive polygon predicate; NumPy supplies the
finite-value summary and fixed histogram. No production/test reference imports.
Histogram domain includes both endpoints; interior bins are left closed/right
open. Mean divisor is the number of finite selected values, not selected rows.
"""
import json
import sys
import numpy as np
from shapely.geometry import Point, Polygon

if "--grid-composition" in sys.argv:
    # /usr/bin/python3 -B modules/analysis/test/gpu-spatial-analysis/region-statistics/region-definition.oracle.py --grid-composition
    # Independent rational interval enumeration of cells touched by the query plus one neighbor
    # on each side. This checks candidate capacity, before the exact region predicate runs.
    from fractions import Fraction
    from itertools import accumulate
    coordinates = [Fraction(1, 4), Fraction(3, 4)]
    for width in [2, 4]:
        intervals = [(Fraction(column, width), Fraction(column + 1, width)) for column in range(width)]
        cells = [next(column for column, (minimum, maximum) in enumerate(intervals)
                      if minimum <= coordinate < maximum) for coordinate in coordinates]
        offsets = [0] + list(accumulate(cells.count(column) for column in range(width)))
        print("grid", width, "row cells", cells, "cellOffsets", offsets)
        for name, query in [("initial", (Fraction(-1), Fraction(2))),
                            ("recovery", (Fraction(0), Fraction(49, 100)))]:
            touched = [column for column, (minimum, maximum) in enumerate(intervals)
                       if minimum <= query[1] and maximum >= query[0]]
            padded = set(range(max(0, min(touched) - 1), min(width - 1, max(touched) + 1) + 1))
            rows = [row for row, cell in enumerate(cells) if cell in padded]
            print(name, "padded columns", sorted(padded), "candidate rows", rows,
                  "count", len(rows), "overflow at capacity 1", int(len(rows) > 1))
    raise SystemExit

positions = [(0, 0), (2, 0), (2, 2), (0, 2), (1, 1), (3, 1), (-1, 1),
             (1, 2), (1, -1), (2, 1), (4, 4), (1, 0)]
values = np.array([4, 8, 8, np.nan, -2, 20, np.inf, 1, 6, 0, 10, 2])
identities = [91, 7, 43, 500, 12, 42, 19, 3, 80, 77, 9, 101]
polygon = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])

for kind in ["square", "radius", "empty"]:
    selected = [row for row, position in enumerate(positions)
                if (polygon.covers(Point(position)) if kind == "square" else
                    Point(position).distance(Point(1, 1)) <= 1 if kind == "radius" else False)]
    finite = values[selected]
    finite = finite[np.isfinite(finite)]
    histogram = np.histogram(finite, bins=[0, 2, 4, 6, 8])[0].tolist()
    summary = [len(selected), len(finite), float(finite.sum()),
               float(finite.mean()) if len(finite) else 0,
               float(finite.min()) if len(finite) else 0,
               float(finite.max()) if len(finite) else 0,
               len(finite) - sum(histogram), int(len(selected) > 5)] + histogram
    packed = np.zeros(12, dtype="<u4")
    packed[:2] = summary[:2]
    packed[6:] = summary[6:]
    packed[2:6] = np.asarray(summary[2:6], dtype="<f4").view("<u4")
    print(kind, json.dumps(dict(summary=summary, words=packed.tolist(),
                                mask=[int(row in selected) for row in range(12)],
                                ids=[identities[row] for row in selected[:5]],
                                count=min(len(selected), 5), requiredCount=len(selected),
                                overflow=int(len(selected) > 5), candidateOverflow=0)))
