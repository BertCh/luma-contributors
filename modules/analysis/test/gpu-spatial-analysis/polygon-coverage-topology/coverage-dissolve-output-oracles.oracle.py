# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/polygon-coverage-topology/coverage-dissolve-output-oracles.oracle.py
from shapely import coverage_union_all
from shapely.geometry import Polygon, box

first = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)],
                [[(1, 1), (1, 3), (3, 3), (3, 1)]])
for label, polygons in [(37, [first]), (91, [box(10, 0, 12, 2)])]:
    surface = coverage_union_all(polygons)
    print({"label": label, "area": surface.area, "rings": 1 + len(surface.interiors),
           "vertices": sum(len(ring.coords) - 1 for ring in [surface.exterior, *surface.interiors])})
