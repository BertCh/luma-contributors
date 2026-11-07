# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Direct Shapely 2.1.2 references; run in the pinned geometry oracle environment."""
import json
import math
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiPoint
assert shapely.__version__ == '2.1.2'

def layout(geometry):
    polygons = list(geometry.geoms) if isinstance(geometry, MultiPolygon) else [geometry]
    positions, ring_offsets, polygon_offsets = [], [0], [0]
    for polygon in polygons:
        for ring in [polygon.exterior, *polygon.interiors]:
            positions.extend(np.asarray(ring.coords, dtype=np.float32).flatten().tolist())
            ring_offsets.append(len(positions) // 2)
        polygon_offsets.append(len(ring_offsets) - 1)
    return dict(positions=positions, ringOffsets=ring_offsets, polygonOffsets=polygon_offsets, featureOffsets=[0, len(polygons)])

polygons = [
    Polygon([(4, 4), (0, 4), (0, 0), (4, 0)]),
    MultiPolygon([
        Polygon([(0, 0), (9, 0), (9, 9), (0, 9)], [[(2, 2), (2, 3), (3, 3), (3, 2)], [(5, 5), (6, 5), (6, 6), (5, 6)]]),
        Polygon([(20, 0), (25, 0), (25, 4), (20, 4)])
    ]),
    Polygon([(0, 0), (6, 0), (0, 8)]),
    Polygon([(0, 0), (8, 0), (8, 2), (2, 2), (2, 8), (0, 8)])
]
cases = []
for polygon in polygons:
    circle = shapely.maximum_inscribed_circle(polygon, tolerance=0.00001)
    cases.append(dict(input=layout(polygon), normalized=layout(shapely.normalize(polygon)), circle=list(circle.coords[0]) + [circle.length]))
line = LineString([(0, 0), (6, 0)])
references = MultiPoint([(4, 0.1), (2, 0), (2, 0), (3, 5)])
ratio_points = [(0, 0), (0, 3), (1, 1), (4, 0), (5, 3)]
ratio_cases = [dict(positions=[value for point in ratio_points for value in point], ratio=ratio, expectedArea=shapely.concave_hull(MultiPoint(ratio_points), ratio=ratio).area) for ratio in [0.1, 1.0]]
hole_points = np.asarray([(10 * math.cos(index * math.tau / 24), 10 * math.sin(index * math.tau / 24)) for index in range(24)] + [(4 * math.cos(index * math.tau / 8 + 0.1), 4 * math.sin(index * math.tau / 8 + 0.1)) for index in range(8)], dtype=np.float32).tolist()
hole_reference = shapely.concave_hull(MultiPoint(hole_points), ratio=0.2, allow_holes=True)
ratio_cases.append(dict(positions=[value for point in hole_points for value in point], ratio=0.2, allowHoles=True, expectedArea=hole_reference.area, expectedHoleCount=len(hole_reference.interiors)))
result = dict(ratioHulls=ratio_cases, generator='geometry-refinement.py', versions=dict(shapely=shapely.__version__, geos=shapely.geos_version_string, numpy=np.__version__), polygons=cases, snap=dict(input=list(line.coords), references=[list(point.coords[0]) for point in references.geoms], tolerance=0.2, expected=list(shapely.snap(line, references, tolerance=0.2).coords)))
path = Path(__file__).parent / 'fixtures' / 'geometry-refinement.json'
path.write_text(json.dumps(result, indent=2) + '\n')
