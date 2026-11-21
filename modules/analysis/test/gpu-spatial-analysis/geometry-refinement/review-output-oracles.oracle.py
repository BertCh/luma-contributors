# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/geometry-refinement/review-output-oracles.oracle.py
from shapely import MultiPolygon, Polygon, normalize

# Unequal closed-ring lengths: copying input offsets is a visible wrong implementation.
triangle = Polygon([(0,0),(2,0),(0,2)])
quadrilateral = Polygon([(10,0),(14,0),(14,4),(10,4)])
result = normalize(MultiPolygon([triangle,quadrilateral]))
coordinates = [list(polygon.exterior.coords) for polygon in result.geoms]
offsets = [0, len(coordinates[0]), sum(map(len,coordinates))]
print('R7',coordinates,'ringOffsets',offsets)
assert offsets == [0,5,9]
assert coordinates == [[(10.,0.),(10.,4.),(14.,4.),(14.,0.),(10.,0.)],[(0.,0.),(0.,2.),(2.,0.),(0.,0.)]]
