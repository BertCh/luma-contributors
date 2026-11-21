# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Command: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-join/multipart-review.oracle.py
# Independent GEOS/Shapely expectations for R1-R4. All coordinates are exactly float32.
import json
import shapely
from shapely import wkt

cases = [
    ('R1', 'MULTIPOINT ((200 200), (-2.25 11.5))', 'LINESTRING (0 0, 5 12)', [6.5, 6.49]),
    ('R2', 'MULTIPOINT ((1000000.125 1000000), (1000002 1000002))', 'LINESTRING (1000000 1000000, 1000001 1000000.25)', [0, 0.02, 0.04]),
    ('R3', 'MULTILINESTRING ((-1 0, 1 0), (-1 0.5, -0.5 0.5))', 'POLYGON ((-2 -2, 2 -2, 2 2, -2 2, -2 -2), (0 -1, 0 1, 0.000000003725290298461914 1, 0.000000003725290298461914 -1, 0 -1))', []),
    ('R4', 'POLYGON ((1000000 1000000, 1000000.0625 1000000, 1000000.0625 1000000.0625, 1000000 1000000.0625, 1000000 1000000))', 'POLYGON ((1000000 1000000, 1000000.0625 1000000, 1000000.0625 1000000.0625, 1000000 1000000.0625, 1000000 1000000))', []),
]
for name, first, second, thresholds in cases:
    left, right = wkt.loads(first), wkt.loads(second)
    print(json.dumps(dict(name=name, matrix=shapely.relate(left, right), distance=shapely.distance(left, right), intersects=bool(shapely.intersects(left, right)), within=bool(shapely.within(left, right)), crosses=bool(shapely.crosses(left, right)), equals=bool(shapely.equals(left, right)), dwithin=[bool(shapely.dwithin(left, right, threshold)) for threshold in thresholds])))
