# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-join/join-construction-review.oracle.py
# Independent checks for the original findings and the construction regression's literal values.
from math import hypot
from shapely.geometry import LineString, MultiLineString, Point, Polygon

polygon = Polygon([(0, 0), (2, 6), (3, 9), (0, 9)])
for interior in [True, False]:
    parts = [[(0, 0), (3, 9)]] + ([[(0, 5), (1, 6)]] if interior else [])
    line = MultiLineString(parts)
    print("X1", interior, "within", line.within(polygon), "coveredBy", polygon.covers(line))

for reverse in [False, True]:
    road = [(-8191, -8190), (1, 1)]
    if reverse:
        road.reverse()
    print("X2", reverse, "distance", Point(0, 0).distance(LineString(road)))
    endpoint = [(-1e20, 0), (0, 0)]
    if reverse:
        endpoint.reverse()
    print("X6", reverse, "distance", Point(1, 0).distance(LineString(endpoint)))

print("predicate construction", Point(1, 2).relate(Point(1, 2)))
print("nearest construction distances", [hypot(*point) for point in [(6, 8), (3, 4)]])

# Segment rows are start vertices 0 and 2; feature row 0 is empty, feature row 1 owns both.
# Same-part pairs overlap (kind 4); opposite-part pairs cross (kind 1).
segments = [LineString([(0, 0), (2, 2)]), LineString([(0, 2), (2, 0)])]
for left, first in enumerate(segments):
    for right, second in enumerate(segments):
        print("multipart segments", left * 2, right * 2, first.intersection(second).wkt)
