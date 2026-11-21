# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-join/join-review-response.oracle.py
from fractions import Fraction
from math import hypot
import struct
from shapely.geometry import Point, LineString, MultiPoint, Polygon

def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]

def projection(start, end):
    x, y = map(Fraction, start)
    dx, dy = Fraction(end[0]) - x, Fraction(end[1]) - y
    determinant = (-x) * dy - (-y) * dx
    squared_length = dx * dx + dy * dy
    return float(-dy * determinant / squared_length), float(dx * determinant / squared_length), abs(float(determinant)) / hypot(float(dx), float(dy))

for road in [[(-8191, -8190), (1, 1)], [(1, 1), (-8191, -8190)], [(1, 1), (-65535, -65534)], [(-65535, -65534), (1, 1)]]:
    print('foot x,y,distance,f32:', projection(*road), f32(projection(*road)[2]))
    print('Shapely distance:', Point(0, 0).distance(LineString(road)))
end = f32(.01)
distance = projection((-8191, -8190), (end, end))[2]
threshold = 8.633243737676821e-7
print('R7 distance, radius, result:', distance, threshold, distance <= threshold)
square = Polygon([(0,0), (3,0), (3,3), (0,3)])
print('R3 logical counts:', sum(square.intersects(feature) for feature in [MultiPoint([]), MultiPoint([(1,1),(2,2)])]))
print('R5 exact parallel lines have cross-product direction zero and positive vertical offset 2^98.')
