# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-join/join-review2.oracle.py
from fractions import Fraction as F
from math import sqrt
import struct
from shapely.geometry import Point, LineString, MultiPoint, Polygon

def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]

def projection(point, start, end):
    point, start, end = [tuple(map(F, coordinates)) for coordinates in (point, start, end)]
    direction = [end[index] - start[index] for index in range(2)]
    offset = [point[index] - start[index] for index in range(2)]
    length_squared = sum(value ** 2 for value in direction)
    parameter = max(F(0), min(F(1), sum(offset[index] * direction[index] for index in range(2)) / length_squared))
    foot = [start[index] + parameter * direction[index] for index in range(2)]
    distance_squared = sum((point[index] - foot[index]) ** 2 for index in range(2))
    return foot, distance_squared

previous_13 = struct.unpack('f', struct.pack('I', struct.unpack('I', struct.pack('f', 13))[0] - 1))[0]
next_13 = struct.unpack('f', struct.pack('I', struct.unpack('I', struct.pack('f', 13))[0] + 1))[0]
publication_tolerance = 2 ** -19
assert 13 - previous_13 == next_13 - 13 == 2 ** -20
assert publication_tolerance == 2 * (next_13 - 13)
print('R1 f32 publication: one ULP', next_13 - 13, 'two-ULP tolerance', publication_tolerance)
for reverse in (False, True):
    road = [(0, 0), (895, 0)][::(-1 if reverse else 1)]
    foot, squared = projection((1, 13), *road)
    assert squared == 169 and squared > F(previous_13) ** 2
    print('R1', reverse, 'foot', list(map(float, foot)), 'hit at13, miss at', previous_13)
    road = [(-5, -10), (f32(1.1), 2.5)][::(-1 if reverse else 1)]
    foot, squared = projection((0, 0), *road)
    distance = sqrt(squared)
    assert abs(distance - 0.10784390321559371) < 3e-17
    print('R2', reverse, 'distance', distance, 'f32', f32(distance), 'Shapely', Point(0, 0).distance(LineString(road)), 'row0 wins identical copies')
    foot, squared = projection((1, 8192), *[(0, 2**-13), (2, 2**-13)][::(-1 if reverse else 1)])
    assert foot == [1, F(1, 8192)]
    print('R4', reverse, 'foot', list(map(float, foot)), 'f32 distance', f32(sqrt(squared)))
    foot, squared = projection((-1, 1), *[(-2**54, 0), (0, 0)][::(-1 if reverse else 1)])
    assert foot == [-1, 0] and squared == 1
    print('R7', reverse, 'foot', list(map(float, foot)), 'distance', sqrt(squared), 'inclusive hit')
    road = [(-65536, -65536), (2**-10, 2**-9)][::(-1 if reverse else 1)]
    foot, squared = projection((0, 0), *road)
    assert squared > F(2**-11)**2
    print('R9', reverse, 'distance', sqrt(squared), 'radius', 2**-11, 'miss')

square = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
assert sum(square.intersects(feature) for feature in [MultiPoint([]), MultiPoint([(1, 1), (2, 2)])]) == 1
print('R3 one empty logical feature and one matched multipart feature: count1')
large, displacement, small = F(2**120), F(2**98), F(1, 2**100)
left = [(large/4, large/4 + displacement), (large/2, large/2 + displacement)]
right = [(small, small), (large, large)]
assert left[0][1] - left[0][0] == left[1][1] - left[1][0] == displacement > 0
assert all(y - x == 0 for x, y in right)
print('R5 parallel disjoint lines: exact count0; >200-bit span requires diagnostic1 on provisional GPU pair')
