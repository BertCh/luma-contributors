# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-join/multipart-point-review.oracle.py
from fractions import Fraction
import struct
from shapely.geometry import LineString, MultiPoint, Polygon

squares = [Polygon([(0,0),(2,0),(2,2),(0,2)]), Polygon([(2,0),(4,0),(4,2),(2,2)])]
edge = LineString([(2,.5),(2,1.5)])
lengths = [edge.intersection(polygon).length for polygon in squares]
assert lengths == [1,1]
assert [3 * value for value in lengths] == [3,3]
hole = Polygon([(0,0),(4,0),(4,4),(0,4)], [[(1,1),(3,1),(3,3),(1,3)]])
assert LineString([(1,1.5),(1,2.5)]).intersection(hole).length == 1
assert MultiPoint([(29,29),(-2,5)]).distance(LineString([(0,0),(100,100)])) == 0
features = [MultiPoint([(.5,.5),(3,.5)]), MultiPoint([(3,1)]), MultiPoint([(10,10)])]
assert [sum(polygon.intersects(feature) for polygon in squares) for feature in features] == [2,1,0]
total = Fraction(32768) + Fraction(1,1024) - Fraction(32768)
mean = struct.unpack('f', struct.pack('f', float(total / 3)))[0]
assert float(total) == .0009765625 and mean == .00032552084303461015
assert 7 + 11 == 18
print('R1/R2/R3/R4/R8 independent geometry/binary expectations passed')
