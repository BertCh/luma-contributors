# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# /usr/bin/python3 -B modules/analysis/test/gpu-spatial-analysis/cell-topology/s2-covering-output-oracles.oracle.py
from shapely.geometry import Polygon

# Independent gnomonic cell polygons on face zero. S2's published quadratic s->u
# transform; the tiny triangle is strictly inside this face. An exterior cover of
# its complement excludes only leaves completely inside the triangle.
def quadratic(value):
    return (4 * value * value - 1) / 3 if value >= .5 else (1 - 4 * (1 - value) ** 2) / 3

triangle = Polygon([(-.1, -.1), (.1, -.1), (0, .1)])
width = 128
coordinates = [quadratic(index / width) for index in range(width + 1)]
excluded = 0
for column in range(width):
    for row in range(width):
        left, right = coordinates[column:column + 2]
        bottom, top = coordinates[row:row + 2]
        cell = Polygon([(left, bottom), (right, bottom), (right, top), (left, top)])
        excluded += triangle.contains(cell)
print('strictly excluded leaves:', excluded)
print('complement exterior level 7:', 6 * width * width - excluded)
print('whole-globe level 2:', 6 * 4 ** 2)
