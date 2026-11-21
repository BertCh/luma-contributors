# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-join/predicate-output-oracles.oracle.py"""
from shapely.geometry import Point, box

points = [Point(1, 1), Point(2, 1), Point(5, 5), Point(6, 6)]
squares = [box(0, 0, 2, 2), box(1, 0, 3, 2)]
pairs = [(row, column) for row, point in enumerate(points) for column, square in enumerate(squares) if point.intersects(square)]
candidates = [(row, column) for row, point in enumerate(points) for column, square in enumerate(squares) if square.envelope.covers(point)]
print('pairs:', pairs, 'candidateCount:', len(candidates))
print('anti rows:', [row for row in range(4) if not any(left == row for left, _ in pairs)])
