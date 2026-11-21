# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/trajectory-restriction/restriction-output-oracles.oracle.py"""
from shapely.geometry import LineString, Point, Polygon

polygon = Polygon([(0,-2),(8,-2),(8,2),(0,2)], [[(2,-1),(6,-1),(6,1),(2,1)]])
for coordinates, duration in [([(-2,0),(10,0)],12), ([(0.5,1.5),(1.5,1.5)],3)]:
    line = LineString(coordinates)
    result = line.intersection(polygon)
    pieces = list(result.geoms) if hasattr(result, 'geoms') else [result]
    for piece in pieces:
        endpoints = [piece.coords[0], piece.coords[-1]]
        print([duration * line.project(Point(point), normalized=True) for point in endpoints], endpoints)
