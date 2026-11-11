# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Regenerate with /usr/bin/python3; GEOS intersection is independent of the twin's splitter."""

import json
from pathlib import Path

import numpy
from shapely.geometry import LineString, Point, Polygon


def round_input(values):
    return numpy.asarray(values, dtype=numpy.float32).astype(float).tolist()


outer = round_input([[0, 0], [12, 0], [12, 8], [8, 8], [8, 3], [4, 3], [4, 8], [0, 8]])
hole = round_input([[1, 1], [3, 1], [3, 2], [1, 2]])
polygon = Polygon(outer, [hole])
tracks = [
    [[-2, 1.5], [14, 1.5]],
    [[-2, 5], [14, 5]],
    [[2, -2], [2, 10]],
    [[-1.25, -3.5], [13.25, 9.75]],
    [[13.25, 9.75], [-1.25, -3.5]],
    [[1.5, 1.25], [2.5, 1.75]],
    [[0.25, 0.25], [0.75, 0.75]],
]
expected = []
rounded_tracks = []
for track_index, track in enumerate(tracks):
    track = round_input(track)
    rounded_tracks.append(track)
    line = LineString(track)
    intersection = line.intersection(polygon)
    pieces = list(intersection.geoms) if hasattr(intersection, "geoms") else [intersection]
    intervals = []
    for piece in pieces:
        if piece.is_empty:
            continue
        endpoints = [piece.coords[0], piece.coords[-1]]
        fractions = sorted(line.project(Point(point), normalized=True) for point in endpoints)
        begin, end = fractions
        first = line.interpolate(begin, normalized=True)
        last = line.interpolate(end, normalized=True)
        intervals.append({"interval": [16 * begin, 16 * end], "positions": [first.x, first.y, last.x, last.y]})
    for interval in sorted(intervals, key=lambda row: row["interval"]):
        expected.append({"track": track_index, **interval})

fixture = {"oracle": "Shapely/GEOS LineString.intersection(Polygon), binary32 input coordinates", "rings": [outer, hole], "tracks": rounded_tracks, "expected": expected}
Path(__file__).with_name("trajectory-restriction-shapely-fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
