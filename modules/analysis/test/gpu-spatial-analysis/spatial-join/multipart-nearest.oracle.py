# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run from this directory: /usr/bin/python3 multipart-nearest.oracle.py > multipart-nearest.json.

Independent GEOS distances/shortest_line, exhaustive NumPy (distance, feature-row) ranking.
Point and line sets each contain 30 features with single and multiple parts. The first
target has a far first part and a winning later part; query zero ties rows zero/one;
query two touches target two. Radius 2.25 cuts between parts without counting them twice.
"""
import json
import numpy as np
from shapely import distance, shortest_line
from shapely.geometry import MultiPoint, MultiLineString, LineString


def parts(query):
    rows = []
    for row in range(30):
        if query:
            points = [[row * 3, -2 - (row % 5) * .25]]
            if row % 2:
                points.append([row * 3 + 1, 10 + row % 4])
        else:
            points = [[row * 3 + .5, (row % 7) * .5]]
            if row % 3 == 0:
                points = [[-row, 100 + row], *points, [row * 3 + .5, 5 + row % 7]]
        rows.append(points)
    if query:
        rows[:3] = [[[0, 0], [1000, 1000]], [[2, 4]], [[6, 0], [-1000, -1000]]]
    else:
        rows[:3] = [[[100, 40], [-1, 0], [0, 5]], [[1, 0]], [[6, 0], [6, 20]]]
    return rows


def geometry(kind, query):
    rows = parts(query)
    feature_offsets, line_offsets, positions, shapes = [0], [0], [], []
    for row in rows:
        segments = []
        for x, y in row:
            if kind == 'points':
                positions.extend([x, y])
            else:
                segment = ([[x - .25, y], [x + .25, y]] if query
                           else [[x, y - .25], [x, y + .25]])
                positions.extend([value for point in segment for value in point])
                segments.append(segment)
                line_offsets.append(len(positions) // 2)
        feature_offsets.append(feature_offsets[-1] + len(row))
        shapes.append(MultiPoint(row) if kind == 'points' else MultiLineString(segments))
    packed = dict(kind=kind, positions=positions, featureOffsets=feature_offsets,
                  sourceIds=[(900 - row * 7) if query else (500 - row * 3) for row in range(30)])
    if kind == 'lines':
        packed['lineOffsets'] = line_offsets
    return packed, shapes


geometries = {f'{side}{kind}': geometry(kind, side == 'query')
              for side in ['query', 'feature'] for kind in ['points', 'lines']}
cases = []
for query_kind in ['points', 'lines']:
    for feature_kind in ['points', 'lines']:
        queries = geometries[f'query{query_kind}'][1]
        features = geometries[f'feature{feature_kind}'][1]
        ranked = []
        for query in queries:
            distances = np.array([distance(query, feature) for feature in features])
            order = np.lexsort((np.arange(30), distances))
            neighbors = []
            for row in order:
                pair = shortest_line(query, features[row])
                packed = geometries[f'feature{feature_kind}'][0]
                segment = 0xffffffff
                if feature_kind == 'lines':
                    first, last = packed['featureOffsets'][row:row + 2]
                    # Each fixture line part is a single segment. The first minimizing edge
                    # in vertex order is the documented public segment-index tie rule.
                    for part in range(first, last):
                        if distance(query, LineString(features[row].geoms[part - first])) == distances[row]:
                            segment = packed['lineOffsets'][part]
                            break
                neighbors.append(dict(row=int(row), distance=float(distances[row]),
                                      queryPoint=list(pair.coords[0]), featurePoint=list(pair.coords[1]),
                                      segment=segment))
            ranked.append(neighbors)
        cases.append(dict(queryKind=query_kind, featureKind=feature_kind, ranked=ranked))
print(json.dumps(dict(geometries={name: value[0] for name, value in geometries.items()}, cases=cases),
                 separators=(',', ':')))
