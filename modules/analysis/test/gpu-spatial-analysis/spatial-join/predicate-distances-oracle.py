# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Regenerate predicate-distances-fixture.json with /usr/bin/python3 (Shapely)."""
import json
from pathlib import Path
from shapely.geometry import Point, LineString, Polygon, MultiPolygon


def square(x, y, size):
    return [[x, y], [x + size, y], [x + size, y + size], [x, y + size]]


features = {
    'points': [{'kind': 'points', 'vertex': vertex} for vertex in [[3, 3], [1, 1], [8, 8], [3, 4]]],
    'lines': [{'kind': 'lines', 'vertices': vertices} for vertices in [
        [[-1, 1], [7, 1]], [[3, 2.5], [3, 3.5]], [[8, 8], [8, 9]], []]],
    'polygons': [{'kind': 'polygons', 'polygons': polygons} for polygons in [
        [[square(0, 0, 6), square(2, 2, 2)]], [[square(2.5, 2.5, 1)]],
        [[square(9, 8, 2)], [square(-4, -4, 1)]], [], [[square(6, 0, 1)]], [[square(-4.5, -4.5, 2)]]]]
}


def geometry(feature):
    if feature['kind'] == 'points':
        return Point(feature['vertex'])
    if feature['kind'] == 'lines':
        return LineString(feature['vertices'])
    return MultiPolygon([Polygon(rings[0], rings[1:]) for rings in feature['polygons']])


scenes = []
for left_kind, left_features in features.items():
    for right_kind, right_features in features.items():
        distances = []
        intersects = []
        touches = []
        for left_row, left_feature in enumerate(left_features):
            for right_row, right_feature in enumerate(right_features):
                left = geometry(left_feature)
                right = geometry(right_feature)
                if left.is_empty or right.is_empty:
                    continue
                distances.append([left_row, right_row, left.distance(right)])
                if left.intersects(right):
                    intersects.append([left_row, right_row, 0])
                if left.touches(right):
                    touches.append([left_row, right_row, 0])
        scenes.append(dict(leftKind=left_kind, rightKind=right_kind, distances=distances,
                           intersects=intersects, touches=touches))
Path(__file__).with_name('predicate-distances-fixture.json').write_text(json.dumps(dict(features=features, scenes=scenes), indent=2) + '\n')
