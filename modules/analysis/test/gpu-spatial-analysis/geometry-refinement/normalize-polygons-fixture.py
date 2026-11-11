# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Regenerate the independent GEOS canonical polygon fixture using Shapely 2."""
import json
from pathlib import Path

from shapely import normalize
from shapely.geometry import MultiPolygon, Polygon

features = [
    MultiPolygon([
        Polygon([(8, 8), (0, 8), (0, 0), (8, 0), (8, 8)], [
            [(2, 2), (2, 1), (1, 1), (1, 2), (2, 2)],
            [(7, 7), (6, 7), (6, 6), (7, 6), (7, 7)],
        ]),
        Polygon([(14, 0), (14, 2), (12, 2), (12, 0), (14, 0)]),
    ]),
    MultiPolygon([
        Polygon([(-2, -1), (-4, -1), (-4, -3), (-2, -3), (-2, -1)]),
        Polygon([(24, 0), (24, 4), (20, 4), (20, 0), (24, 0)], [
            [(21, 1), (22, 1), (22, 2), (21, 2), (21, 1)],
        ]),
    ]),
]


def flatten(geometries):
    positions = []
    ring_offsets = [0]
    polygon_offsets = [0]
    feature_offsets = [0]
    for geometry in geometries:
        for polygon in geometry.geoms:
            for ring in [polygon.exterior, *polygon.interiors]:
                positions.extend(component for point in ring.coords for component in point)
                ring_offsets.append(len(positions) // 2)
            polygon_offsets.append(len(ring_offsets) - 1)
        feature_offsets.append(len(polygon_offsets) - 1)
    return dict(positions=positions, ringOffsets=ring_offsets, polygonOffsets=polygon_offsets, featureOffsets=feature_offsets)


Path(__file__).with_suffix('.json').write_text(json.dumps(dict(input=flatten(features), expected=flatten([normalize(feature) for feature in features])), indent=2) + '\n')
