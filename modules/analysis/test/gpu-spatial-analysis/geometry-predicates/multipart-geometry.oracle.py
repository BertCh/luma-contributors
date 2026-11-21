# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Reproduce: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/geometry-predicates/multipart-geometry.oracle.py
# Shapely 2.0.6 / GEOS 3.11.4. Output is committed in multipart-geometry-fixture.json.
import json
import math
import shapely
from shapely import wkt
from shapely.geometry import LineString
from shapely import affinity

LINES = [
    ('shared endpoint', '((0 0,1 0),(1 0,2 0))', '((0 0,1 0,2 0))'),
    ('cross', '((0 0,2 2),(0 2,2 0))', '((0 0,2 2),(0 2,2 0))'),
    ('tee', '((0 0,2 0),(1 0,1 1))', '((0 0,2 0),(1 0,1 2))'),
    ('overlap', '((0 0,2 0),(1 0,3 0))', '((0 0,2 0),(1 0,3 0))'),
    ('separate parts', '((0 0,1 0),(5 0,8 0))', '((5 0,8 0),(0 0,1 0))'),
    ('closed parts', '((0 0,2 0,1 1,0 0),(5 0,7 0,6 1,5 0))', '((0 0,2 0,1 1,0 0),(5 0,7 0,6 1,5 0))'),
    ('mixed closure', '((0 0,2 0,1 1,0 0),(5 0,7 0))', '((0 0,2 0,1 1,0 0),(5 0,7 0))'),
    ('closed endpoint contact', '((0 0,2 0,1 1,0 0),(0 0,-1 0))', '((0 0,2 0,1 1,0 0),(0 0,-1 0))'),
    ('bowtie part', '((0 0,2 2,0 2,2 0),(5 0,6 0))', '((0 0,2 2,0 2,2 0),(5 0,6 0))'),
    ('repeated vertex', '((0 0,0 0,1 0),(1 0,1 0,2 0))', '((0 0,0 0,1 0),(1 0,1 0,2 0))'),
    ('split mismatch', '((0 0,1 0),(2 0,3 0,4 0))', '((0 0,1 0,2 0),(3 0,4 0))'),
    ('reversal', '((0 0,1 0),(1 0,2 0))', '((1 0,0 0),(2 0,1 0))'),
    ('within tolerance', '((0 0,1 0),(1 0,2 0))', '((0 0.125,1 0.125),(1 0.125,2 0.125))'),
    ('outside tolerance', '((0 0,1 0),(1 0,2 0))', '((0 0.5,1 0.5),(1 0.5,2 0.5))'),
    ('empty feature', 'EMPTY', 'EMPTY'),
]
POINTS = [
    ('distinct', '(0 0,2 0,3 4)', '(0 0,2 0,3 4)'),
    ('duplicate', '(0 0,2 0,0 0)', '(0 0,2 0,0 0)'),
    ('all duplicate', '(1 2,1 2,1 2)', '(1 2,1 2,1 2)'),
    ('singleton', '(5 3)', '(5 3)'),
    ('empty feature', 'EMPTY', 'EMPTY'),
    ('reordered', '(0 0,1 0)', '(1 0,0 0)'),
    ('split count mismatch', '(0 0,1 0)', '(0 0,1 0,2 0)'),
    ('within tolerance', '(0 0,1 0)', '(0 0.125,1 0.125)'),
    ('outside tolerance', '(0 0,1 0)', '(0 0.5,1 0.5)'),
    ('negative zero', '(-0 0,1 0)', '(0 -0,1 0)'),
    ('negative coordinates', '(-2 -4,2 4)', '(-2 -4,2 4)'),
    ('tail duplicate', '(0 0,1 1,2 2,2 2)', '(0 0,1 1,2 2,2 2)'),
    ('nonconsecutive duplicate', '(0 0,1 0,2 0,1 0)', '(0 0,1 0,2 0,1 0)'),
    ('far apart', '(0 0,64 0,0 64)', '(0 0,64 0,0 64)'),
    ('last unequal', '(0 0,1 0,2 0)', '(0 0,1 0,3 0)'),
]
COLUMNS = {
    'isSimple': shapely.is_simple, 'isRing': shapely.is_ring,
    'isClosed': shapely.is_closed, 'isCcw': shapely.is_ccw,
    'numPoints': shapely.get_num_points, 'numCoordinates': shapely.get_num_coordinates,
    'numInteriorRings': shapely.get_num_interior_rings, 'numGeometries': shapely.get_num_geometries,
}

def generate(kind, records):
    prefix = 'MULTILINESTRING ' if kind == 'lines' else 'MULTIPOINT '
    result = []
    for name, left_text, right_text in records:
        left, right = wkt.loads(prefix + left_text), wkt.loads(prefix + right_text)
        if name == 'negative zero':
            assert math.copysign(1,left.geoms[0].x) == -1 and math.copysign(1,right.geoms[0].y) == -1
        def parts(geometry):
            return [list(map(list, part.coords)) for part in geometry.geoms]
        center = left.centroid
        flat_coordinates = [coordinate for part in left.geoms for coordinate in part.coords]
        extremes = []
        if flat_coordinates:
            for axis, selector in ((0, min), (1, min), (0, max), (1, max)):
                target = selector(coordinate[axis] for coordinate in flat_coordinates)
                extremes.append(next(index for index, coordinate in enumerate(flat_coordinates) if coordinate[axis] == target))
        else:
            extremes = [4294967295] * 4
        transformed = affinity.translate(affinity.scale(left, xfact=2, yfact=-1, origin='center'), xoff=3, yoff=4)
        intersection_count = 0
        if kind == 'lines':
            segments = []
            for part_index, part in enumerate(left.geoms):
                coordinates = list(part.coords)
                edges = [LineString([a, b]) for a, b in zip(coordinates, coordinates[1:]) if a != b]
                segments.extend((part_index, edge_index, len(edges), part.is_closed, edge) for edge_index, edge in enumerate(edges))
            for index, (part, edge_index, edge_count, closed, edge) in enumerate(segments):
                for other_part, other_index, _, _, other in segments[index + 1:]:
                    intersection = edge.intersection(other)
                    adjacent = part == other_part and (other_index == edge_index + 1 or (closed and edge_index == 0 and other_index == edge_count - 1))
                    if not intersection.is_empty and not (adjacent and intersection.geom_type == 'Point'):
                        intersection_count += 1
        result.append(dict(
            name=name, left=parts(left), right=parts(right),
            expected={key: int(function(left)) for key, function in COLUMNS.items()},
            equalsExact=[int(shapely.equals_exact(left, right, tolerance)) for tolerance in (0, 0.25)],
            # Finite f32 coordinates in these fixtures: exact zero tolerance also gives identical.
            equalsIdentical=int(shapely.equals_exact(left, right, 0)),
            unique=[list(point.coords[0]) for point in shapely.extract_unique_points(left).geoms],
            length=left.length, area=left.area,
            centroid=None if center.is_empty else list(center.coords[0]),
            bounds=None if left.is_empty else list(left.bounds),
            intersectionCount=intersection_count,
            extremeVertices=extremes,
            affineCenter=[list(coordinate) for part in transformed.geoms for coordinate in part.coords],
        ))
    return result

print(json.dumps({'lines': generate('lines', LINES), 'points': generate('points', POINTS)}, indent=2))
