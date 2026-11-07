# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Independent GEOS oracle for the archived Curry/Josephine linework scenarios."""
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString
from shapely.ops import polygonize_full, shared_paths, unary_union

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'public' / 'data' / 'us-counties'
MANIFEST = json.loads((DATA / 'manifest.json').read_text())


def load_values(name, dtype):
    values = np.fromfile(DATA / MANIFEST['columns'][name]['file'], dtype=dtype)
    return values


vertices = load_values('vertices', '<f4').reshape((-1, 2))
ring_offsets = load_values('ringOffsets', '<u4')
polygon_ring_offsets = load_values('polygonRingOffsets', '<u4')
county_polygon_offsets = load_values('countyPolygonOffsets', '<u4')
fips_values = load_values('fips', '<u4')

# Same USGS contiguous-US Albers parameters and Earth radius as the scene, then
# round every projected coordinate through float32 exactly as its GPU input does.
radians = math.pi / 180
n = (math.sin(29.5 * radians) + math.sin(45.5 * radians)) / 2
c = math.cos(29.5 * radians) ** 2 + 2 * n * math.sin(29.5 * radians)
rho0 = math.sqrt(c - 2 * n * math.sin(37.5 * radians)) / n
radius = 6371008.8


def project(point):
    longitude, latitude = map(float, point)
    rho = math.sqrt(c - 2 * n * math.sin(latitude * radians)) / n
    theta = n * (longitude + 96) * radians
    x = radius * rho * math.sin(theta)
    y = radius * (rho0 - rho * math.cos(theta))
    return (struct.unpack('<f', struct.pack('<f', x))[0], struct.unpack('<f', struct.pack('<f', y))[0])


def feature_rings(feature_row):
    rings = []
    for polygon in range(int(county_polygon_offsets[feature_row]), int(county_polygon_offsets[feature_row + 1])):
        for ring in range(int(polygon_ring_offsets[polygon]), int(polygon_ring_offsets[polygon + 1])):
            start, end = map(int, ring_offsets[ring:ring + 2])
            path = [project(point) for point in vertices[start:end]]
            if len(path) > 1 and path[0] == path[-1]:
                path.pop()
            if path:
                path.append(path[0])
                rings.append(path)
    return rings


rows = {int(fips): row for row, fips in enumerate(fips_values) if int(fips) in (41015, 41033)}
assert set(rows) == {41015, 41033}
global_rings = {fips: feature_rings(row) for fips, row in rows.items()}
global_coordinates = [point for paths in global_rings.values() for path in paths for point in path]
global_bounds = (
    min(point[0] for point in global_coordinates),
    min(point[1] for point in global_coordinates),
    max(point[0] for point in global_coordinates),
    max(point[1] for point in global_coordinates),
)


def float32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


# Rebase only after preserving the same global float32 projection uploaded by the scene.
coordinate_origin = (
    float32((global_bounds[0] + global_bounds[2]) / 2),
    float32((global_bounds[1] + global_bounds[3]) / 2),
)
base_rings = {
    fips: [
        [(float32(x - coordinate_origin[0]), float32(y - coordinate_origin[1])) for x, y in path]
        for path in paths
    ]
    for fips, paths in global_rings.items()
}
# The translation exactly preserves every already-quantized source coordinate.
for fips, paths in global_rings.items():
    for source_path, local_path in zip(paths, base_rings[fips]):
        for source_point, local_point in zip(source_path, local_path):
            assert tuple(local_point[axis] + coordinate_origin[axis] for axis in range(2)) == source_point
assert len(base_rings[41015]) == 1 and len(base_rings[41033]) == 1
assert len(base_rings[41015][0]) == 77 and len(base_rings[41033][0]) == 58

distinct_vertices = sorted({point for paths in base_rings.values() for path in paths for point in path[:-1]})
minimum_distinct_vertex_separation = min(
    math.dist(first, second)
    for first_index, first in enumerate(distinct_vertices)
    for second in distinct_vertices[first_index + 1:]
)
maximum_local_coordinate = max(abs(coordinate) for point in distinct_vertices for coordinate in point)
local_float32_spacing = 2 ** (math.floor(math.log2(maximum_local_coordinate)) - 23)
vertex_tolerance = max(0.001, 4 * local_float32_spacing)
assert vertex_tolerance < minimum_distinct_vertex_separation / 4


def line_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == 'LineString':
        return [geometry]
    if hasattr(geometry, 'geoms'):
        return [part for child in geometry.geoms for part in line_parts(child)]
    return []


def describe(encoding):
    rings = {fips: [list(path) for path in paths] for fips, paths in base_rings.items()}
    if encoding == 'reverse-right-ring':
        rings[41033][0] = list(reversed(rings[41033][0]))
    if encoding == 'omit-outer-edge':
        unique = rings[41015][0][:-1]
        rings[41015][0] = unique[1:] + unique[:1]
    left = LineString(rings[41015][0])
    right = LineString(rings[41033][0])
    shared = shared_paths(left, right)
    shared_forward, shared_backward = shared.geoms
    shared_segments = sum(len(part.coords) - 1 for part in line_parts(shared))
    shared_runs = line_parts(shared)
    shared_length = sum(part.length for part in shared_runs)

    source_lines = [LineString(path) for paths in rings.values() for path in paths]
    raw_source_segments = sum(len(path) - 1 for paths in rings.values() for path in paths)
    noded_union = unary_union(source_lines)
    atomic_lines = line_parts(noded_union)
    union_atomic_segments = sum(len(part.coords) - 1 for part in atomic_lines)
    faces, cuts, dangles, invalid = polygonize_full(atomic_lines)
    return {
        'encoding': encoding,
        'sourceRows': {'41015': rows[41015], '41033': rows[41033]},
        'storedRingVertices': {'41015': len(base_rings[41015][0]), '41033': len(base_rings[41033][0])},
        'inputVertices': sum(len(path) for paths in rings.values() for path in paths),
        'rawSourceAtomicSegments': raw_source_segments,
        'shared': {
            'sharedSegments': shared_segments,
            'runs': len(shared_runs),
            'vertices': sum(len(part.coords) for part in shared_runs),
            'lengthMeters': shared_length,
            'forwardSegments': sum(len(part.coords) - 1 for part in line_parts(shared_forward)),
            'backwardSegments': sum(len(part.coords) - 1 for part in line_parts(shared_backward)),
        },
        'unionNodedAtomicSegments': union_atomic_segments,
        'polygonize': {
            'faces': len(line_parts(faces)) if faces.geom_type in ('LineString', 'MultiLineString') else len(faces.geoms),
            'faceAreas': sorted(face.area for face in faces.geoms),
            'cutLines': len(line_parts(cuts)),
            'dangleLines': len(line_parts(dangles)),
            'invalidRings': len(line_parts(invalid)),
        },
    }


references = [describe(encoding) for encoding in ('archived', 'reverse-right-ring', 'omit-outer-edge')]
assert [item['sourceRows'] for item in references] == [{'41015': 2182, '41033': 2191}] * 3
assert [item['storedRingVertices'] for item in references] == [{'41015': 77, '41033': 58}] * 3
assert [item['shared']['sharedSegments'] for item in references] == [46, 46, 46]
assert abs(references[0]['shared']['lengthMeters'] - 116339.25827293281) < 0.01
assert [item['unionNodedAtomicSegments'] for item in references] == [87, 87, 86]
assert [item['rawSourceAtomicSegments'] for item in references] == [133, 133, 132]
assert [item['polygonize']['faces'] for item in references] == [2, 2, 1]
assert [item['polygonize']['cutLines'] for item in references] == [0, 0, 0]
assert [item['polygonize']['invalidRings'] for item in references] == [0, 0, 0]
assert references[2]['polygonize']['dangleLines'] == 2

report = {'shapelyVersion': shapely.__version__, 'geosVersion': shapely.geos_version_string, 'references': references}
report['localFrame'] = {
    'coordinateOrigin': coordinate_origin,
    'maximumLocalCoordinateMagnitude': maximum_local_coordinate,
    'vertexToleranceMeters': vertex_tolerance,
    'minimumDistinctVertexSeparationMeters': minimum_distinct_vertex_separation,
}
output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/tmp/county-linework-reference.json')
output_path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, separators=(',', ':')))
