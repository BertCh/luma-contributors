# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Command: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-join/multipart-predicates.oracle.py
# Emits the committed fixture as a patch. Shapely 2.0.6 / GEOS is independent of both kernels.
import json
import shapely
from shapely import wkt

predicates = ['intersects', 'within', 'coveredBy', 'contains', 'covers', 'containsProperly', 'touches', 'crosses', 'overlaps', 'equals']
cases = [
    ('point subset', 'MULTIPOINT ((0 0), (2 0))', 'MULTIPOINT ((0 0), (1 0), (2 0))'),
    ('equal point sets', 'MULTIPOINT ((0 0), (2 0), (0 0))', 'MULTIPOINT ((2 0), (0 0))'),
    ('partial point sets', 'MULTIPOINT ((0 0), (2 0))', 'MULTIPOINT ((2 0), (4 0))'),
    ('disjoint point sets', 'MULTIPOINT ((0 0), (2 2))', 'MULTIPOINT ((0 2), (2 0))'),
    ('point single', 'POINT (0 0)', 'MULTIPOINT ((0 0), (2 0))'),
    ('cancelled endpoint', 'MULTIPOINT ((1 0))', 'MULTILINESTRING ((0 0, 1 0), (1 0, 2 0))'),
    ('line boundary points', 'MULTIPOINT ((0 0), (2 0))', 'MULTILINESTRING ((0 0, 1 0), (1 0, 2 0))'),
    ('point line crossing', 'MULTIPOINT ((1 0), (1 1))', 'LINESTRING (0 0, 2 0)'),
    ('point polygon boundary', 'MULTIPOINT ((0 1), (2 1))', 'POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))'),
    ('point polygon mixed', 'MULTIPOINT ((1 1), (3 1))', 'POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))'),
    ('union line coverage', 'LINESTRING (0 0, 2 0)', 'MULTILINESTRING ((0 0, 1 0), (1 0, 2 0))'),
    ('union oblique coverage', 'LINESTRING (0.1 0.2, 0.4 0.8)', 'MULTILINESTRING ((0.1 0.2, 0.2 0.4), (0.2 0.4, 0.4 0.8))'),
    ('line outside part', 'MULTILINESTRING ((0 0, 1 0), (3 0, 4 0))', 'LINESTRING (0 0, 2 0)'),
    ('touching lines', 'MULTILINESTRING ((0 0, 1 0), (3 0, 4 0))', 'LINESTRING (1 0, 2 1)'),
    ('crossing lines', 'MULTILINESTRING ((0 0, 2 0), (0 2, 2 2))', 'MULTILINESTRING ((1 -1, 1 1), (1 1, 1 3))'),
    ('closed lines', 'MULTILINESTRING ((0 0, 2 0, 2 2, 0 0))', 'LINESTRING (0 0, -1 0)'),
    ('line polygon mixed', 'MULTILINESTRING ((0.5 1, 1.5 1), (3 1, 4 1))', 'POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))'),
    ('line polygon boundary', 'MULTILINESTRING ((0 0, 1 0), (1 0, 2 0))', 'POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))'),
    ('line polygon crossing', 'MULTILINESTRING ((-1 1, 1 1), (1 1, 3 1))', 'POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))'),
    ('multipart polygon', 'MULTIPOINT ((1 1), (5 1))', 'MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((4 0, 6 0, 6 2, 4 2, 4 0)))'),
    ('polygon hole filler', 'POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))', 'POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1))'),
    ('polygon contained', 'POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1))', 'POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))'),
    ('empty feature', 'MULTIPOINT EMPTY', 'MULTILINESTRING ((0 0, 1 0), (1 0, 2 0))'),
    ('long line one-unit gap', 'LINESTRING (-100000000 0, 100000000 0)', 'MULTILINESTRING ((-100000000 0, 0 0), (1 0, 100000000 0))'),
]

def float32(value):
    import struct
    return struct.unpack('f', struct.pack('f', value))[0]

def geometry(value):
    kind = 'points' if 'Point' in value.geom_type else 'lines' if 'LineString' in value.geom_type else 'polygons'
    parts = list(value.geoms) if value.geom_type.startswith('Multi') else [value]
    result = {'kind': kind, 'positions': []}
    if kind == 'lines': result['lineOffsets'] = [0]
    if kind == 'polygons': result.update(featureOffsets=[0, len(parts)], polygonOffsets=[0], ringOffsets=[0])
    elif value.geom_type.startswith('Multi'): result['featureOffsets'] = [0, len(parts)]
    for part in parts:
        rings = [part.exterior, *part.interiors] if kind == 'polygons' else [part]
        for ring in rings:
            coordinates = list(ring.coords)
            if kind == 'polygons': coordinates = coordinates[:-1]
            result['positions'].extend(float32(number) for coordinate in coordinates for number in coordinate)
            if kind != 'points': result['ringOffsets' if kind == 'polygons' else 'lineOffsets'].append(len(result['positions']) // 2)
        if kind == 'polygons': result['polygonOffsets'].append(len(result['ringOffsets']) - 1)
    return result

# Recreate GEOS operands from represented float32 vertices, so exact orientation expectations
# refer to the same vertices that the GPU receives (not the decimal input strings).
def rounded(value):
    from shapely.ops import transform
    return transform(lambda x, y, z=None: (tuple(map(float32, x)), tuple(map(float32, y))), value) if not value.is_empty else value

fixtures = []
for name, first, second in cases:
    left, right = rounded(wkt.loads(first)), rounded(wkt.loads(second))
    for reverse in [False, True]:
        source, target = (right, left) if reverse else (left, right)
        fixtures.append(dict(name=name + (' reversed' if reverse else ''), left=geometry(source), right=geometry(target), matrix=shapely.relate(source, target), predicates=[bool(getattr(shapely, {'coveredBy': 'covered_by', 'containsProperly': 'contains_properly'}.get(predicate, predicate))(source, target)) for predicate in predicates], dwithin=bool(shapely.dwithin(source, target, 0.5)), pattern=bool(shapely.relate_pattern(source, target, 'T********'))))

lines = ['// SPDX-License-Identifier: MIT', '// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors', '// Shapely 2.0.6 literals; regenerate with multipart-predicates.oracle.py.', 'export const MULTIPART_PREDICATES = ' + json.dumps(predicates) + ' as const;', 'export const MULTIPART_PREDICATE_FIXTURES = ' + json.dumps(fixtures, indent=2) + ' as const;']
print('*** Begin Patch\n*** Add File: modules/analysis/test/gpu-spatial-analysis/spatial-join/multipart-predicates-fixtures.ts')
for line in '\n'.join(lines).splitlines(): print('+' + line)
print('*** End Patch')
