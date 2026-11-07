# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Independent GEOS coverage and union reference on the declared Albers float32 frame."""
import json
import math
from pathlib import Path
import sys
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon

source = Path(__file__).resolve().parent.parent / 'public/data/us-counties'
vertices = np.fromfile(source / 'vertices.bin', dtype='<f4').reshape(-1, 2)
rings = np.fromfile(source / 'ringOffsets.bin', dtype='<u4')
polygons = np.fromfile(source / 'polygonRingOffsets.bin', dtype='<u4')
features = np.fromfile(source / 'countyPolygonOffsets.bin', dtype='<u4')
states = np.fromfile(source / 'stateFips.bin', dtype='u1')
fips = np.fromfile(source / 'fips.bin', dtype='<u4')
radians = math.pi / 180
conic = (math.sin(29.5 * radians) + math.sin(45.5 * radians)) / 2
constant = math.cos(29.5 * radians) ** 2 + 2 * conic * math.sin(29.5 * radians)
origin_rho = math.sqrt(constant - 2 * conic * math.sin(37.5 * radians)) / conic

def project(point):
    longitude, latitude = map(float, point)
    rho = math.sqrt(constant - 2 * conic * math.sin(latitude * radians)) / conic
    angle = conic * (longitude + 96) * radians
    return np.asarray([6371008.8 * rho * math.sin(angle), 6371008.8 * (origin_rho - rho * math.cos(angle))], dtype=np.float32).astype(float).tolist()

selected = []
for feature in range(len(states)):
    if int(states[feature]) not in (17, 18, 55):
        continue
    geometry_parts = []
    stored = 0
    for polygon in range(int(features[feature]), int(features[feature + 1])):
        coordinates = [[project(point) for point in vertices[rings[ring]:rings[ring + 1]]] for ring in range(int(polygons[polygon]), int(polygons[polygon + 1]))]
        stored += sum(map(len, coordinates))
        geometry_parts.append(Polygon(coordinates[0], coordinates[1:]))
    selected.append({'sourceRow': feature, 'stateFips': int(states[feature]), 'countyFips': int(fips[feature]), 'storedVertices': stored, 'geometry': MultiPolygon(geometry_parts)})
assert len(selected) == 266
assert sum(item['storedVertices'] for item in selected) == 3511
geometries = [item['geometry'] for item in selected]
validity = []
for gap in (0, 10, 100):
    invalid = shapely.coverage_invalid_edges(geometries, gap_width=gap)
    validity.append({'gapWidthMeters': gap, 'valid': bool(shapely.coverage_is_valid(geometries, gap_width=gap)), 'invalidFeatureCount': sum(not edge.is_empty for edge in invalid), 'invalidLengthMeters': float(sum(edge.length for edge in invalid))})

def describe(geometry):
    parts = list(geometry.geoms) if geometry.geom_type == 'MultiPolygon' else [geometry]
    return {'polygons': len(parts), 'rings': sum(1 + len(part.interiors) for part in parts), 'holes': sum(len(part.interiors) for part in parts), 'areaSquareMeters': geometry.area, 'valid': bool(geometry.is_valid)}

unions = {str(state): describe(shapely.union_all([item['geometry'] for item in selected if item['stateFips'] == state])) for state in (17, 18, 55)}
unions['combined'] = describe(shapely.union_all(geometries))
report = {'shapelyVersion': shapely.__version__, 'geosVersion': shapely.geos_version_string, 'featureCount': len(selected), 'storedVertices': sum(item['storedVertices'] for item in selected), 'sourceInvalidFeatures': [item['countyFips'] for item in selected if not item['geometry'].is_valid], 'coverage': validity, 'unionReference': unions}
Path(sys.argv[1] if len(sys.argv) > 1 else '/tmp/coverage-audit-reference.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
