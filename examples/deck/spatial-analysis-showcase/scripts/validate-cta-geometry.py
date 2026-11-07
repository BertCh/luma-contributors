# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Independent GEOS references for the retained CTA route and Loop boundary.

Run with Python containing numpy and Shapely. Node reads the shipped typed arrays
and reproduces the scene's declared local projection, so GEOS receives exactly
the float32 coordinates supplied to the GPU. No GPU algorithm is imported.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

SHOWCASE_DIRECTORY = Path(__file__).resolve().parents[1]
REPOSITORY_DIRECTORY = SHOWCASE_DIRECTORY.parents[2]
PROJECT_SOURCE = r"""
import {readFileSync} from 'node:fs';
import {addMetersToLngLat, getDistanceScales} from '@math.gl/web-mercator';
const directory = process.argv[1];
function column(dataset, name, Type) {
  const bytes = readFileSync(`${directory}/public/data/${dataset}/${name}.bin`);
  return new Type(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}
const origin = [-87.6298, 41.8781];
const scales = getDistanceScales({longitude: origin[0], latitude: origin[1], highPrecision: true});
const metersPerDegree = [scales.unitsPerDegree[0] * scales.metersPerUnit[0], scales.unitsPerDegree[1] * scales.metersPerUnit[1]];
function project(coordinates) {
  const result = [];
  for (let index = 0; index < coordinates.length; index += 2) {
    let x = (coordinates[index] - origin[0]) * metersPerDegree[0];
    let y = (coordinates[index + 1] - origin[1]) * metersPerDegree[1];
    const estimated = addMetersToLngLat(origin, [x, y]);
    x += (coordinates[index] - estimated[0]) * metersPerDegree[0];
    y += (coordinates[index + 1] - estimated[1]) * metersPerDegree[1];
    result.push([Math.fround(x), Math.fround(y)]);
  }
  return result;
}
const offsets = column('cta-transit', 'shapePathOffsets', Uint32Array);
const coordinates = column('cta-transit', 'shapeVertices', Float32Array);
const source = coordinates.slice(offsets[16958] * 2, offsets[16959] * 2);
const ringOffsets = column('chicago-community-areas', 'ringOffsets', Uint32Array);
const boundaryCoordinates = column('chicago-community-areas', 'vertices', Float32Array);
const boundary = boundaryCoordinates.slice(ringOffsets[32] * 2, ringOffsets[33] * 2);
const stopCoordinates = column('cta-transit', 'stopPosition', Float32Array);
const stopModes = column('cta-transit', 'stopMode', Uint8Array);
const stopOffsets = column('cta-transit', 'stopRouteOffsets', Uint32Array);
const stopRoutes = column('cta-transit', 'stopRoutes', Uint16Array);
const stations = [];
const stationRows = [];
for (let row = 0; row < stopModes.length; row++) {
  if (stopModes[row] === 1 && stopRoutes.subarray(stopOffsets[row], stopOffsets[row + 1]).includes(124)) {
    stationRows.push(row);
    stations.push(stopCoordinates[row * 2], stopCoordinates[row * 2 + 1]);
  }
}
console.log(JSON.stringify({origin, source: project(source), boundary: project(boundary), stations: project(stations), stationRows}));
"""


def paths(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == 'LineString':
        return [list(geometry.coords)]
    return [list(part.coords) for part in geometry.geoms if part.geom_type == 'LineString']


def editing_references(source, stations):
    """GEOS cleanup/precision and direct nearest/reference-insertion definitions."""
    cases = []

    def add(edit, coordinates, tolerance=20, grid_size=1, **extra):
        cases.append({'edit': edit, 'tolerance': tolerance, 'gridSize': grid_size,
                      'coordinates': np.asarray(coordinates).tolist(), **extra})

    def snap_points(tolerance, inclusive=False):
        distances = np.linalg.norm(source[:, None] - stations[None, :], axis=2)
        nearest = np.argmin(distances, axis=1)
        minimum = distances[np.arange(len(source)), nearest]
        qualifies = minimum <= tolerance if inclusive else minimum < tolerance
        coordinates = source.copy()
        coordinates[qualifies] = stations[nearest[qualifies]]
        return coordinates, np.where(qualifies, nearest, 0xffffffff).tolist()

    add('reverse', source[::-1])
    for tolerance in [0, 20, 100]:
        rounded = shapely.set_precision(LineString(source), 1, mode='pointwise')
        cleaned = shapely.remove_repeated_points(rounded, tolerance)
        add('cleanup', list(cleaned.coords), tolerance=tolerance)
    for grid_size in [1, 10, 100]:
        rounded = shapely.set_precision(LineString(source), grid_size, mode='valid_output')
        add('precision', list(rounded.coords), grid_size=grid_size)
    for tolerance in [5, 20, 100]:
        snapped, reference_rows = snap_points(tolerance)
        add('vertex-snap', snapped, tolerance=tolerance, referenceRows=reference_rows)
        endpoints, _ = snap_points(tolerance, inclusive=True)
        output = [endpoints[0]]
        for start, end in zip(endpoints[:-1], endpoints[1:]):
            direction = end - start
            squared_length = direction @ direction
            candidates = []
            if squared_length > 0:
                for row, station in enumerate(stations):
                    fraction = (station - start) @ direction / squared_length
                    distance = np.linalg.norm(station - (start + fraction * direction))
                    if 0 < fraction < 1 and distance <= tolerance:
                        candidates.append((fraction, row, station))
            for _, _, coordinate in sorted(candidates, key=lambda entry: entry[:2]):
                if not np.array_equal(coordinate, output[-1]):
                    output.append(coordinate)
            if not np.array_equal(end, output[-1]):
                output.append(end)
        add('segment-snap', output, tolerance=tolerance)
    add('loop-revisit', source[:6])
    return cases


def check_gpu_output(filename, line, boundary, rectangle, diameter):
    """Check geometry, not a particular GPU partition or path-fragment ordering."""
    cases = json.loads(Path(filename).read_text())
    checks = []
    for case in cases:
        assert case['minimumOverflow'] == 0, case
        assert case['subdivisionStatus'][2:] == [0, 0], case
        assert case['subdivisionStatus'][0] == case['subdivisionStatus'][1], case
        assert abs(case['diameter'] - diameter) < 0.02, case['diameter']
        gpu_rectangle = Polygon(case['corners'])
        assert abs(gpu_rectangle.area - rectangle.area) / rectangle.area < 1e-5
        polygons = [Polygon(coordinates) for coordinates in case['subdivisionPolygons']]
        assert all(polygon.is_valid for polygon in polygons)
        union = unary_union(polygons)
        area_error = union.symmetric_difference(boundary).area
        overlap_area = sum(polygon.area for polygon in polygons) - union.area
        # Rectilinear output adds float32 intersections, so allow small metric area error.
        assert area_error < 20, (case['method'], area_error)
        assert abs(overlap_area) < 5, (case['method'], overlap_area)
        clip_checks = {}
        for mode in ['inside', 'outside']:
            status = case[mode]['status']
            assert status['overflow'] == status['uncertain'] == status['candidateOverflow'] == 0
            assert status['count'] == status['required']
            observed = unary_union([LineString(coordinates) for coordinates in case[mode]['paths']])
            expected = line.intersection(boundary) if mode == 'inside' else line.difference(boundary)
            length_error = abs(observed.length - expected.length)
            distance_error = observed.hausdorff_distance(expected)
            assert length_error < 0.1, (case['method'], mode, length_error)
            assert distance_error < 0.02, (case['method'], mode, distance_error)
            clip_checks[mode] = {'lengthErrorMeters': length_error, 'hausdorffDistanceMeters': distance_error}
        checks.append({'method': case['method'], 'subdivisionDifferenceSquareMeters': area_error,
                       'overlapSquareMeters': overlap_area, 'clips': clip_checks})
    return checks


def main():
    process = subprocess.run(
        ['node', '--input-type=module', '-e', PROJECT_SOURCE, str(SHOWCASE_DIRECTORY)],
        cwd=REPOSITORY_DIRECTORY, check=True, capture_output=True, text=True
    )
    inputs = json.loads(process.stdout)
    source = np.asarray(inputs['source'])
    line = LineString(source)
    boundary = Polygon(inputs['boundary'])
    inside = line.intersection(boundary)
    outside = line.difference(boundary)
    rectangle = line.minimum_rotated_rectangle
    diameter = float(np.sqrt(((source[:, None] - source[None, :]) ** 2).sum(axis=2)).max())
    assert boundary.is_valid
    assert abs(inside.length + outside.length - line.length) < 1e-7
    report = {
        'reference': f'Shapely {shapely.__version__}, GEOS {shapely.geos_version_string}',
        'coordinateContract': 'Same float32 local metric XY inputs as GPU; independent double-precision GEOS geometry operations.',
        'sourceShapeRow': 16958,
        'routeTableRow': 124,
        'communityFeatureRow': 31,
        'communityAreaId': 32,
        'inputs': inputs,
        'editingCases': editing_references(source, np.asarray(inputs['stations'])),
        'expected': {
            'sourceVertexCount': len(source),
            'sourceLengthMeters': line.length,
            'sourceIsSimple': line.is_simple,
            'sourceIsClosed': line.is_closed,
            'loopAreaSquareMeters': boundary.area,
            'diameterMeters': diameter,
            'hullVertexCount': len(line.convex_hull.exterior.coords) - 1,
            'minimumRectangleAreaSquareMeters': rectangle.area,
            'minimumRectangleCoordinates': list(rectangle.exterior.coords),
            'insideLengthMeters': inside.length,
            'outsideLengthMeters': outside.length,
            'insidePaths': paths(inside),
            'outsidePaths': paths(outside),
        },
        'inputChecksums': {},
    }
    for dataset, names in {
        'cta-transit': ['shapePathOffsets', 'shapeVertices', 'stopPosition', 'stopMode', 'stopRouteOffsets', 'stopRoutes'],
        'chicago-community-areas': ['ringOffsets', 'vertices'],
    }.items():
        for name in names:
            relative = f'public/data/{dataset}/{name}.bin'
            report['inputChecksums'][relative] = hashlib.sha256((SHOWCASE_DIRECTORY / relative).read_bytes()).hexdigest()
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else REPOSITORY_DIRECTORY / 'dev-docs/research/showcase-overhaul-2026-10-25/cta-geometry-validation.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'file': str(destination), **{key: value for key, value in report['expected'].items() if not isinstance(value, list)}}, indent=2))
    if len(sys.argv) > 2:
        print(json.dumps({'gpuGeometryChecks': check_gpu_output(sys.argv[2], line, boundary, rectangle, diameter)}, indent=2))


if __name__ == '__main__':
    main()
