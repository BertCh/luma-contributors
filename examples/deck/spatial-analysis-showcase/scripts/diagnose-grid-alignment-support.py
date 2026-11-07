# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Rank exact CRS-resampling support boundaries that a GPU mask readback should inspect."""

import json
import math
import sys
from pathlib import Path

import numpy
from rasterio.warp import transform

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'public/data/dixie-spectral'
DEFAULT_OUTPUT = Path('/tmp/luma-showcase-overhaul-20261025/grid-support-boundaries.json')
WIDTH = 64
HEIGHT = 64
TILE_WIDTH = 250
CELL_SIZE = 60.0
DESTINATION_CENTER = (666640.0, 4454120.0)
SOURCE_BOUNDS = (662800.0, 4441040.0, 677800.0, 4456040.0)
RADIUS = 6378137.0


def load_array(name, dtype, shape):
    return numpy.fromfile(DATA / name, dtype=dtype).reshape(shape)


def make_destination_source_coordinates(center_longitude_latitude=None):
    center_longitude, center_latitude = transform(
        'EPSG:32610', 'EPSG:4326', [DESTINATION_CENTER[0]], [DESTINATION_CENTER[1]]
    )
    center_longitude = center_longitude[0]
    center_latitude = center_latitude[0]
    if center_longitude_latitude is not None:
        center_longitude, center_latitude = center_longitude_latitude
    cosine = math.cos(math.radians(center_latitude))
    mercator_y = RADIUS * math.log(math.tan(math.pi / 4 + math.radians(center_latitude) / 2))
    columns, rows = numpy.meshgrid(numpy.arange(WIDTH), numpy.arange(HEIGHT))
    local_x = (columns + 0.5) * CELL_SIZE - WIDTH * CELL_SIZE / 2
    local_y = HEIGHT * CELL_SIZE / 2 - (rows + 0.5) * CELL_SIZE
    longitudes = center_longitude + numpy.degrees(local_x / (RADIUS * cosine))
    latitudes = numpy.degrees(
        2 * numpy.arctan(numpy.exp((mercator_y + local_y / cosine) / RADIUS)) - math.pi / 2
    )
    eastings, northings = transform(
        'EPSG:4326', 'EPSG:32610', longitudes.ravel().tolist(), latitudes.ravel().tolist()
    )
    minimum_easting, minimum_northing, maximum_easting, maximum_northing = SOURCE_BOUNDS
    source_columns = (numpy.asarray(eastings) - minimum_easting) / CELL_SIZE - 0.5
    source_rows = (maximum_northing - numpy.asarray(northings)) / CELL_SIZE - 0.5
    return numpy.column_stack((source_columns, source_rows)), numpy.column_stack((eastings, northings))


def sample_support(mask, column, row, method, policy):
    if method == 'nearest':
        sample_column = math.floor(column + 0.5)
        sample_row = math.floor(row + 0.5)
        return (
            0 <= sample_column < TILE_WIDTH
            and 0 <= sample_row < TILE_WIDTH
            and bool(mask[sample_row, sample_column])
        )

    base_column = math.floor(column)
    base_row = math.floor(row)
    fraction_column = column - base_column
    fraction_row = row - base_row
    valid_weight = 0.0
    complete = True
    for row_offset in range(2):
        for column_offset in range(2):
            weight = (fraction_column if column_offset else 1 - fraction_column) * (
                fraction_row if row_offset else 1 - fraction_row
            )
            if weight == 0:
                continue
            sample_column = base_column + column_offset
            sample_row = base_row + row_offset
            valid = (
                0 <= sample_column < TILE_WIDTH
                and 0 <= sample_row < TILE_WIDTH
                and bool(mask[sample_row, sample_column])
            )
            complete = complete and valid
            valid_weight += weight if valid else 0.0
    return valid_weight > 0 and (policy == 'renormalize' or complete)


def get_boundary_distance(mask, column, row, method, policy):
    """Return the nearest support-changing coordinate boundary, in projected metres."""
    if method == 'nearest':
        boundaries = (
            ('column', math.floor(column - 0.5) + 0.5),
            ('row', math.floor(row - 0.5) + 0.5),
        )
    else:
        boundaries = (('column', round(column)), ('row', round(row)))
    candidates = []
    for axis, boundary in boundaries:
        distance_cells = abs((column if axis == 'column' else row) - boundary)
        if distance_cells > 0.1:
            continue
        before = boundary - 1e-7
        after = boundary + 1e-7
        if axis == 'column':
            left_valid = sample_support(mask, before, row, method, policy)
            right_valid = sample_support(mask, after, row, method, policy)
        else:
            left_valid = sample_support(mask, column, before, method, policy)
            right_valid = sample_support(mask, column, after, method, policy)
        if left_valid != right_valid:
            candidates.append({
                'axis': axis,
                'boundary': boundary,
                'distanceMetres': distance_cells * CELL_SIZE,
                'supportBefore': left_valid,
                'supportAfter': right_valid
            })
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: candidate['distanceMetres'])


def main():
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    source_coordinates, projected_coordinates = make_destination_source_coordinates()
    reports = []
    for date in ('before', 'after'):
        validity = load_array(f'validity_{date}.bin', 'u1', (250, 250)) != 0
        values = load_array(f'nir_{date}.bin', '<f4', (250, 250))
        mask = validity & numpy.isfinite(values)
        for method in ('nearest', 'bilinear'):
            for policy in ('strict', 'renormalize'):
                output_mask = numpy.zeros(WIDTH * HEIGHT, dtype=bool)
                boundary_candidates = []
                for index, ((column, row), (easting, northing)) in enumerate(
                    zip(source_coordinates, projected_coordinates)
                ):
                    output_mask[index] = sample_support(mask, column, row, method, policy)
                    boundary = get_boundary_distance(mask, column, row, method, policy)
                    if boundary:
                        boundary_candidates.append({
                            'destinationRow': index // WIDTH,
                            'destinationColumn': index % WIDTH,
                            'sourceColumn': float(column),
                            'sourceRow': float(row),
                            'easting': float(easting),
                            'northing': float(northing),
                            'referenceSupport': bool(output_mask[index]),
                            **boundary
                        })
                boundary_candidates.sort(key=lambda candidate: candidate['distanceMetres'])
                reports.append({
                    'date': date,
                    'method': method,
                    'policy': policy,
                    'rasterioSupportCount': int(output_mask.sum()),
                    'supportChangingBoundaryCandidateCount': len(boundary_candidates),
                    'nearestBoundaryCandidates': boundary_candidates[:12]
                })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        'status': 'candidate-ranking-only; the GPU mask must be read back to identify the exact differing pixel',
        'coordinateMethod': 'Rasterio/GDAL WGS84 UTM10N transforms and direct scalar sampling, matching the independent support oracle',
        'destinationCenterUtm': DESTINATION_CENTER,
        'sourceBoundsUtm': SOURCE_BOUNDS,
        'cellSizeMetres': CELL_SIZE,
        'cases': reports
    }, indent=2) + '\n')
    for report in reports:
        nearest = report['nearestBoundaryCandidates'][0] if report['nearestBoundaryCandidates'] else None
        print(json.dumps({
            'case': f"{report['date']} {report['method']} {report['policy']}",
            'referenceCount': report['rasterioSupportCount'],
            'closestSupportChangingBoundary': nearest
        }))
    print(f'Wrote candidate diagnostics to {output_path}')


if __name__ == '__main__':
    main()
