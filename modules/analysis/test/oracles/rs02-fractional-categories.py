# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""RS02 direct exactextract categorical reference, without production code imports.

Reproduce with Python 3.12 and:
  pip install exactextract==0.3.0 numpy==2.2.6 shapely==2.1.2
  python modules/analysis/test/oracles/rs02-fractional-categories.py

Calls exactextract frac/majority/minority/count directly. Shapely builds adversarial
multipart/hole polygons only; it does not compute the statistics. Full-coverage
cell IDs/fractions come from a separate all-valid unit raster, retaining nodata
support. Planar cells are 2 by 3 coordinate units. Category ties follow exactextract
0.3.0: greatest majority, lowest minority. All expected categories are integer IDs;
fractions are binary-representable. Empty fractions are undefined (JSON null).

Sources:
https://isciences.github.io/exactextract/operations.html
https://github.com/isciences/exactextract/blob/v0.3.0/src/raster_stats.h
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource
import numpy
from shapely.geometry import Polygon, box, mapping
from shapely.ops import unary_union

VERSIONS = {name: importlib.metadata.version(name) for name in ['exactextract', 'numpy', 'shapely']}
assert VERSIONS == {'exactextract': '0.3.0', 'numpy': '2.2.6', 'shapely': '2.1.2'}
NODATA = 0xffffffff
WIDTH, HEIGHT, CATEGORIES = 4, 2, 6
CELL_AREA = 6


def raster(values):
    return NumPyRasterSource(numpy.asarray(values, dtype=numpy.uint32).reshape(HEIGHT, WIDTH),
                             xmin=0, ymin=0, xmax=8, ymax=6, nodata=NODATA)


def partial_cells(fractions):
    return unary_union([box((cell % WIDTH) * 2, 6 - (cell // WIDTH + 1) * 3,
                            (cell % WIDTH) * 2 + fraction * 2, 6 - (cell // WIDTH) * 3)
                        for cell, fraction in enumerate(fractions) if fraction > 0])


geometries = [
    partial_cells([1, .25, .25, 1, .5, .75, .25, 0]),
    partial_cells([1, 0, 0, 1, 0, 0, 0, 0]),
    partial_cells([0, 0, 0, 0, 0, .5, 0, 0]),
    box(10, 10, 11, 11),
    Polygon([(0, 0), (8, 0), (8, 6), (0, 6)], holes=[[(2, 1.5), (6, 1.5), (6, 4.5), (2, 4.5)]])
]
features = [{'type': 'Feature', 'properties': {}, 'geometry': mapping(geometry)} for geometry in geometries]
full_reference = exact_extract(raster([1] * 8), features, ['cell_id', 'coverage', 'count'])
tuples = {'zoneIds': [], 'cellIds': [], 'fractions': []}
for zone, feature in enumerate(full_reference):
    for cell, fraction in zip(feature['properties']['cell_id'], feature['properties']['coverage']):
        tuples['zoneIds'].append(zone)
        tuples['cellIds'].append(int(cell))
        tuples['fractions'].append(float(fraction))

positions, ring_offsets, polygon_offsets, feature_offsets = [], [0], [0], [0]
for geometry in geometries:
    polygons = list(geometry.geoms) if geometry.geom_type == 'MultiPolygon' else [geometry]
    for polygon in polygons:
        for ring in [polygon.exterior, *polygon.interiors]:
            positions.extend(coordinate for point in list(ring.coords)[:-1] for coordinate in point)
            ring_offsets.append(len(positions) // 2)
        polygon_offsets.append(len(ring_offsets) - 1)
    feature_offsets.append(len(polygon_offsets) - 1)

frames = []
for name, values, validity in [
    ('fractional-support', [0, 1, 1, 2, 3, NODATA, 2, 0], [1] * 8),
    ('updated-categories-and-mask', [2, 0, 0, 2, 1, 4, 3, 5], [1, 1, 0, 1, 0, 1, 1, 1]),
    ('all-missing', [0, 1, 1, 2, 3, NODATA, 2, 0], [0] * 8)
]:
    masked_values = [value if valid else NODATA for value, valid in zip(values, validity)]
    reference = exact_extract(raster(masked_values), features, ['unique', 'frac', 'majority', 'minority', 'count', 'class_total=count(coverage_weight=area_cartesian)'])
    expected = {key: [] for key in ['coverageSums', 'classAreas', 'proportions', 'coveredArea', 'validArea', 'missingArea', 'majority', 'minority', 'validity']}
    for zone, feature in enumerate(reference):
        properties = feature['properties']
        total = properties['count']
        fractions = dict(zip(map(int, properties['unique']), map(float, properties['frac'])))
        expected['coverageSums'].extend(fractions.get(category, 0) * total for category in range(CATEGORIES))
        expected['classAreas'].extend(fractions.get(category, 0) * properties['class_total'] for category in range(CATEGORIES))
        expected['proportions'].extend(fractions.get(category, 0) if total > 0 else None for category in range(CATEGORIES))
        covered = full_reference[zone]['properties']['count'] * CELL_AREA
        observed = properties['class_total']
        expected['coveredArea'].append(covered)
        expected['validArea'].append(observed)
        expected['missingArea'].append(covered - observed)
        expected['majority'].append(int(properties['majority']) if total > 0 else NODATA)
        expected['minority'].append(int(properties['minority']) if total > 0 else NODATA)
        expected['validity'].append(int(total > 0))
    frames.append({'name': name, 'values': values, 'validity': validity, 'expected': expected})

fixture = {
    'versions': {**VERSIONS, 'python': platform.python_version()},
    'generatorSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'conventions': __doc__,
    'width': WIDTH, 'height': HEIGHT, 'affine': [2, 0, 0, 0, -3, 6],
    'categoryCapacity': CATEGORIES, 'zoneCount': len(features), 'noDataValue': NODATA,
    'coverage': tuples,
    'geometry': {'positions': positions, 'ringOffsets': ring_offsets, 'polygonOffsets': polygon_offsets, 'featureOffsets': feature_offsets},
    'frames': frames
}
Path(__file__).with_name('fixtures').joinpath('rs02-fractional-categories.json').write_text(json.dumps(fixture, indent=2, allow_nan=False) + '\n')
