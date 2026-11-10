# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent UR04 composed oracle, plus direct Tobler binary-mask comparisons.

Install pinned reference dependencies in an isolated environment:
  pip install tobler==0.14.0 shapely==2.1.2 geopandas==1.1.1 rasterio==1.4.3 numpy==2.2.6
Run: python -I modules/analysis/test/oracles/dasymetric-redistribution.py

GEOS intersections supply fractional source/cell support. Continuous ancillary density
is a weighted sum of these exact geometric areas, composed with extensive allocation;
this is NOT direct Tobler continuous-weight parity. Binary fixtures additionally call
masked_area_interpolate on a real temporary GeoTIFF and original source geometries.
Targets are unions of complete raster cells. No products of marginal fractions occur.
All source identities survive in the fixture even when Tobler drops masked-empty sources.
"""
import importlib.metadata
import json
from pathlib import Path
import tempfile

import geopandas
import numpy
import rasterio
from rasterio.transform import from_origin
from shapely import box, union_all
from tobler.dasymetric import masked_area_interpolate

PINNED = {'tobler': '0.14.0', 'shapely': '2.1.2', 'geopandas': '1.1.1',
          'rasterio': '1.4.3', 'numpy': '2.2.6'}
for package, version in PINNED.items():
    assert importlib.metadata.version(package) == version, (package, version)

cells = [box(index, 0, index + 1, 1) for index in range(6)]
sources = [box(.5, 0, 2.5, 1), box(3.5, 0, 5, 1), box(5, 0, 6, 1)]
source_totals = numpy.array([120., 80., 50.])
target_ids = [0, 0, 1, 1, 4294967295, 2]
targets = [union_all([cell for cell, owner in zip(cells, target_ids) if owner == target])
           for target in range(4)]
coverage = [[source.intersection(cell).area for cell in cells] for source in sources]
tuples = [(source, cell, fraction) for source, fractions in enumerate(coverage)
          for cell, fraction in enumerate(fractions) if fraction > 0]
cases = []
for name, ancillary in [('continuous', [2., 0., 1., 3., 1., 0.]),
                        ('binary', [1., 0., 1., 1., 1., 0.]),
                        ('aligned-binary', [1., 0., 1., 1., 1., 0.])]:
    if name == 'aligned-binary':
        sources = [box(0, 0, 3, 1), box(3, 0, 5, 1), box(5, 0, 6, 1)]
    coverage = [[source.intersection(cell).area for cell in cells] for source in sources]
    case_tuples = [(source, cell, fraction) for source, fractions in enumerate(coverage)
                   for cell, fraction in enumerate(fractions) if fraction > 0]
    eligible = numpy.array(coverage) @ numpy.array(ancillary)
    pair_weights = numpy.array([[sum(source.intersection(cell).intersection(target).area * weight
                                    for cell, weight in zip(cells, ancillary))
                                for source in sources] for target in targets])
    overlap = pair_weights.sum(axis=0)
    for denominator in ['zone', 'overlap']:
        support = eligible if denominator == 'zone' else overlap
        reciprocal = numpy.divide(1, support, out=numpy.zeros(3), where=support > 0)
        target_totals = pair_weights @ (source_totals * reciprocal)
        fractions = overlap * reciprocal
        allocated = source_totals * fractions
        unallocated = source_totals - allocated
        numpy.testing.assert_allclose(target_totals.sum() + unallocated.sum(), source_totals.sum())
        result = {
            'name': name + '-' + denominator, 'reference': 'composed GEOS area intersections and weighted extensive allocation',
            'ancillary': ancillary, 'denominator': denominator,
            'coverage': {'zoneIds': [row[0] for row in case_tuples], 'cellIds': [row[1] for row in case_tuples], 'fractions': [row[2] for row in case_tuples]},
            'targetTotals': target_totals.tolist(), 'targetSupportWeights': pair_weights.sum(axis=1).tolist(),
            'sourceEligibleWeights': eligible.tolist(), 'sourceTransferredFractions': [float(value) if eligible[index] > 0 else None for index, value in enumerate(fractions)],
            'sourceAllocatedTotals': allocated.tolist(), 'sourceUnallocatedTotals': unallocated.tolist()
        }
        if name == 'aligned-binary':
            with tempfile.TemporaryDirectory() as temporary:
                raster = str(Path(temporary) / 'eligibility.tif')
                with rasterio.open(raster, 'w', driver='GTiff', width=6, height=1, count=1,
                                   dtype='uint8', crs='EPSG:3857', transform=from_origin(0, 1, 1, 1), nodata=255) as dataset:
                    dataset.write(numpy.array([ancillary], dtype=numpy.uint8), 1)
                direct = masked_area_interpolate(
                    geopandas.GeoDataFrame({'mass': source_totals}, geometry=sources, crs='EPSG:3857'),
                    geopandas.GeoDataFrame(geometry=targets, crs='EPSG:3857'), raster,
                    pixel_values=[1], extensive_variables=['mass'], allocate_total=denominator == 'overlap',
                    nodata=255, n_jobs=1, fill_nan=None)
                numpy.testing.assert_allclose(direct['mass'], target_totals, atol=1e-6)
                result['directReference'] = 'tobler.dasymetric.masked_area_interpolate'
                result['directTargetTotals'] = direct['mass'].tolist()
        cases.append(result)
fixture = {
    'generator': Path(__file__).name,
    'command': 'python -I modules/analysis/test/oracles/dasymetric-redistribution.py',
    'versions': {name: importlib.metadata.version(name) for name in [*PINNED, 'scipy', 'pandas', 'pyproj']},
    'conventions': {'coordinates': 'EPSG:3857 planar meters', 'areaUnits': 'square meters',
                    'ancillaryUnits': 'relative density per square meter', 'targetOwnership': 'whole cells',
                    'sourceIds': 'original feature rows, including zero-eligible sources',
                    'tolerance': 'absolute 1e-4 plus relative 1e-5', 'rounding': 'float32 GPU inputs, float64 GEOS oracle'},
    'sourceTotals': source_totals.tolist(), 'targetZones': target_ids, 'targetZoneCount': 4,
    'coverage': {'zoneIds': [row[0] for row in tuples], 'cellIds': [row[1] for row in tuples], 'fractions': [row[2] for row in tuples]},
    'sourcePositions': [.5, 0, 2.5, 0, 2.5, 1, .5, 1, 3.5, 0, 5, 0, 5, 1, 3.5, 1, 5, 0, 6, 0, 6, 1, 5, 1],
    'cases': cases
}
output = Path(__file__).parent / 'fixtures/dasymetric-redistribution.json'
output.write_text(json.dumps(fixture, indent=2, allow_nan=False) + '\n')
print(output)
