# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Executed upstream fixtures. Install terrain-upstream-requirements.txt in an isolated environment.
Whitebox binary v2.4.0 is additionally required; the Python wrapper downloads it on first use.
No production runtime code or GPU output is consulted by this generator.
"""
import importlib.metadata
import json
from pathlib import Path
import tempfile
import numpy as numpy
import scipy.ndimage as ndimage
from affine import Affine
from pysheds.grid import Grid
from pysheds.sview import Raster, MultiRaster, ViewFinder
import rasterio
from rasterio.crs import CRS
import xdem
from whitebox import WhiteboxTools

versions = {name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'pysheds', 'xdem', 'rasterio', 'whitebox']}
expected = {'numpy': '2.2.6', 'scipy': '1.15.3', 'pysheds': '0.5', 'xdem': '0.1.6', 'rasterio': '1.4.3', 'whitebox': '2.3.6'}
assert versions == expected, versions
width, height = 32, 24
transform = Affine(1, 0, 100, 0, -1, 100)
view = ViewFinder(shape=(height, width), affine=transform, nodata=-9999)
grid = Grid(viewfinder=view)
directions = numpy.ones((height, width), dtype=numpy.int64)
directions[:, -1] = 4
directions[-1, -1] = 0
accumulation = grid.accumulation(Raster(directions, viewfinder=view))
fractions = numpy.zeros((8, height, width))
fractions[2, :, :-1] = 0.375
fractions[4, :-1, :] = 0.625
fractions[2, -1, :-1] = 1
fractions[4, :-1, -1] = 1
branching = grid.accumulation(MultiRaster(fractions, viewfinder=ViewFinder(shape=fractions.shape, affine=transform, nodata=-9999)), routing='mfd')
# A composed catchment oracle: pysheds establishes membership; NumPy supplies area/hypsometry.
padded = numpy.pad(directions, 1, constant_values=-9999)
padded_view = ViewFinder(shape=padded.shape, affine=transform, nodata=-9999)
catchment = Grid(viewfinder=padded_view).catchment(width, height, Raster(padded, viewfinder=padded_view), xytype='index')[1:-1, 1:-1]
elevation = (height - numpy.arange(height)[:, None]) * 2 + width - numpy.arange(width)[None, :]
selected = elevation[numpy.asarray(catchment, dtype=bool)]

reference = numpy.arange(1024, dtype=numpy.float32).reshape(32, 32) * 0.125
residual = ((numpy.arange(1024) * 7) % 17 - 8).reshape(32, 32).astype(numpy.float32) * 0.25 + 3
residual.ravel()[::61] += 30
stable = numpy.ones((32, 32), dtype=bool)
stable.ravel()[::13] = False
moving = reference + residual
model = xdem.coreg.VerticalShift()
model.fit(reference, moving, inlier_mask=stable, transform=transform, crs=CRS.from_epsg(32631))
corrected, _ = model.apply(moving, transform=transform, crs=CRS.from_epsg(32631), resample=False)
vertical = {'width': 32, 'height': 32, 'bias': -float(model.to_matrix()[2, 3]), 'nmad': float(xdem.spatialstats.nmad(residual[stable])), 'corrected': (corrected-reference).ravel().tolist(), 'stableCount': int(stable.sum()), 'volume': float(numpy.sum(corrected-reference))}


# Exact dense exponential covariance under a sheared physical affine, independently in NumPy.
coordinate_index = numpy.arange(1024)
columns = coordinate_index % 32
rows = coordinate_index // 32
coordinates = numpy.column_stack((2*columns+0.5*rows, -0.25*columns+3*rows))
distance = numpy.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
correlation = numpy.exp(-distance / 3.0)
variance = 1 + (coordinate_index % 5) * 0.25
measurement_covariance = numpy.sqrt(variance[:, None]*variance[None, :]) * correlation
stable_covariance = correlation[numpy.ix_(stable.ravel(), stable.ravel())].sum()
bias_error = numpy.sqrt(numpy.pi/2) * vertical['nmad'] * numpy.sqrt(stable_covariance) / stable.sum()
area = 6.125
vertical['correlated'] = {'affine': [2, 0.5, 0, -0.25, 3, 0], 'length': 3, 'biasStandardError': float(bias_error), 'volumeStandardDeviation': float(numpy.sqrt(measurement_covariance.sum() + 1024**2 * bias_error**2) * area), 'cellStandardDeviations': numpy.sqrt(variance + bias_error**2).tolist()}

# Connected sublevel components independently establish active hierarchy area/volume.
columns, rows = numpy.meshgrid(numpy.arange(32), numpy.arange(24))
hierarchy = numpy.minimum((columns-7)**2+(rows-11)**2, 2+(columns-24)**2+(rows-12)**2).astype(numpy.float32)
stages = [0, 8, 32, 90, 200]
components = []
for stage in stages:
    labels, count = ndimage.label(hierarchy <= stage, structure=numpy.ones((3, 3)))
    components.append(sorted([[int(numpy.sum(labels == label)), float(numpy.sum(stage-hierarchy[labels == label]))] for label in range(1, count+1)]))

whitebox = WhiteboxTools()
whitebox.set_verbose_mode(False)
assert 'v2.4.0' in whitebox.version()
with tempfile.TemporaryDirectory(prefix='luma-terrain-reference-') as directory:
    directory = Path(directory)
    source = numpy.full((5, 7), 50, dtype=numpy.float32)
    source[2, 1:] = [2, 2, 5, 0, -1, -2]
    input_path, output_path = directory/'dem.tif', directory/'breached.tif'
    with rasterio.open(input_path, 'w', driver='GTiff', width=7, height=5, count=1, dtype='float32', transform=transform, crs='EPSG:32631', nodata=-9999) as destination:
        destination.write(source, 1)
    result = whitebox.breach_depressions_least_cost(str(input_path), str(output_path), dist=3, max_cost=20, min_dist=False, flat_increment=0.5, fill=False)
    assert result == 0 and output_path.exists()
    with rasterio.open(output_path) as result:
        breached = result.read(1)
    assert numpy.all(numpy.diff(breached[2, 1:]) < 0)
    breach = {'width': 7, 'height': 5, 'source': source.ravel().tolist(), 'pit': 15, 'output': breached.ravel().tolist()}

result = {'provenance': {'versions': versions, 'whiteboxBinary': '2.4.0', 'claims': 'Executed pysheds accumulation (D8 and supplied MFD fractions), xDEM VerticalShift, Whitebox least-cost breach path corridor (vertical edits differ because Whitebox lowers the pit); composed catchment reductions and SciPy sublevel components. No claim of general Whitebox path-objective identity.'}, 'routing': {'width': width, 'height': height, 'd8': numpy.asarray(accumulation).ravel().tolist(), 'branching': numpy.asarray(branching).ravel().tolist()}, 'catchment': {'area': len(selected), 'mean': float(selected.mean()), 'minimum': float(selected.min()), 'maximum': float(selected.max()), 'hypsometry': [float(numpy.mean((selected-selected.min())/(selected.max()-selected.min()) <= (index+1)/8)) for index in range(8)]}, 'vertical': vertical, 'hierarchy': {'width':32, 'height':24, 'elevation': hierarchy.ravel().tolist(), 'stages': stages, 'components': components}, 'breach': breach}
Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
print('Executed and wrote', Path(__file__).with_suffix('.json'))
