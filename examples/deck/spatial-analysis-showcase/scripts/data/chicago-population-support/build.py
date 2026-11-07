#!/usr/bin/env python3
# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Build the pinned Hyde Park population-support input from bundled tracts and ESA WorldCover."""

import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[6]
SHOWCASE = ROOT / 'examples/deck/spatial-analysis-showcase'
TRACTS = SHOWCASE / 'public/data/chicago-tracts'
OUTPUT = SHOWCASE / 'public/data/chicago-population-support'
SOURCE_URL = 'https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N39W090_Map.tif'
SOURCE_ETAG = '91546122dc8cf821894accc63dee8462-9'
FRAME_ORIGIN = (-87.59, 41.795)
FRAME_BOUNDS = (-2000.0, -2000.0, 2000.0, 2000.0)
GRID_SIZE = 64
CELL_SIZE = 62.5
EXPECTED_GEOIDS = [
    '17031410100', '17031410200', '17031410500', '17031410600',
    '17031410700', '17031410800', '17031410900', '17031411000',
    '17031411100', '17031411200', '17031836200', '17031836300'
]
EXPECTED_BOUNDS = [-87.606407, 41.785874, -87.575801, 41.802512]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = json.loads((TRACTS / 'manifest.json').read_text())
    geoids = manifest['geoid']
    community = np.fromfile(TRACTS / 'communityArea.bin', dtype='<u1')
    population = np.fromfile(TRACTS / 'population.bin', dtype='<f4')
    black = np.fromfile(TRACTS / 'nhBlack.bin', dtype='<f4')
    selected = np.flatnonzero(community == 41)
    selected_geoids = [geoids[index] for index in selected]
    if selected_geoids != EXPECTED_GEOIDS:
        raise RuntimeError(f'Hyde Park tract IDs changed: {selected_geoids}')
    if not np.isfinite(population[selected]).all() or not np.isfinite(black[selected]).all():
        raise RuntimeError('SVI population/race counts contain missing or non-finite values')
    if (population[selected] < 0).any() or (black[selected] < 0).any() or (black[selected] > population[selected]).any():
        raise RuntimeError('SVI population/race counts violate count bounds')

    features = json.loads((TRACTS / manifest['geometry']['file']).read_text())['features']
    selected_features = [feature for feature in features if int(feature['properties']['index']) in selected]
    if len(selected_features) != len(selected):
        raise RuntimeError('The tract geometry and attribute rows are not aligned')
    all_coordinates = []
    for feature in selected_features:
        geometry = feature['geometry']
        stack = [geometry['coordinates']]
        while stack:
            item = stack.pop()
            if item and isinstance(item[0], (int, float)):
                all_coordinates.append(item)
            else:
                stack.extend(item)
    bounds = [min(p[0] for p in all_coordinates), min(p[1] for p in all_coordinates),
              max(p[0] for p in all_coordinates), max(p[1] for p in all_coordinates)]
    if not np.allclose(bounds, EXPECTED_BOUNDS, atol=1.1e-6):
        raise RuntimeError(f'Whole-tract bounds changed: {bounds}')

    target_crs = rasterio.crs.CRS.from_proj4(
        f'+proj=laea +lon_0={FRAME_ORIGIN[0]} +lat_0={FRAME_ORIGIN[1]} +ellps=WGS84 +units=m +no_defs'
    )
    affine = rasterio.transform.from_bounds(*FRAME_BOUNDS, GRID_SIZE, GRID_SIZE)
    from rasterio.warp import transform as reproject_coordinates
    projected_x, projected_y = reproject_coordinates(
        'EPSG:4326', target_crs,
        [point[0] for point in all_coordinates], [point[1] for point in all_coordinates]
    )
    if min(projected_x) < FRAME_BOUNDS[0] or max(projected_x) > FRAME_BOUNDS[2] or min(projected_y) < FRAME_BOUNDS[1] or max(projected_y) > FRAME_BOUNDS[3]:
        raise RuntimeError('The chosen 4 km analysis frame does not contain all selected tracts')
    raster_bounds = transform_bounds(target_crs, 'EPSG:4326', *FRAME_BOUNDS, densify_pts=21)
    request = urllib.request.Request(SOURCE_URL, method='HEAD')
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            etag = response.headers.get('ETag', '').strip('"')
            content_length = response.headers.get('Content-Length')
            last_modified = response.headers.get('Last-Modified')
    except urllib.error.URLError as error:
        raise RuntimeError(f'Cannot verify the ESA WorldCover source: {error.reason}') from error
    if etag != SOURCE_ETAG:
        raise RuntimeError(f'Unexpected WorldCover ETag: {etag}')
    if content_length != '74618789' or last_modified != 'Wed, 26 Oct 2022 12:48:33 GMT':
        raise RuntimeError(f'Unexpected WorldCover source metadata: {content_length=} {last_modified=}')
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
        with rasterio.open(SOURCE_URL) as source:
            if source.crs is None or source.nodata not in (None, 0):
                raise RuntimeError('Unexpected WorldCover CRS or NoData definition')
            window = from_bounds(*raster_bounds, transform=source.transform).round_offsets().round_lengths()
            window = window.intersection(rasterio.windows.Window(0, 0, source.width, source.height))
            native = source.read(1, window=window)
            if native.size == 0 or not np.isin(native, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]).all():
                raise RuntimeError('WorldCover window is empty or contains unknown class codes')
            native_transform = source.window_transform(window)
            native_crs = source.crs.to_string()
            source_etag = SOURCE_ETAG

    output_bands = []
    for indicator in (native == 50, native == 10, native != 0):
        destination = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        reproject(
            source=indicator.astype(np.float32), destination=destination,
            src_transform=native_transform, src_crs=native_crs,
            dst_transform=affine, dst_crs=target_crs,
            resampling=Resampling.average, src_nodata=None, dst_nodata=0
        )
        output_bands.append(destination)
    fractions = np.stack(output_bands).astype('<f4')
    if not np.isfinite(fractions).all() or (fractions < 0).any() or (fractions > 1.0001).any():
        raise RuntimeError('Area-averaged WorldCover indicators escaped [0, 1]')
    if not np.allclose(fractions[2], 1.0, atol=1e-5):
        raise RuntimeError('The selected analysis grid has missing WorldCover support')

    OUTPUT.mkdir(parents=True, exist_ok=True)
    native_bytes = native.astype('u1').tobytes(order='C')
    fractions_bytes = fractions.tobytes(order='C')
    (OUTPUT / 'worldcover-native.bin').write_bytes(native_bytes)
    (OUTPUT / 'support-fractions.bin').write_bytes(fractions_bytes)
    (OUTPUT / 'tracts.geojson').write_text(json.dumps({
        'type': 'FeatureCollection', 'features': selected_features
    }, separators=(',', ':')) + '\n')
    (OUTPUT / 'tract-geoid.txt').write_text('\n'.join(selected_geoids) + '\n')
    (OUTPUT / 'population.bin').write_bytes(population[selected].astype('<f4').tobytes())
    (OUTPUT / 'nhBlack.bin').write_bytes(black[selected].astype('<f4').tobytes())
    source_window = {
        'colOff': int(window.col_off), 'rowOff': int(window.row_off),
        'width': int(window.width), 'height': int(window.height),
        'affine': list(native_transform)[:6], 'crs': native_crs,
        'sha256': sha256(native_bytes)
    }
    (OUTPUT / 'manifest.json').write_text(json.dumps({
        'id': 'chicago-population-support', 'version': 1, 'kind': 'raster',
        'count': GRID_SIZE * GRID_SIZE, 'bbox': EXPECTED_BOUNDS, 'crs': target_crs.to_string(),
        'raster': {
            'file': 'support-fractions.bin', 'encoding': 'float32-bin',
            'width': GRID_SIZE, 'height': GRID_SIZE, 'depth': 3,
            'bounds': list(raster_bounds), 'unit': 'fraction',
            'projection': {
                'kind': 'lambert-equal-area', 'origin': list(FRAME_ORIGIN),
                'cellSizeMeters': CELL_SIZE, 'affine': list(affine)[:6],
                'frameBoundsMeters': list(FRAME_BOUNDS), 'crs': target_crs.to_string()
            },
            'bands': ['builtUpFraction', 'treeCoverFraction', 'validFraction']
        },
        'properties': {
            'source': SOURCE_URL, 'sourceEtag': source_etag,
            'nativeWindow': source_window,
            'nativeFile': 'worldcover-native.bin',
            'indicatorBeforeAreaAverage': True,
            'resampling': 'area average of binary class indicators onto 62.5 m cells',
            'sourceClasses': {'builtUp': 50, 'treeCover': 10, 'noData': 0},
            'sourcePopulation': 'CDC/ATSDR SVI 2022, ACS 2018–2022 five-year; selected on communityArea=41',
            'geoidFile': 'tract-geoid.txt', 'populationFile': 'population.bin', 'nhBlackFile': 'nhBlack.bin',
            'tractGeometry': 'tracts.geojson', 'populationSum': float(population[selected].sum()),
            'tractCount': len(selected), 'licenses': {'population': 'Public domain', 'worldcover': 'CC BY 4.0'}
        }
    }, indent=2) + '\n')
    print(f'Wrote {OUTPUT}; {len(selected)} tracts; population={int(population[selected].sum())}; native={native.shape}; source crop sha256={source_window["sha256"]}')


if __name__ == '__main__':
    main()
