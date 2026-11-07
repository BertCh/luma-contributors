#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Pinned Sentinel-2 six-band, strict-quality 60 m extract of the Dixie Fire window."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import rasterio
from rasterio.windows import from_bounds

DATASET_ID = 'dixie-spectral'
BOUNDS = (662800, 4441040, 677800, 4456040)
GEOGRAPHIC_BOUNDS = [-121.08993894007818, 40.103921542233465, -120.90991598607712, 40.235939588107]
SIZE = 250
BANDS = ['blue', 'green', 'red', 'nir', 'swir16', 'swir22']
SCENES = {'before': 'S2B_10TFK_20210713_0_L2A', 'after': 'S2B_10TFK_20210921_0_L2A'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_native(asset):
    with rasterio.open(asset['href']) as dataset:
        assert dataset.crs.to_epsg() == 32610
        window = from_bounds(*BOUNDS, transform=dataset.transform).round_offsets().round_lengths()
        values = dataset.read(1, window=window)
        factor = values.shape[0] // SIZE
        assert values.shape == (SIZE * factor, SIZE * factor)
        assert dataset.transform.a * factor == 60 and -dataset.transform.e * factor == 60
        return values, factor


def build():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('/tmp/showcase-dixie-spectral'))
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[3] / 'public/data' / DATASET_ID)
    arguments = parser.parse_args()
    arguments.cache.mkdir(parents=True, exist_ok=True)
    arguments.output.mkdir(parents=True, exist_ok=True)
    raster_specs = {}
    checksums = {}
    provenance = []
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', AWS_NO_SIGN_REQUEST='YES'):
        for label, scene_id in SCENES.items():
            metadata_url = 'https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/' + scene_id
            metadata_path = arguments.cache / (scene_id + '.json')
            if not metadata_path.exists():
                with urllib.request.urlopen(metadata_url, timeout=60) as response:
                    metadata_path.write_bytes(response.read())
            metadata_bytes = metadata_path.read_bytes()
            scene = json.loads(metadata_bytes)
            assert float(scene['properties']['s2:processing_baseline']) < 4
            classification_path = arguments.cache / (label + '-scl.npy')
            if not classification_path.exists():
                classification, factor = read_native(scene['assets']['scl'])
                assert factor == 3
                np.save(classification_path, classification)
            classification = np.load(classification_path)
            valid = np.isin(classification, [4, 5, 6]).reshape(SIZE, 3, SIZE, 3).all(axis=(1, 3))
            land_valid = np.isin(classification, [4, 5]).reshape(SIZE, 3, SIZE, 3).all(axis=(1, 3))
            values_by_band = {}
            for band in BANDS:
                asset = scene['assets'][band]
                source_path = arguments.cache / (label + '-' + band + '.npy')
                if not source_path.exists():
                    native, factor = read_native(asset)
                    np.save(source_path, native)
                native = np.load(source_path)
                factor = native.shape[0] // SIZE
                blocks = native.reshape(SIZE, factor, SIZE, factor)
                valid &= (blocks > 0).all(axis=(1, 3))
                scale = asset.get('raster:bands', [{}])[0].get('scale', 0.0001)
                offset = asset.get('raster:bands', [{}])[0].get('offset', 0)
                assert float(scale) == 0.0001 and float(offset) == 0
                values_by_band[band] = blocks.mean(axis=(1, 3), dtype=np.float64) * scale
            for band, values in values_by_band.items():
                values[~valid] = np.nan
                filename = f'{band}_{label}.bin'
                data = values.astype('<f4').tobytes()
                (arguments.output / filename).write_bytes(data)
                checksums[filename] = digest(data)
                raster_specs[f'{band}_{label}'] = {
                    'file': filename, 'encoding': 'float32-bin', 'dtype': 'float32',
                    'width': SIZE, 'height': SIZE, 'bounds': GEOGRAPHIC_BOUNDS,
                    'boundsProjected': list(BOUNDS), 'projection': 'EPSG:32610',
                    'cellSizeM': 60, 'noData': 'NaN', 'unit': 'unitless surface reflectance',
                    'scale': 1, 'offset': 0, 'date': scene['properties']['datetime'][:10],
                    'scene': scene_id, 'sourceUrl': scene['assets'][band]['href']
                }
            mask_filename = f'validity_{label}.bin'
            mask_bytes = valid.astype('uint8').tobytes()
            (arguments.output / mask_filename).write_bytes(mask_bytes)
            checksums[mask_filename] = digest(mask_bytes)
            land_valid &= valid
            land_filename = f'land_valid_{label}.bin'
            land_bytes = land_valid.astype('uint8').tobytes()
            (arguments.output / land_filename).write_bytes(land_bytes)
            checksums[land_filename] = digest(land_bytes)
            provenance.append({
                'label': label, 'scene': scene_id, 'date': scene['properties']['datetime'][:10],
                'sourceUrl': metadata_url, 'metadataSha256': digest(metadata_bytes),
                'processingBaseline': scene['properties']['s2:processing_baseline'],
                'boaOffsetApplied': scene['properties'].get('earthsearch:boa_offset_applied'),
                'validityFile': mask_filename, 'validCells': int(valid.sum()),
                'landValidityFile': land_filename, 'landValidCells': int(land_valid.sum()),
                'classificationSource': scene['assets']['scl']['href']
            })
            (arguments.output / f'{label}-source.json').write_bytes(metadata_bytes)
            print(label, scene_id, 'valid', int(valid.sum()), '/', SIZE * SIZE, flush=True)
    manifest = {
        'id': DATASET_ID, 'version': 1, 'kind': 'raster', 'count': SIZE * SIZE,
        'bbox': GEOGRAPHIC_BOUNDS, 'crs': 'EPSG:32610',
        'raster': raster_specs['nir_before'], 'rasters': raster_specs,
        'properties': {
            'source': 'Copernicus Sentinel-2 L2A; pinned Earth Search STAC items and AWS COG windows; source URLs, metadata hashes and calibration are recorded in scenes and rasters.',
            'layout': 'row-major, row zero north, little-endian float32',
            'dates': ['2021-07-13', '2021-09-21'], 'bands': BANDS, 'scenes': provenance,
            'checksums': checksums, 'retrieved': '2026-10-25',
            'attribution': 'Contains modified Copernicus Sentinel data 2021',
            'license': 'Copernicus Sentinel data: free, full and open',
            'qualityRule': 'Every contributing native 20 m SCL pixel must be vegetation (4), nonvegetated (5), or water (6), and every contributing reflectance sample must be nonzero in all six bands.',
            'landQualityRule': 'Land masks additionally require every native SCL pixel to be vegetation (4) or nonvegetated (5), excluding any water contribution. Intersect both date masks for stable valid land.',
            'resampling': 'Arithmetic mean of native reflectance within exactly aligned 60 m blocks; no reprojection or gap filling.',
            'limitations': [
                'Strict SCL screening is conservative but is not manual cloud verification.',
                'The before scene is the fire ignition date; it is not a pre-season climatology.',
                'Indices and spectral mixture weights are sensor-dependent models, not validated severity or cover classes.',
                'Gao-style NDWI with Sentinel SWIR1 1.61 micrometers is a sensor adaptation, not the original 1.24 micrometer band.'
            ]
        }
    }
    (arguments.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    build()
