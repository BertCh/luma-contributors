#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Build a bounded monthly NOAA ERSSTv5 extract; Python standard library only."""

import argparse
import array
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import urllib.parse
import urllib.request

DATASET_ID = 'northeast-pacific-sst'
BASE_URL = 'https://coastwatch.pfeg.noaa.gov/erddap'
SELECTION = ('[(2012-01-15T00:00:00Z):1:(2017-12-15T00:00:00Z)]'
             '[(0):1:(0)][(20):1:(60)][(-160):1:(-110)]')
SOURCE_URL = BASE_URL + '/griddap/nceiErsstv5_LonPM180.csv?' + urllib.parse.quote(
    'sst' + SELECTION + ',ssta' + SELECTION, safe=':,')
METADATA_URL = BASE_URL + '/info/nceiErsstv5_LonPM180/index.csv'
WIDTH, HEIGHT, DEPTH = 26, 21, 72
DATES = [f'{year}-{month:02d}-15T00:00:00Z'
         for year in range(2012, 2018) for month in range(1, 13)]


def fetch_cached(path, url):
    if not path.exists():
        with urllib.request.urlopen(url, timeout=90) as response:
            data = response.read()
        path.write_bytes(data)
    return path.read_bytes()


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def build():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('/tmp/showcase-northeast-pacific-sst'))
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[3] / 'public/data' / DATASET_ID)
    arguments = parser.parse_args()
    arguments.cache.mkdir(parents=True, exist_ok=True)
    arguments.output.mkdir(parents=True, exist_ok=True)
    source = fetch_cached(arguments.cache / 'source.csv', SOURCE_URL)
    metadata = fetch_cached(arguments.cache / 'metadata.csv', METADATA_URL)
    reader = csv.DictReader(io.StringIO(source.decode()))
    units = next(reader)
    assert units['sst'] == 'degree_C' and units['ssta'] == 'degree_C'
    count = WIDTH * HEIGHT * DEPTH
    temperature = array.array('f', [math.nan]) * count
    anomaly = array.array('f', [math.nan]) * count
    admitted = set()
    date_indices = {date: index for index, date in enumerate(DATES)}
    for record in reader:
        longitude, latitude = float(record['longitude']), float(record['latitude'])
        column = int((longitude + 160) / 2)
        row = int((60 - latitude) / 2)
        assert longitude == -160 + column * 2 and latitude == 60 - row * 2
        assert 0 <= column < WIDTH and 0 <= row < HEIGHT and float(record['depth']) == 0
        index = (row * WIDTH + column) * DEPTH + date_indices[record['time']]
        assert index not in admitted
        admitted.add(index)
        for field, values, lower, upper in [('sst', temperature, -3, 45), ('ssta', anomaly, -12, 12)]:
            value = float(record[field])
            assert math.isnan(value) or lower <= value <= upper
            values[index] = value
    assert len(admitted) == count
    if sys.byteorder != 'little':
        temperature.byteswap()
        anomaly.byteswap()
    files = {'sst.bin': temperature.tobytes(), 'anomaly.bin': anomaly.tobytes()}
    for name, data in files.items():
        (arguments.output / name).write_bytes(data)
    manifest = {
        'id': DATASET_ID, 'version': 1, 'kind': 'raster', 'count': count,
        'bbox': [-161, 19, -109, 61], 'crs': 'EPSG:4326',
        'raster': {'file': 'sst.bin', 'encoding': 'float32-bin', 'dtype': 'float32',
                   'width': WIDTH, 'height': HEIGHT, 'depth': DEPTH,
                   'layout': 'cell-major [north-first row][west-first column][month]',
                   'bounds': [-161, 19, -109, 61], 'cellSizeDegrees': 2,
                   'noData': 'NaN', 'unit': 'degrees Celsius'},
        'properties': {
            'dates': DATES, 'period': 12, 'anomalyFile': 'anomaly.bin',
            'anomalyReference': 'NOAA supplied monthly anomalies relative to 1971-2000 climatology',
            'source': 'NOAA NCEI ERSSTv5 via NOAA CoastWatch ERDDAP; bounded monthly SST and anomaly query with source and metadata hashes.',
            'sourceVersion': 'ERSSTv5', 'sourceUrl': SOURCE_URL, 'metadataUrl': METADATA_URL,
            'sourceSha256': checksum(source), 'metadataSha256': checksum(metadata),
            'checksums': {name: checksum(data) for name, data in files.items()},
            'files': {name: {'bytes': len(data), 'sha256': checksum(data)} for name, data in files.items()},
            'license': 'No constraints on data access or use; free redistribution (NOAA ERDDAP metadata).',
            'attribution': 'NOAA National Centers for Environmental Information; NOAA CoastWatch ERDDAP',
            'doi': 'https://doi.org/10.7289/V5T72FNM',
            'method': 'Exact 2 degree grid subset; row order reversed to north-first; no interpolation or invented values.',
            'limitations': [
                'Monthly reconstructed SST from in-situ observations, not satellite pixels or raw buoy readings.',
                'Regular calendar-month slots differ in elapsed days; seasonal period is 12 months.',
                '2 degree cells do not resolve nearshore or daily marine heatwave events.',
                'Land and unavailable values remain NaN; no spatial or temporal gap filling.',
                'An offline single-break model is descriptive and is not causal attribution or operational detection.'
            ]
        }
    }
    (arguments.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (arguments.output / 'source-metadata.csv').write_bytes(metadata)
    print(json.dumps({'cells': WIDTH * HEIGHT, 'months': DEPTH, 'sourceRows': len(admitted),
                      'payloadBytes': sum(len(data) for data in files.values()),
                      'sourceSha256': checksum(source)}, indent=2))


if __name__ == '__main__':
    build()
