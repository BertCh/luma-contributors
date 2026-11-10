# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Extract a native-resolution Midwest NAD27-to-NAD83 grid from NOAA's NADCON.zip."""

import io
import json
import struct
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WEST, SOUTH, EAST, NORTH = -96, 35.5, -80, 43
SPACING = 0.25


def correction_band(jar, filename, sign):
    raw = jar.read(filename)
    columns, rows, _, west, dx, south, dy, _ = struct.unpack_from('<3i5f', raw, 64)
    assert columns == 273 and rows == 121 and dx == dy == SPACING
    values = []
    for row in range(round((SOUTH - south) / dy), round((NORTH - south) / dy) + 1):
        for col in range(round((WEST - west) / dx), round((EAST - west) / dx) + 1):
            # One full header row, then each data row starts with an unused word.
            offset = ((row + 1) * (columns + 1) + col + 1) * 4
            shift = struct.unpack_from('<f', raw, offset)[0]
            assert abs(shift) < 5
            values.append(round(sign * shift * 1000))
    return values


def state_boundaries():
    geo = json.loads((ROOT / 'examples/deck/spatial-analysis-showcase/public/data/us-states/states.geojson').read_text())
    lines = set()
    for feature in geo['features']:
        shape = feature['geometry']
        polygons = shape['coordinates'] if shape['type'] == 'MultiPolygon' else [shape['coordinates']]
        for polygon in polygons:
            for ring in polygon:
                for first, second in zip(ring, ring[1:]):
                    if (max(first[0], second[0]) < WEST or min(first[0], second[0]) > EAST or
                        max(first[1], second[1]) < SOUTH or min(first[1], second[1]) > NORTH):
                        continue
                    endpoints = sorted(tuple(round(value, 3) for value in point) for point in (first, second))
                    if endpoints[0] != endpoints[1]:
                        lines.add((*endpoints[0], *endpoints[1]))
    return sorted(lines)


with zipfile.ZipFile(sys.argv[1]) as package:
    member = next(name for name in package.namelist() if name.lower().endswith('/nadcon.jar') or name.lower() == 'nadcon.jar')
    with zipfile.ZipFile(io.BytesIO(package.read(member))) as jar:
        # NOAA .los uses positive west. GPUDatumGridTransform consumes positive east.
        east = correction_band(jar, 'grids/conus.los', -1)
        north = correction_band(jar, 'grids/conus.las', 1)

fixture = {'bounds': [WEST, SOUTH, EAST, NORTH], 'spacing': SPACING,
           'width': round((EAST - WEST) / SPACING) + 1, 'height': round((NORTH - SOUTH) / SPACING) + 1,
           'eastNorthMilliarcseconds': [n for pair in zip(east, north) for n in pair],
           'stateBoundaries': state_boundaries()}
(ROOT / 'website/src/gpu/data/midwest-nadcon.json').write_text(json.dumps(fixture, separators=(',', ':')) + '\n')
