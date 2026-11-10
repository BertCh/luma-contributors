# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""External reference for the RTS-08 HEALPix nested point-to-cell kernel: healpix-geo 0.4.1.

healpix-geo (Rust over cdshealpix, Apache-2.0 / MIT) `nested.lonlat_to_healpix` on the sphere
(`ellipsoid='sphere'`), evaluated in binary64 on longitude/latitude values rounded to float32 first,
the data the GPU kernel reads. The package needs `marray` (pure Python); both were installed with
`pip install --target <dir> --no-deps`, so no environment was created.

Run: RTS08_ORACLE_SITE=<dir with healpix_geo and marray> <venv>/bin/python -I \
  modules/analysis/test/oracles/rts08-healpix-geo.py
"""
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

for entry in os.environ['RTS08_ORACLE_SITE'].split(os.pathsep):
    sys.path.append(entry)

import numpy as np  # noqa: E402
import healpix_geo.nested as nested  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'fixtures' / 'rts08-healpix-geo.json'
DEPTHS = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 29]


def to_float32(values):
    return np.asarray(values, dtype=np.float64).astype(np.float32).astype(np.float64)


def main():
    generator = np.random.default_rng(20261109)
    count = 3000
    longitude = generator.uniform(-180, 180, count)
    latitude = np.degrees(np.arcsin(generator.uniform(-1, 1, count)))
    # Probes: poles, equator, the antimeridian and meridian, the equatorial-polar boundary
    # (asin(2/3) = 41.8103148957786 degrees), face corners and 45 degree longitudes.
    boundary = float(np.degrees(np.arcsin(2 / 3)))
    probes = [
        (0, 0), (180, 0), (-180, 0), (90, 0), (-90, 0), (45, 0), (0, 90), (0, -90), (123.4, 90),
        (123.4, -90), (0, boundary), (0, -boundary), (45, boundary), (90, 41.81), (90, 41.82),
        (180, 41.81), (-179.99, -41.82), (45, 19.47122063449069), (135, -19.47122063449069),
        (0.0001, 0.0001), (359.9999, 0.0001), (22.5, 3.0), (67.5, -10.0)
    ]
    longitude = np.concatenate([longitude, [p[0] for p in probes]])
    latitude = np.concatenate([latitude, [p[1] for p in probes]])
    longitude = to_float32(longitude)
    latitude = to_float32(latitude)
    cells = {}
    for depth in DEPTHS:
        indexes = nested.lonlat_to_healpix(longitude, latitude, depth)
        cells[str(depth)] = [format(int(index), 'x') for index in indexes]
    fixture = {
        'description': 'healpix-geo nested.lonlat_to_healpix on the sphere for float32-rounded longitude/latitude.',
        'versions': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'healpix-geo': metadata.version('healpix-geo'),
            'marray': metadata.version('marray')
        },
        'longitude': longitude.tolist(),
        'latitude': latitude.tolist(),
        'depths': DEPTHS,
        'nestedHex': cells
    }
    OUTPUT.write_text(json.dumps(fixture) + '\n')
    print(f'wrote {OUTPUT}')


if __name__ == '__main__':
    main()
