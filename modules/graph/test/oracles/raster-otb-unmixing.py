# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Extract OTB 9.1.1 published UCLS golden data; this script does NOT execute OTB.

Requires numpy==2.2.6, tifffile==2026.10.15, imagecodecs==2026.3.6.
The three original LFS objects are hash verified before TIFF decoding. Inputs are
encoded as little-endian float32 to match GPU upload; expected abundances remain
upstream float64 values. No bands, pixels, or endmembers are dropped.
"""
import base64
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

import imagecodecs
import numpy as np
import tifffile

COMMIT = '154726b379fdb39b481ee8fd793fbd3d584884e6'
SOURCES = {
    'values': ('Data/Input/Hyperspectral/synthetic/hsi_cube.tif',
               '514951a85a6887e2623aac6811f01c23e21bee36f8c609be534effac70fcf177'),
    'endmembers': ('Data/Input/Hyperspectral/synthetic/endmembers.tif',
                   '2341b27e9e47cb8de7b5a5a9e4139264b6d8150085b3b15b773ac4dbf6095ca6'),
    'expectedAbundances': ('Data/Baseline/OTB/Images/apTvHyHyperspectralUnmixing_UCLS.tif',
                          '8e9a2bd8f7b406a2145f2e39017b6f54c0e09507e0e1f0f280aacb5fefb752d1')
}
arrays = {}
for name, (relative_path, expected_hash) in SOURCES.items():
    url = f'https://media.githubusercontent.com/media/orfeotoolbox/OTB/{COMMIT}/{relative_path}'
    data = urlopen(url).read()
    assert hashlib.sha256(data).hexdigest() == expected_hash
    arrays[name] = tifffile.imread(io.BytesIO(data))
assert arrays['values'].shape == (16, 16, 224)
assert arrays['endmembers'].shape == (1, 5, 224)
assert arrays['expectedAbundances'].shape == (16, 16, 5)
fixture = {
    'provenance': {
        'reference': 'OTB 9.1.1 published apTvHyHyperspectralUnmixing_UCLS baseline',
        'commit': COMMIT,
        'execution': 'Downloaded upstream golden output; OTB was not executed locally.',
        'upstreamTest': f'https://github.com/orfeotoolbox/OTB/blob/{COMMIT}/Modules/Hyperspectral/Applications/test/CMakeLists.txt',
        'command': 'HyperspectralUnmixing -in hsi_cube.tif -ie endmembers.tif -out output.tif double -ua ucls',
        'sources': {name: {'path': source[0], 'sha256': source[1]} for name, source in SOURCES.items()},
        'encoding': 'Inputs are complete TIFF arrays cast to little-endian float32 then base64; abundances retain upstream float64.',
        'license': 'Apache-2.0; see ../upstream-golden/OTB/LICENSE and NOTICE',
        'decoderVersions': {'numpy': np.__version__, 'tifffile': tifffile.__version__, 'imagecodecs': imagecodecs.__version__}
    },
    'cellCount': 256,
    'bandCount': 224,
    'endmemberCount': 5,
    'values': base64.b64encode(arrays['values'].astype('<f4').tobytes()).decode(),
    'endmembers': base64.b64encode(arrays['endmembers'].astype('<f4').tobytes()).decode(),
    'expectedAbundances': arrays['expectedAbundances'].reshape(-1).tolist()
}
Path(__file__).with_name('fixtures').joinpath('raster-otb-unmixing.json').write_text(json.dumps(fixture, indent=2) + '\n')
# Preserve upstream license and attribution verbatim alongside the derived fixture.
notice_directory = Path(__file__).with_name('upstream-golden') / 'OTB'
notice_directory.mkdir(exist_ok=True)
for name in ['LICENSE', 'NOTICE']:
    (notice_directory / name).write_bytes(urlopen(f'https://raw.githubusercontent.com/orfeotoolbox/OTB/{COMMIT}/{name}').read())
