# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent NumPy mosaic/sample composition; Rasterio direct bilinear interior check.

Run with Python + numpy + rasterio. Values are unitless; grids use EPSG:3857 metres,
area pixels, source Affine(1,0,0,0,-1,2), destination translated +0.5 source pixels.
Rasterio's outer-edge clamping is deliberately excluded: strict missing positive-weight taps
and ordered overwrite including nodata are explicit composition contracts, not GDAL warp parity.
"""
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from affine import Affine

source = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype='float32')
transform = Affine(1, 0, 0, 0, -1, 2)
warped = np.full((2, 3), np.nan, dtype='float32')
reproject(source, warped, src_transform=transform, dst_transform=transform * Affine.translation(.5, 0),
          src_crs='EPSG:3857', dst_crs='EPSG:3857', resampling=Resampling.bilinear)

diagonal = np.full((1, 3), np.nan, dtype='float32')
reproject(source, diagonal, src_transform=transform, dst_transform=transform * Affine.translation(.5, .5),
          src_crs='EPSG:3857', dst_crs='EPSG:3857', resampling=Resampling.bilinear)

def sample(mosaic, renormalize):
    result = []
    for row in mosaic:
        for column in range(4):
            taps = row[column:column+2]
            valid = np.isfinite(taps)
            if (renormalize and valid.any()) or (len(taps) == 2 and valid.all()):
                result.append(float(taps[valid].mean()))
            else:
                result.append(None)
    return result

missing = source.copy()
missing[:, 2:] = np.nan
overlap = source.copy()
overlap[:, 1:3] = [[200, np.nan], [600, 700]]
seam = sample(source, False)
assert np.allclose(warped.flatten(), np.array(seam, dtype=object).reshape(2, 4)[:, :3].astype(float).flatten())
fixture = {
    'provenance': {'numpy': np.__version__, 'rasterio': rasterio.__version__, 'gdal': rasterio.__gdal_version__,
                   'reference': 'NumPy ordered mosaic + equal-weight sample composition; Rasterio direct interior bilinear'},
    'source': source.flatten().tolist(),
    'seamStrict': seam,
    'seamRenormalize': sample(source, True),
    'diagonalRasterio': diagonal.flatten().tolist(),
    'missingStrict': sample(missing, False),
    'missingRenormalize': sample(missing, True),
    'overlapStrict': sample(overlap, False),
    'overlapRenormalize': sample(overlap, True),
    'rasterioInterior': warped.flatten().tolist()
}
Path(__file__).with_suffix('.json').write_text(json.dumps(fixture, indent=2) + '\n')
