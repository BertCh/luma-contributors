# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Prepare small website rasters. Run with numpy, rasterio and Pillow installed."""

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform, Resampling

WEBSITE = Path(__file__).resolve().parents[1]
SOURCE = WEBSITE.parent / "examples/deck/spatial-analysis-showcase/public/data"
OUTPUT = WEBSITE / "src/gpu/data/rasters"
DEM = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/historical/n37w113/USGS_1_n37w113_20240614.tif"


def prepare_dem(tile):
    with rasterio.open(tile) as dataset:
        east, north = transform("EPSG:4326", "EPSG:32612", [-112.09], [36.104])
        elevation = np.full((80, 128), np.nan, dtype="float32")
        reproject(rasterio.band(dataset, 1), elevation,
                  dst_transform=from_origin(east[0] - 4480, north[0] + 2800, 70, 70),
                  dst_crs="EPSG:32612", dst_nodata=np.nan, resampling=Resampling.average)
    assert np.all(np.isfinite(elevation)) and 0 < np.min(elevation) < np.max(elevation) < 3000
    np.rint(elevation).astype("<u2").tofile(OUTPUT / "grand-canyon.bin")


def prepare_spectral():
    # 192 x 120 source pixels in a 250 x 250 scene. Average 3 x 3 clear measurements per pixel.
    bands = []
    for band in ["blue", "green", "red", "nir", "swir16", "swir22"]:
        values = np.fromfile(SOURCE / f"dixie-spectral/{band}_before.bin", dtype="<f4")
        crop = values.reshape(250, 250)[65:185, 29:221]
        averaged = np.nanmean(crop.reshape(40, 3, 64, 3), axis=(1, 3))
        assert np.all(np.isfinite(averaged))
        bands.append(np.rint(np.clip(averaged, 0, 1) * 4095).astype("<u2"))
    np.stack(bands).tofile(OUTPUT / "dixie-spectral.bin")


def prepare_temporal():
    series = np.fromfile(SOURCE / "ndvi-timeseries/ndvi.bin", dtype=np.uint8).reshape(16, 256, 256)
    # Retain pre- and post-fire coverage, with 0 reserved for missing observations throughout.
    crop = series[:, 68:188, 32:224].astype("float32")
    cells = crop.reshape(16, 40, 3, 64, 3)
    total = cells.sum(axis=(2, 4))
    count = (cells > 0).sum(axis=(2, 4))
    average = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    np.rint(average).astype("uint8").tofile(OUTPUT / "dixie-ndvi.bin")


def prepare_wind():
    with Image.open(SOURCE / "gfs-wind/wind10m_14.png") as dataset:
        # Row 0 of the source is 50N, column 0 is 105W. Crop the Gulf of Mexico.
        np.asarray(dataset)[68:108, 48:112, :2].tofile(OUTPUT / "helene-wind.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tile", type=Path, help="USGS DEM download/cache path")
    tile = parser.parse_args().tile
    if not tile.exists():
        urlretrieve(DEM, tile)
    OUTPUT.mkdir(exist_ok=True, parents=True)
    prepare_dem(tile)
    prepare_spectral()
    prepare_temporal()
    prepare_wind()
