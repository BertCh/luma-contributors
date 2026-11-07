# Dixie spectral evidence

Rebuild with Python 3, NumPy and Rasterio:

```sh
python scripts/data/dixie-spectral/build.py
```

Run from the showcase directory. Optional `--cache` and `--output` paths make repeatability checks independent of shipped files. The default source cache is `/tmp/showcase-dixie-spectral`.

The script fetches two pinned Earth Search STAC items, then only the native COG windows needed for the exact 15 km UTM 10N rectangle: easting 662800–677800, northing 4441040–4456040. Source dates are 2021-07-13 and 2021-09-21. July 13 is the fire ignition date, not a pre-season climatology.

Six bands (blue, green, red, NIR, SWIR1, SWIR2) are averaged within aligned 60 m blocks. Native 10 m bands contribute 36 samples; native 20 m bands contribute nine. The pinned products precede processing baseline 04.00 and use scale 0.0001, offset zero. The builder asserts these calibration assumptions. Outputs are calibrated unitless float32 reflectance, little-endian, 250 rows × 250 columns, north row first, with NaN for invalid cells. No reprojection or gap filling is performed.

`validity_before.bin` and `validity_after.bin` are uint8 masks (1 valid). Every native 20 m SCL pixel must be class 4, 5 or 6, and every native sample in all six spectral bands must be nonzero. `land_valid_before.bin` and `land_valid_after.bin` additionally require SCL 4 or 5 only; intersect both to exclude water and invalid support from change topology. These strict masks reduce contamination but do not constitute manual cloud validation.

The manifest records source asset URLs, full pinned STAC metadata hashes, calibration, bounds, per-date support, and SHA-256 checksums for every derived binary. Full source STAC JSON accompanies the binaries. Native sample windows remain in the cache for reproducible extraction; original satellite assets are not republished.

Data permission and method sources:

- [Copernicus Sentinel legal notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice): free, full and open use, adaptation and redistribution; attribution preserved as “Contains modified Copernicus Sentinel data 2021”.
- [AWS Sentinel-2 L2A archive](https://registry.opendata.aws/sentinel-2-l2a-cogs/): public source COGs.
- [ESA Sentinel-2 processing documentation](https://sentiwiki.copernicus.eu/web/s2-processing): L2A scene classification, reflectance quantification and processing-baseline offsets.

Indices are spectral proxies, not validated burn severity or land-cover classes. SWIR1-based NDWI adapts the moisture contrast to Sentinel's 1.61 µm band; it does not reproduce Gao's original 1.24 µm measurement. The broad geographic bounding rectangle is for navigation; computational geometry uses the exact EPSG:32610 grid.
