# Terrain, imagery and weather samples

Four north-up rasters supply 97,280 bytes of measurements and forecast values to the website. Each live operation prepares its inputs from the same local fixture used by the snapshot harness. The app samples a centered crop to fit the requested grid while preserving the source cell proportions.

| File in `rasters/` | Contents | SHA-256 |
| --- | --- | --- |
| `grand-canyon.bin` | 128 × 80 little-endian uint16 elevations | `876d9c5a696f368d41219dddf8b678aa205ec74014f3955a3ed626e2b9bf954b` |
| `dixie-spectral.bin` | 6 × 64 × 40 little-endian uint16 reflectances | `20f9fa24ae47e8fef5df12ef823787bbba8d7ead2be2b5f65cb2649008ac48ec` |
| `dixie-ndvi.bin` | 16 × 64 × 40 uint8 NDVI values with validity | `44c25270f14557078689cb799eff5c3b54dc6f0e4a6d416a137e45397f98b8b9` |
| `helene-wind.bin` | 64 × 40 pairs of uint8 wind components | `e1f30907d849fd8942ac48de3dfd1052efe3cfb9f8192cc849a2864f6d82660b` |

## Grand Canyon elevation

The USGS 3D Elevation Program publishes one-arc-second National Map elevations in the public domain. This crop comes directly from tile `n37w113`, edition 2024-06-14. Area averaging in UTM zone 12N produces 70 m square cells centered on 36.104° N, 112.09° W near Phantom Ranch. The window measures 8.96 km by 5.6 km and ranges from 727 to 2,134 m above sea level. Elevations are rounded to whole meters. Relief, horizons, profiles and slopes use metric heights and spacing. The normalized scalar rasters use a zero-to-one display range. Hydrology runs with world-coordinate outputs convert both height and distance by the same factor. [USGS product and public-domain terms](https://data.usgs.gov/datacatalog/data/USGS%3A35f9c4d4-b113-4c8d-8691-47c428c29a5b) · [Source GeoTIFF](https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/historical/n37w113/USGS_1_n37w113_20240614.tif)

## Feather River surface reflectance

Contains modified Copernicus Sentinel data 2021. Six surface-reflectance bands come from Sentinel-2 L2A scene `S2B_10TFK_20210713_0_L2A` in the Feather River watershed, California. The local `dixie-spectral` showcase preserves the Earth Search scene metadata and native-validity masks. Each 180 m output cell averages its valid 60 m inputs. Every output cell has a measurement. The integers encode blue B02, green B03, red B04, NIR B08, SWIR B11 and SWIR B12 in band-major order at 1/4,095 reflectance units. Indices use the measured bands. The material fractions and threshold-based cover labels are model estimates. [Source scenes](https://registry.opendata.aws/sentinel-2-l2a-cogs/) · [Copernicus terms](https://dataspace.copernicus.eu/terms-and-conditions)

## Dixie Fire vegetation time series

Contains modified Copernicus Sentinel data 2017–2024. The local `ndvi-timeseries` showcase manifest records the sixteen acquisition dates and individual scenes. The crop averages valid 100 m pixels into 300 m cells and retains zero for missing observations. Positive integers decode as `value / 127.5 - 1`. The temporal composite and change statistics use twelve acquisitions from September 2017 through August 2022. July 13 and September 11, 2021 bracket the fire-related change. Additional cloud masking and the adjustable post-fire change multiplier are scenarios. The seasonal-break run models a regular monthly cycle using observed July and September 2021 vegetation levels, since the archival acquisitions are irregularly spaced. [Sentinel-2 archive](https://registry.opendata.aws/sentinel-2-l2a-cogs/) · [Copernicus terms](https://dataspace.copernicus.eu/terms-and-conditions)

## Hurricane Helene wind

NOAA/NCEP Global Forecast System, a public-domain US government product. The forecast was initialized on September 26, 2024 at 12 UTC. The 14-hour forecast is valid September 27 at 02 UTC. It supplies the 10 m eastward and northward wind on a 0.25° grid over the Gulf of Mexico. The local `gfs-wind` showcase retains the forecast filenames and component calibration. The 64 × 40 crop spans grid centers from 93° W to 77.25° W and from 33° N to 23.25° N. Display velocities share one time scale, with longitude spacing adjusted at the representative latitude of 28° N. [NOAA GFS source](https://registry.opendata.aws/noaa-gfs-bdp-pds/)

## Other inputs and scenarios

Raster coverage and zonal summaries use Chicago tract outlines and a nearest-tract grid of relative poverty estimates. Raster join assigns Chicago workplace block jobs to the tract identified by the rasterized cell. The distance field uses NOAA rain gauge locations in the Carolinas. Solar position evaluates a global latitude/longitude grid and eighteen Natural Earth cities at the selected UTC time. Added pits, travel barriers, image shifts, experimental noise and parameter-controlled missing values are deliberate input scenarios. [Chicago tract preparation](./area-sources.md) · [NOAA GHCN](https://registry.opendata.aws/noaa-ghcn/) · [Natural Earth terms](https://www.naturalearthdata.com/about/terms-of-use/)

## Reproduction

Run from the repository root with Python, NumPy, rasterio and Pillow. The command downloads the pinned USGS tile if that cache path is empty. All other inputs come from the versioned `examples/deck/spatial-analysis-showcase/public/data` files.

```sh
python3 website/scripts/sample-raster-data.py /tmp/USGS_1_n37w113_20240614.tif
```
