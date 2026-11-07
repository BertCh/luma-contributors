# chicago-population-support

Hyde Park (Chicago community area 41) source tracts and a 64 × 64, 500 m equal-area WorldCover
support grid. Source tracts are the complete whole-polygons selected by the `chicago-tracts`
dataset's `communityArea` field; the 2020 Census geography carries CDC/ATSDR SVI 2022 counts
(ACS 2018–2022 five-year estimates). The raster is ESA WorldCover 2021 v200 class data from
`ESA_WorldCover_10m_2021_v200_N39W090_Map.tif`.

Licence: Census and SVI data are public domain. WorldCover is CC BY 4.0. Credit: © ESA WorldCover
project 2021 / Contains modified Copernicus Sentinel data 2021 processed by WorldCover consortium.
Citation: Zanaga et al. (2022), <https://doi.org/10.5281/zenodo.7254221>. Product manual:
<https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf>.

Run with the isolated raster environment:

```sh
/tmp/showcase-statistics-raster-env/bin/python scripts/data/chicago-population-support/build.py
```

The builder uses GDAL HTTP range reads, verifies the known source ETag, preserves the exact native
classification window and its affine transform/CRS/checksum, then converts classes 50 and 10 to
separate indicator bands before area averaging onto the analysis grid. It never averages class
codes. It also verifies the 12 sorted GEOIDs, whole-tract bounds, and population sum against the
existing `chicago-tracts` bundle. A source/network, ETag, or tract mismatch is a hard failure; no
substitute land-cover raster is generated.
