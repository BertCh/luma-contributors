# Chicago building footprints

Source: [City of Chicago Data Portal — Building Footprints](https://data.cityofchicago.org/Buildings/Building-Footprints-current-/hz9b-7nh8), dataset ID `syp8-uezg`.

The GeoJSON is queried directly from `https://data.cityofchicago.org/resource/syp8-uezg.geojson` using the fixed bbox `-87.65, 41.87, -87.62, 41.895` (west, south, east, north) and the source's `bldg_statu = 'ACTIVE'` filter. Rebuild with `node scripts/data/chicago-buildings/build.mjs`. The script fails if the source is unavailable, empty, or reaches its 10,000-row cap.

Attribution: City of Chicago Data Portal, Building Footprints. Review the [dataset terms of use](https://data.cityofchicago.org/terms-of-use) for reuse conditions. The payload SHA-256 is recorded in the manifest; the download timestamp changes on every build. Available source fields include `bldg_id`, `bldg_statu`, `stories`, `year_built`, and address components such as `f_add1` and `t_add1`. Story counts are incomplete and must not be interpreted as complete height coverage.
