# USGS 3DEP lidar-ground sample

`process.py` downloads the official USGS 3DEP King County 2021 tile `USGS_LPC_WA_KingCounty_2021_B21_King_2282.laz`, checks its SHA-256, clips a fixed 180 m square, selects one deterministic original return per occupied 0.5 m cell, converts horizontal coordinates from EPSG:6597 US survey feet to WGS 84 longitude/latitude, and converts NAVD88 Geoid18 elevation from US survey feet to metres. It does not synthesize or interpolate points.

The source contains 6,373,072 LAS 1.4 point-format 6 returns. The bundled sample contains 116,213 original records from 301,727 points in the crop. `public/data/lidar-ground/metadata.json` records the source checksum, source and output bounds, crop, sample rule, date range, CRS, units and retained class counts. The bundled manifest keeps positions, elevation, and LAS classifications as separate typed columns.

The USGS 3DEP point cloud collection is public domain as a U.S. Government work. Source references:

- Catalog record: <https://www.sciencebase.gov/catalog/item/619ce206d34eb622f694418b>
- Product metadata: <https://thor-f5.er.usgs.gov/ngtoc/metadata/waf/elevation/lidar_point_cloud/laz/WA_KingCo_1_2021/USGS_LPC_WA_KingCounty_2021_B21_King_2282.xml>
- USGS 3DEP catalog: <https://data.usgs.gov/datacatalog/data/USGS%3Ab7e353d2-325f-4fc6-8d95-01254705638a>

Rebuild with Python 3.11 or newer:

```sh
python3 -m venv /tmp/lidar-ground-venv
/tmp/lidar-ground-venv/bin/pip install laspy lazrs numpy pyproj
/tmp/lidar-ground-venv/bin/python scripts/data/lidar-ground/process.py
```

The downloaded 31 MiB LAZ stays in `/tmp`; only the approximately 2.7 MiB clipped columns ship with the example. Longitude and latitude use float64 so neighboring samples retain their sub-metre spacing.
