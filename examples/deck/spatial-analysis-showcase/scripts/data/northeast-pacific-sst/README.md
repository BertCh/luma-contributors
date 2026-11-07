# Northeast Pacific monthly SST

NOAA ERSSTv5 reconstructed monthly sea-surface temperature and NOAA-provided temperature anomalies,
January 2012–December 2017. This is a genuine calendar-month raster sequence suitable for temporal
smoothing and bounded seasonal-model demonstrations. It is not satellite imagery or a collection of
raw buoy readings.

## Rebuild

```sh
python3 examples/deck/spatial-analysis-showcase/scripts/data/northeast-pacific-sst/build.py
```

Only the Python standard library is required. The builder downloads a bounded 2.1 MB CSV subset
from NOAA CoastWatch ERDDAP, caches raw responses under `/tmp/showcase-northeast-pacific-sst`, and
writes the two float32 arrays, manifest, and source metadata snapshot. Override `--cache` and
`--output` if needed. A cached response makes regeneration independent of later archive revisions;
the source SHA-256 is recorded in the manifest. Removing the cache intentionally fetches current
archive values, which may have been revised.

## Source and reuse

- [NOAA ERSSTv5 archive record and DOI](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc%3AC00927)
- [NOAA ERDDAP source metadata](https://coastwatch.pfeg.noaa.gov/erddap/info/nceiErsstv5_LonPM180/index.html)
- [NOAA ERSST product description](https://www.ncei.noaa.gov/products/extended-reconstructed-sst)
- [NOAA PMEL: 2014 Northeast Pacific warm anomaly](https://www2.pmel.noaa.gov/news-and-media/highlights/causes-and-impacts-2014-warm-anomaly-ne-pacific)

Source access verified 2026-10-25. NOAA's dataset metadata permits free use and redistribution with
no access or use constraints; its accuracy and fitness disclaimers are preserved in
`source-metadata.csv`. Credit NOAA National Centers for Environmental Information and NOAA
CoastWatch ERDDAP. Dataset DOI: `10.7289/V5T72FNM`. The pinned source version is v5; the example does
not claim it is NOAA's latest product version.

## Exact data contract

| Property | Value |
| --- | --- |
| Grid | 26 columns × 21 rows, original 2° spacing |
| Cell centers | Longitude −160° through −110°; latitude 60° through 20° |
| Footprint | `[−161, 19, −109, 61]` degrees; centers expanded by half a cell |
| CRS / orientation | EPSG:4326; row 0 is north; columns west to east |
| Time | 72 calendar-month slots, January 2012 through December 2017; source timestamps are the 15th |
| Layout | Little-endian float32 `[row][column][month]`, time fastest |
| `sst.bin` | °C, reconstructed SST |
| `anomaly.bin` | °C, NOAA anomalies from the 1971–2000 monthly climatology |
| Missing | IEEE NaN, preserved from source; never replaced with zero |
| Runtime payload | 314,496 bytes for both arrays |
| Processing | Exact subset, north-first row reversal, float32 encoding; no interpolation |

The builder verifies 39,312 unique coordinate/month records, expected units, exact grid positions,
all 72 dates and the published valid-value ranges. Both source and output checksums are embedded
in the manifest. The archive's spatial reconstruction fills observational coverage statistically;
that is a property of ERSST, not gap filling introduced by this builder.

## Interpretation limits

Calendar-month indexing is regular for a period-12 seasonal model even though months have different
numbers of days. The coarse grid smooths local and coastal detail. A fitted break is a descriptive
single level-shift model over six years; it cannot establish the cause, onset date, duration, or
severity category of a marine heatwave. NOAA's supplied climatological anomaly is an external
reference, distinct from the residual of the showcase's fitted seasonal model. Do not identify
those two quantities as the same anomaly.
