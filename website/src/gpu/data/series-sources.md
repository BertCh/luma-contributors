# Lightning, lidar, departures and workplace records

These small extracts are inputs to the website's GPU operations. The images are captures of those operations. The browser reads the local fixtures, so opening a card does not call a data service.

## Midwest lightning

NOAA GOES-R Geostationary Lightning Mapper Level 2 flashes cover an 18-hour period beginning May 21, 2024 at 12:00 UTC. We use the committed `poopdeck-goes-glm-lightning` showcase sample. The website keeps 4,096 flashes sampled without replacement inside 99°–83° W, 36°–46° N. `midwest-flashes.b64` stores thousandths of a degree, elapsed whole seconds, optical energy in whole femtojoules and flash footprint in whole square kilometres. This is a sample of detections, so counts describe the fixture. The original NOAA data are public domain. Credit NOAA GOES-R GLM, via the poopdeck.gl archive. [NOAA GLM instrument](https://www.nesdis.noaa.gov/our-satellites/currently-flying/goes-east-west/geostationary-lightning-mapper-glm) · [Showcase archive manifest](https://tiles.poopdeck.gl/data/goes-glm-lightning/manifest.json)

Time windows and temporal reductions use the actual elapsed times. Temporal reductions group detections into twelve longitude bands. Distribution previews use energy in units of 10 fJ or fJ. Spatial and space-time variograms use log-transformed energy and area. Point-pattern controls can replace a share of the flashes with uniformly sampled points to explore complete spatial randomness. The default shows measured positions. The marked-point run splits flashes into two types by footprint area and uses energy as the mark. Controls for spread and nugget noise change the observations for comparison.

## King County, Washington lidar

The source is USGS 3DEP King County 2021 tile 2282, acquired April 1–24, 2021. The showcase `lidar-ground` fixture already thins original returns to one per occupied 0.5 m cell. From it we take a 128 m by 72 m window in the western part of the tile's showcase crop and sample 9,000 returns without replacement. The long axis points north, the short axis west. `king-county.b64` records local horizontal offsets and NAVD88 elevation in centimetres and retains the original LAS classification. The 9,000 returns include 3,142 class 2 ground returns. Credit U.S. Geological Survey, 3D Elevation Program, King County 2021. USGS 3DEP products are available without use restrictions. [USGS 3DEP](https://www.usgs.gov/3d-elevation-program) · [Original point cloud](https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/WA_KingCounty_2021_B21/WA_KingCo_1_2021/LAZ/USGS_LPC_WA_KingCounty_2021_B21_King_2282.laz)

All eleven point-cloud previews use these returns and preserve the relative scale of the axes. The ICP comparison rotates and translates a copy of the scan. The change comparison imposes a controlled slump on a second subset of returns from the same acquisition. The local-origin exercise moves and scales the cloud to study numerical precision. The outlier and coordinate-jitter controls add simulated perturbations.

## Atlanta departures

`atlantadepartures.json` contains 4,000 actual departure times sampled without replacement from noncancelled July 2023 reporting-carrier flights originating at Atlanta (ATL). Times are in local minutes relative to July 1 midnight, calculated from the scheduled date and departure time plus the reported departure delay. The GPU calendar uses absolute timestamps and a selectable UTC offset. Atlanta used UTC−4 that month. Credit U.S. Department of Transportation, Bureau of Transportation Statistics, Reporting Carrier On-Time Performance. The extract contains public-domain government transportation statistics. [BTS departure statistics](https://www.transtats.bts.gov/ONTIME/) · [July 2023 source ZIP](https://www.transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2023_7.zip)

## Chicago workplace blocks

Group statistics, inequality, key joins and mask transfers use 1,528 positive-job workplace blocks in 64 Chicago tracts. Each row has its official Census tract key, local block position and published 2021 LODES workplace job count. The inequality indices describe the concentration of jobs among workplace blocks within each tract. The composite score uses published tract estimates of poverty share, unemployment, older population share and population size. See [area-sources.md](./area-sources.md) for the extraction, credit and terms.

## Reproduction

From the repository root, run the following to rebuild the lightning and lidar extracts from the checked-in showcase fixtures. Passing the BTS archive as an argument also rebuilds the departure times. The BTS ZIP used here has SHA-256 `b1b73635dcb2add363bb001d95c3f951e0ce4fc32130cbc76b565a531ced2146`.

```sh
python3 website/scripts/extract-series-fixtures.py /path/to/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2023_7.zip
```
