# Network and trajectory demo data

These website-only extracts are U.S. federal public data. The extraction script writes JSON with integer-meter east/north coordinates. The renderer fits both map axes at the same scale.

| Fixture | Source and selection | Demo interpretation |
| --- | --- | --- |
| `sf-market-streets.json` | U.S. Census Bureau, 2024 TIGER/Line All Lines, San Francisco County. Secondary and local roads around Market Street, joined by the Census node IDs. The extract contains 700 connected nodes and 971 unique undirected edges. It also includes the 16 longest joined named streets in the window for line noding. [County shapefile](https://www2.census.gov/geo/tiger/TIGER2024/EDGES/tl_2024_06075_edges.zip) · [TIGER/Line documentation](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) | Small runs select a connected prefix. Intersections use straight, bidirectional links. Costs use displayed link length. Congestion, closures, facilities, demand and positioning error are modeled scenario inputs. The noding example resamples the joined roads to nine vertices each. |
| `sf-bay-ais.json` | U.S. Coast Guard, NOAA Office for Coastal Management and BOEM, Marine Cadastre AIS. Forty distinct vessels' contiguous in-window legs from January 9, 2023 UTC. Selected from the checked-in `poopdeck-ais-us` showcase archive, favoring legs covering some distance with multiple reports. [NOAA source](https://coast.noaa.gov/digitalcoast/data/vesseltraffic.html) · [2023 metadata](https://www.fisheries.noaa.gov/inport/item/70162) · [archive manifest](https://tiles.poopdeck.gl/data/ais-all-us/manifest.json) | Observed positions and UTC offsets are projected and resampled to each operation's fix budget. The trajectory runs compress the shared 24-hour clock into 600 playback seconds. Track quality maps the day to 40 time units and injects scenario glitches. Similarity compares legs ordered by origin longitude. Noise and zones are illustrative. |
| `chicago-commutes.json` | U.S. Census Bureau LEHD, Illinois LODES8 OD main file JT00, 2021, via the checked-in `chicago-lodes-od` tract aggregation. Includes the 1,200 largest inter-tract job flows within a central Chicago bounding box and 226 tract locations. Positions use Census 2021 Illinois Gazetteer internal points joined on GEOID. [LEHD data](https://lehd.ces.census.gov/data/) · [Illinois Gazetteer](https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2021_Gazetteer/2021_gaz_tracts_17.txt) | Aggregation sums jobs. Bundling shows the strongest 140 origin–destination links. Adjacency counts tract links and can group rows into north-to-south bands. Louvain uses the busiest 32 tracts and sums the two directions. Counts refer to jobs assigned to home and workplace tracts. |

## Reproduction

Use Python with `pyshp`. Download the San Francisco ZIP and Illinois Gazetteer linked above and run the following command from the repository root.

```sh
python website/scripts/data/extract-networks.py tl_2024_06075_edges.zip 2021_gaz_tracts_17.txt
```

The archive SHA-256 is `e5e5484038e69ad33d6fd41c3ca71ed9516281e850f05be6e2c232c8bc55da07`. The Gazetteer SHA-256 is `c8cc8d5b162cf74a2a57c7d7099414c12ba95d03aa86767f4bcdbea5ac19131a`. The script also reads AIS and LODES columns already in the showcase. The generated JSON retains the source data's public-domain status.

## Midwest datum corrections

The datum-grid-transform scene uses NOAA National Geodetic Survey's public-domain NADCON grid and U.S. Census Bureau 2022 state boundaries. The bounds are 96° W to 80° W and 35.5° N to 43° N. Each of the 65 × 31 nodes retains its native quarter-degree location. The extract rounds the east and north corrections to the nearest milliarcsecond. NOAA's longitude file uses west-positive offsets, which the extractor converts to east-positive for the GPU. [NOAA source ZIP](https://www.ngs.noaa.gov/PC_PROD/NADCON/NADCON.zip) · [PROJ source and license](https://cdn.proj.org/us_noaa_README.txt) · [Census cartographic boundaries](https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html)

Run `python website/scripts/data/extract-datum-grid.py NADCON.zip` from the root to regenerate `midwest-nadcon.json`. The transform evaluates a lattice of test locations in longitude and latitude degrees. Tip color uses the approximate horizontal displacement in meters. The lines drawn from the test locations to the tips magnify that displacement 1,500 times by default. The magnification slider changes drawing geometry alone. Direction and test longitude change the transformation itself.
