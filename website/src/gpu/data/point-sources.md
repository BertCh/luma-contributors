# Station observations and earthquake epicenters

The point scenes ship local extracts of the showcase snapshots. Regenerate all three assets with `node website/scripts/extract-point-observations.mjs` from the repository root.

## Rain gauges

`helene-rain.json` retains 6,135 southeastern US stations reporting daily precipitation for September 27, 2024, during Hurricane Helene. Each row holds longitude, latitude and two precipitation totals in tenths of a millimeter for September 27 and September 26. The earlier day uses 65535 for a missing measurement. The showcase extract excludes quality-flagged observations. The Carolinas-centered window used for the flagship scenes spans 85.3°W to 78°W and 32°N to 36.7°N. It has 1,654 stations reporting on both days. Daily observing times vary by station. Credit NOAA National Centers for Environmental Information, GHCN-Daily, US public-domain data as recorded in the showcase catalog. [NOAA GHCN-Daily](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily) · [Showcase manifest](../../../../examples/deck/spatial-analysis-showcase/public/data/ghcn-stations/manifest.json)

The IDW scene interpolates millimeters at 64 observed sites. Local Moran and Gi* use the measured values at 720 sites. Gi* switches between observation days, Local Moran optionally adds simulated measurement noise, and the tessellation scene allows a hypothetical station relocation. Shared run inputs normalize current-day totals by the maximum in the source extract, 343.4 mm. Some runs apply seeded noise to these normalized values or construct modeled variables at the measured sites.

## Earthquakes around Japan

`japan-earthquakes.json` holds 3,930 earthquake-type records of magnitude 4 or greater from the archived ComCat selection for January 1, 2020 through December 30, 2024, within 126–146°E and 28–42°N. Each row contains longitude, latitude, magnitude in hundredths, and whole days elapsed since the archive origin (00:28:20 UTC on January 1, 2020). This fixed subset inherits the archive's catalog coverage. Credit USGS ComCat, via poopdeck.gl, US public-domain data as recorded in the showcase catalog. [USGS ComCat](https://earthquake.usgs.gov/data/comcat/) · [Showcase manifest](../../../../examples/deck/spatial-analysis-showcase/public/data/poopdeck-earthquakes/manifest.json)

H3, S2 and Quadbin runs index earthquake longitude/latitude directly in the Honshu window, 140–144.4°E and 36.1–38.6°N. It contains 700 events. A local geographic camera displays decoded boundaries at the same positions as the input epicenters. Cell table comparison can model an eastward displacement, and cell set outline can model a change of spread.

Cell topology and cap covers draw hypothetical neighborhoods around three actual epicenters. Grid paths connect pairs of observed epicenters selected near geographic targets. Compaction uses the largest event in the window, magnitude 7.3 on March 16, 2022, as the center of a disk of H3 cells with a simulated completeness mask. The H3 and S2 measure and identity examples use occupied event cells as their study area.

The density scene samples 800 events. DBSCAN and K-means sample 820 events. Density measures event counts per normalized map area. In grouped run inputs, longitude quantiles define illustrative region labels. The scatter control can preferentially draw from half-degree cells with eight or fewer events. Network links and other scenario attributes constructed from these positions are modeled.

`japan-coast.json` holds 1:110 million coastline segments with at least one endpoint in the earthquake window. Made with Natural Earth, public domain. [Natural Earth terms](https://www.naturalearthdata.com/about/terms-of-use/) · [Showcase coastline](../../../../examples/deck/spatial-analysis-showcase/public/data/natural-earth/ne_110m_coastline.geojson)

## Encoding and display

Station and event rows occupy eight bytes each in the Base64 field of their JSON asset. The first two unsigned 16-bit little-endian values encode longitude and latitude as fractions of the snapshot bounds, rounded to 1/10,000 of each axis extent. The other two fields hold source measurements. The scenes use a seeded partial shuffle to select rows without replacement within each pass. Coordinates are projected to local east/north using the cosine of the window's midpoint latitude and a single display scale on both axes. The algorithm radius and bandwidth controls use scene units. The Japanese coastline shares the earthquake projection.
