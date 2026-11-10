# Third-party data notice

This file lists the third-party datasets committed to this repository, with source, licence,
attribution and redistribution status. The website section links to preparation notes and
provider terms for the distributed extracts. The showcase and lens tables use the licence
information recorded in `examples/deck/spatial-analysis-showcase/src/data/datasets/*.dataset.ts`,
the dataset manifests, the READMEs under `examples/deck/spatial-analysis-showcase/scripts/data/*/`
and the lens data README. Where those records lack a licence the entry reads `UNVERIFIED`.

Nineteen showcase manifests have no `license`/`attribution` fields because oracle fixtures or tests
pin their SHA-256: `alps-dem`, `celestrak-ground-tracks`, `chicago-311-rats`, `chicago-crashes`,
`chicago-facilities`, `chicago-nature`, `chicago-places`, `chicago-roads`, `chicago-tracts`,
`ghcn-stations`, `grand-canyon-dem`, `gull-migration`, `ndvi-timeseries`, `openflights`,
`poopdeck-ais-us`, `poopdeck-earthquakes`, `poopdeck-ecco-currents`, `poopdeck-osm-nyc` and
`us-counties`. This file is the licence record for those datasets.

Source code ported from other projects is not listed here; it carries SPDX headers in the files
themselves (and, for example, `PARCELS-LICENSE.txt` beside the Parcels port).

## Website demo extracts

The private `website` workspace ships compact data extracts and WebP captures of GPU operations on
those inputs. The extracts and derived images retain their source terms. The website's **Demo data**
page publishes the provider credits, and each function page identifies the sources for its scene.
The [source registry](website/build/demo-data-sources.ts) maps the distributed assets to their
providers and terms. The following preparation notes identify the geographic windows, dates,
units, resampling and modeled inputs.

| Inputs | Preparation and attribution |
| --- | --- |
| Rain gauges and earthquakes | [Point observations](website/src/gpu/data/point-sources.md) |
| Tract boundaries, population, adjacency and building footprints | [Chicago tracts and Amsterdam buildings](website/src/gpu/data/area-sources.md) |
| DEM, spectral bands, vegetation indices and wind | [Terrain and imagery](website/src/gpu/data/terrain-sources.md) |
| Streets, vessel tracks, origin–destination flows and datum corrections | [Networks, trajectories and NADCON](website/src/gpu/data/network-sources.md) |
| Hydrography and city coordinates | [Geometry inputs](website/src/gpu/data/geometry-sources.md) |
| Events, panels and lidar returns | [Series, tables and point clouds](website/src/gpu/data/series-sources.md) |
| Regression, rates and space-time statistics | [Statistical inputs](website/src/gpu/data/statistics-sources.md) |
| Land cover, population and regional scenarios | [Recipe inputs](website/src/gpu/data/recipe-sources.md) |

The Hyde Park land-cover sample is adapted from ESA WorldCover v200 (2021).
© ESA WorldCover project 2021. Contains modified Copernicus Sentinel data (2021)
processed by the WorldCover consortium. The website crops, resamples and groups the classes.
[WorldCover terms](https://esa-worldcover.org/en/data-access) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The Overtoom building outlines and heights are a modified extract of 3DBAG.
© 3DBAG by tudelft3d and 3DGI. [3DBAG attribution](https://docs.3dbag.nl/en/copyright/) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Must resolve before merge

The following datasets carry non-commercial terms or OpenStreetMap / ODbL obligations that conflict
with, or need explicit handling in, an MIT-licensed repository that is published to npm and used
commercially. Each needs a decision (replace with a permissive source, remove from the package and
fetch on demand, or record explicit permission and share-alike handling) before this branch merges.

**Non-commercial**

- `chicago-nature` (showcase) and every derived copy: iNaturalist observations, each with its own
  licence (CC0, CC BY 4.0 or CC BY-NC 4.0). The CC BY-NC records cannot be redistributed commercially.
  Derived copies: lens data `f-chicago-nature.json`, `s2-chicago-north-nature.json`, and the nature
  counts inside `chicago-tracts` (showcase), `f-chicago-tracts.json`, `s1-chicago-tracts.json`
  and `s2-chicago-north-tracts.json`.
- `poopdeck-adsb-paths` (showcase): OpenSky Network ADS-B flights. The manifest records
  "research and non-commercial use with attribution"; the licence text could not be fetched when the
  entry was written. Treat as NON-COMMERCIAL.

**ODbL / OpenStreetMap-derived**

- `alps-context`: ODbL 1.0 (Open Database License)
- `chicago-gps-traces`: Synthetic; derived from OpenStreetMap, ODbL 1.0
- `chicago-parks`: ODbL 1.0 (Open Database License)
- `chicago-roads`: ODbL 1.0
- `chicago-walk`: ODbL 1.0
- `openflights`: Open Database License (ODbL) 1.0
- `poopdeck-bixi-rides`: ODbL 1.0 (OSRM routes on OpenStreetMap) and Creative Commons Attribution (BIXI Montreal trip history)
- `poopdeck-nyc-taxi`: NYC Open Data terms of use (NYC TLC trip records, no restrictions); route endpoints snapped by OSRM on OpenStreetMap data (ODbL)
- `poopdeck-nyc-taxi-paths`: NYC Open Data terms of use (NYC TLC trip records, no restrictions); OSRM routes on OpenStreetMap data (ODbL)
- `poopdeck-osm-nyc`: ODbL 1.0 (Open Database License)
- `us-airline-flows`: US government public domain (BTS); airport coordinates ODbL 1.0 (OpenFlights)

Lens data derived from the above (all under the lens data directory listed below): `s3-chicago-parks`,
`s3-chicago-roads`, `s3-openflights-ord`, `s2-chicago-north-parks`, `s2-chicago-north-roads`,
`n-chicago-loop-roads`, `n-chicago-loop-polylines`, `n-chicago-loop-traces`.

## Also review


- `alps-dem`: custom terms; attribution required, confirm redistribution. Mapterhorn terrain tiles built from swisstopo swissALTI3D (OGD, free use with attribution) and Copernicus GLO-30
- `alps-dem-wide`: custom terms; attribution required, confirm redistribution. Mapterhorn terrain tiles built from swisstopo swissALTI3D (OGD), Copernicus GLO-30 and Italian regional models (CC BY 4.0)
- `celestrak-ground-tracks`: custom terms; attribution required, confirm redistribution. CelesTrak element sets: free to use with attribution; SGP4 propagation by this project
- `chicago-311-rats`: custom terms; attribution required, confirm redistribution. City of Chicago Data Portal Terms of Use
- `chicago-boundary`: custom terms; attribution required, confirm redistribution. City of Chicago Data Portal Terms of Use; public domain (Natural Earth)
- `chicago-buildings`: custom terms; attribution required, confirm redistribution. City of Chicago Data Portal Terms of Use
- `chicago-community-areas`: custom terms; attribution required, confirm redistribution. City of Chicago Data Portal Terms of Use
- `chicago-crashes`: custom terms; attribution required, confirm redistribution. City of Chicago Data Portal Terms of Use
- `chicago-facilities`: custom terms; attribution required, confirm redistribution. City of Chicago Data Portal Terms of Use; hospitals from Overture Maps (CDLA-Permissive-2.0)
- `chicago-taxi-od`: custom terms; attribution required, confirm redistribution. City of Chicago Data Portal Terms of Use
- `cta-transit`: custom terms; attribution required, confirm redistribution. CTA Developer License Agreement and Terms of Use (derived data for transit-accessibility use; credit optional; no stand-alone resale)
- `grand-canyon-dem`: custom terms; attribution required, confirm redistribution. AWS Open Data Terrain Tiles (public bucket); underlying USGS 3DEP/NED is public domain, other sources require attribution per Tilezen list
- `meuse-validation`: GPL-2.0-or-later data; confirm redistribution inside an MIT repository. GPL-2.0-or-later (R package sp data)
- `ncovr-new-jersey`: no licence asserted by the source. Source data provided as-is with no warranties; no separate license asserted
- `northeast-pacific-sst`: custom terms; attribution required, confirm redistribution. NOAA: no constraints on data access or use; free redistribution
- `poopdeck-ecco-currents`: custom terms; attribution required, confirm redistribution. NASA Earth science data, open and free to use
- `sentinel-cloud-repair`: custom terms; attribution required, confirm redistribution. Copernicus Sentinel data legal notice; free, full and open access with attribution
- `swissimage-zermatt`: custom terms; attribution required, confirm redistribution. swisstopo Open Government Data (OGD), free use with attribution
- `wildfire-california-dem`: custom terms; attribution required, confirm redistribution. Terrain Tiles on AWS: public domain and open source data, attribution required

## Showcase datasets

Location: `examples/deck/spatial-analysis-showcase/public/data/<id>/`. Each directory has a
`manifest.json` carrying the same `license` and `attribution` strings as this table. Build
scripts and per-dataset READMEs are in `examples/deck/spatial-analysis-showcase/scripts/data/<id>/`.

| Dataset id | Source | Licence (as recorded) | Attribution text | Redistribution status |
| --- | --- | --- | --- | --- |
| ahk-rock-glacier | https://doi.org/10.11588/data/XHYB10 | CC BY 4.0 | Winiwarter, Anders, Zahs, Hämmerle and Höfle (2021), heiDATA, doi:10.11588/data/XHYB10; sample via 3DGeo Heidelberg py4dgeo-test-data (CC BY 4.0) | Attribution required; redistribution permitted |
| ais-vessels | https://github.com/ocm-marinecadastre/ais-vessel-traffic | CC0 1.0 Universal (public domain) | NOAA Office for Coastal Management / U.S. Coast Guard Navigation Center, Nationwide AIS 2024 | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| ais-zones | https://encdirect.noaa.gov/ | NOAA Coast Survey ENC data (public domain); approximate polygons hand-drawn | NOAA Coast Survey ENCDirect; approximate zones by the showcase authors | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| alps-context | https://www.openstreetmap.org/copyright | ODbL 1.0 (Open Database License) | © OpenStreetMap contributors (ODbL) | MUST RESOLVE BEFORE MERGE: ODbL / OpenStreetMap-derived (attribution and share-alike obligations) |
| alps-dem | https://mapterhorn.com/ | Mapterhorn terrain tiles built from swisstopo swissALTI3D (OGD, free use with attribution) and Copernicus GLO-30 | (c) swisstopo (swissALTI3D), (c) Mapterhorn; contains modified Copernicus data | REVIEW: custom terms; attribution required, confirm redistribution |
| alps-dem-wide | https://mapterhorn.com/attribution/ | Mapterhorn terrain tiles built from swisstopo swissALTI3D (OGD), Copernicus GLO-30 and Italian regional models (CC BY 4.0) | "(c) swisstopo (swissALTI3D), (c) Mapterhorn; contains modified Copernicus data (provided under COPERNICUS by the European Union and ESA); Regione Autonoma Valle d'Aosta and Regione Piemonte DTM (CC BY 4.0)" | REVIEW: custom terms; attribution required, confirm redistribution |
| bixi-flows | https://bixi.com/en/open-data/ | Creative Commons Attribution (BIXI Montreal trip history, as listed on the Montreal open data portal) | BIXI Montréal open data, trip history 2024 | Attribution required; redistribution permitted |
| celestrak-ground-tracks | https://celestrak.org/NORAD/elements/ | CelesTrak element sets: free to use with attribution; SGP4 propagation by this project | CelesTrak (celestrak.org, Dr T.S. Kelso), GP element sets; propagation with satellite.js (SGP4) | REVIEW: custom terms; attribution required, confirm redistribution |
| chicago-311-rats | https://data.cityofchicago.org/Service-Requests/311-Service-Requests/v6vf-nfxy | City of Chicago Data Portal Terms of Use | City of Chicago 311 | REVIEW: custom terms; attribution required, confirm redistribution |
| chicago-boundary | https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Community-Areas/igwz-8jzy | City of Chicago Data Portal Terms of Use; public domain (Natural Earth) | City of Chicago; Made with Natural Earth | REVIEW: custom terms; attribution required, confirm redistribution |
| chicago-buildings | https://data.cityofchicago.org/Buildings/Building-Footprints-current-/hz9b-7nh8 | City of Chicago Data Portal Terms of Use | City of Chicago Data Portal — Building Footprints | REVIEW: custom terms; attribution required, confirm redistribution |
| chicago-community-areas | https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Community-Areas/igwz-8jzy | City of Chicago Data Portal Terms of Use | City of Chicago | REVIEW: custom terms; attribution required, confirm redistribution |
| chicago-crashes | https://data.cityofchicago.org/resource/85ca-t3if | City of Chicago Data Portal Terms of Use | City of Chicago / Chicago Police Department | REVIEW: custom terms; attribution required, confirm redistribution |
| chicago-facilities | https://data.cityofchicago.org/ | City of Chicago Data Portal Terms of Use; hospitals from Overture Maps (CDLA-Permissive-2.0) | City of Chicago, Chicago Public Library, Chicago Public Schools, Chicago Fire Department; Overture Maps Foundation | REVIEW: custom terms; attribution required, confirm redistribution |
| chicago-gps-traces | https://www.openstreetmap.org/copyright | Synthetic; derived from OpenStreetMap, ODbL 1.0 | © OpenStreetMap contributors | MUST RESOLVE BEFORE MERGE: ODbL / OpenStreetMap-derived (attribution and share-alike obligations) |
| chicago-lodes-od | https://lehd.ces.census.gov/data/ | Public domain (US Census Bureau LEHD) | US Census Bureau, LEHD Origin-Destination Employment Statistics (LODES8, 2021) | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| chicago-nature | https://www.inaturalist.org/observations?place_id=49906&year=2023 | CC0, CC BY 4.0 or CC BY-NC 4.0 per observation (non-commercial use) | iNaturalist contributors | MUST RESOLVE BEFORE MERGE: non-commercial terms |
| chicago-parks | https://www.openstreetmap.org/copyright | ODbL 1.0 (Open Database License) | © OpenStreetMap contributors (ODbL) | MUST RESOLVE BEFORE MERGE: ODbL / OpenStreetMap-derived (attribution and share-alike obligations) |
| chicago-places | https://docs.overturemaps.org/attribution/places | CDLA-Permissive-2.0 (Overture Places; some sources Apache-2.0 or CC0) | Overture Maps Foundation (release 2026-09-23.1) | Attribution required; redistribution permitted |
| chicago-population-support | https://esa-worldcover.org/en/data-access | Public domain (US Census Bureau, CDC/ATSDR); ESA WorldCover CC BY 4.0 | US Census Bureau 2022 TIGER/Cartographic Boundary; CDC/ATSDR SVI 2022; © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data 2021 processed by WorldCover consortium | Attribution required; redistribution permitted |
| chicago-roads | https://www.openstreetmap.org/copyright | ODbL 1.0 | © OpenStreetMap contributors | MUST RESOLVE BEFORE MERGE: ODbL / OpenStreetMap-derived (attribution and share-alike obligations) |
| chicago-taxi-od | https://data.cityofchicago.org/resource/wrvz-psew | City of Chicago Data Portal Terms of Use | City of Chicago (taxi trip data reported by licensed taxi companies) | REVIEW: custom terms; attribution required, confirm redistribution |
| chicago-tracts | https://svi.cdc.gov/dataDownloads/data-download.html | Public domain (US Census Bureau, CDC/ATSDR); nature counts from iNaturalist observations (CC0 / CC BY / CC BY-NC) | US Census Bureau (TIGER/Cartographic Boundary, ACS), CDC/ATSDR SVI 2022, CDC PLACES, LEHD LODES, City of Chicago / CPD | MUST RESOLVE BEFORE MERGE: non-commercial terms |
| chicago-walk | https://www.openstreetmap.org/copyright | ODbL 1.0 | © OpenStreetMap contributors | MUST RESOLVE BEFORE MERGE: ODbL / OpenStreetMap-derived (attribution and share-alike obligations) |
| cta-transit | https://www.transitchicago.com/developers/gtfs/ | CTA Developer License Agreement and Terms of Use (derived data for transit-accessibility use; credit optional; no stand-alone resale) | Data provided by Chicago Transit Authority (not affiliated with or endorsed by CTA) | REVIEW: custom terms; attribution required, confirm redistribution |
| dixie-fire | https://registry.opendata.aws/sentinel-2-l2a-cogs/ | Copernicus Sentinel data: free, full and open; ESA WorldCover: CC BY 4.0; terrain: AWS Terrain Tiles | Contains modified Copernicus Sentinel data 2021; (c) ESA WorldCover 2021 (CC BY 4.0); terrain from AWS Terrain Tiles / USGS 3DEP | Attribution required; redistribution permitted |
| dixie-perimeter | https://data-nifc.opendata.arcgis.com/ | Public domain (US Government work, NIFC) | Fire perimeters: National Interagency Fire Center (NIFC) Open Data, public domain; packaged by poopdeck.gl | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| dixie-spectral | https://registry.opendata.aws/sentinel-2-l2a-cogs/ | Copernicus Sentinel data: free, full and open | Contains modified Copernicus Sentinel data 2021 | Attribution required; redistribution permitted |
| gfs-wind | https://registry.opendata.aws/noaa-gfs-bdp-pds/ | US Government work (NOAA), public domain | NOAA / NCEP Global Forecast System | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| ghcn-stations | https://registry.opendata.aws/noaa-ghcn/ | NOAA NCEI GHCN-Daily, public domain | NOAA National Centers for Environmental Information, GHCN-Daily | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| grand-canyon-dem | https://registry.opendata.aws/terrain-tiles/ | AWS Open Data Terrain Tiles (public bucket); underlying USGS 3DEP/NED is public domain, other sources require attribution per Tilezen list | Terrain Tiles on AWS (Mapzen/Tilezen): USGS 3DEP/NED, SRTM and other sources; see https://github.com/tilezen/joerd/blob/master/docs/attribution.md | REVIEW: custom terms; attribution required, confirm redistribution |
| gtfs-nl-rail-graph | https://gtfs.ovapi.nl/nl/gtfs-nl.zip | CC0 1.0 Universal (public domain) | OVapi / NDOV national GTFS feed (derived graph) | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| gull-migration | https://zenodo.org/records/5068540 | CC0 1.0 Universal (public domain) | Stienen E. et al., LifeWatch/INBO LBBG_ZEEBRUGGE GPS tracking (Zenodo, doi:10.5281/zenodo.5068540) | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| ibtracs-north-atlantic | https://www.ncei.noaa.gov/products/international-best-track-archive | Public domain (NOAA NCEI) | Knapp, K. R., Kruk, M. C., Levinson, D. H., Diamond, H. J., Neumann, C. J. (2010). IBTrACS, Bulletin of the American Meteorological Society 91, 363-376. NOAA NCEI, v04r01 | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| lidar-ground | https://data.usgs.gov/datacatalog/data/USGS%3Ab7e353d2-325f-4fc6-8d95-01254705638a | Public domain (U.S. Government work; USGS 3DEP) | U.S. Geological Survey, 3D Elevation Program (3DEP), King County 2021 | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| meuse-validation | https://edzer.r-universe.dev/sp/data/meuse/json | GPL-2.0-or-later (R package sp data) | Meuse topsoil and river data distributed with R package sp 2.2-4; data collectors Ruud van Rijn and Mathieu Rikken; compiled by Edzer Pebesma. | REVIEW: GPL-2.0-or-later data; confirm redistribution inside an MIT repository |
| montreal-boroughs | https://donnees.montreal.ca/dataset/limites-administratives-agglomeration | Creative Commons Attribution 4.0 International | Ville de Montréal, données ouvertes (CC BY 4.0) | Attribution required; redistribution permitted |
| natural-earth | https://www.naturalearthdata.com/ | Public domain (Natural Earth) | Made with Natural Earth | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| naturalearth-atlantic-coast | https://www.naturalearthdata.com/downloads/50m-physical-vectors/50m-coastline/ | Public domain (Natural Earth) | Made with Natural Earth | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| ncovr-new-jersey | https://geodacenter.github.io/data-and-lab/ncovr/ | Source data provided as-is with no warranties; no separate license asserted | Messner, Anselin, Hawkins, Deane, Tolnay and Baller (2000), NCOVR; GeoDa archive prepared by Luc Anselin | REVIEW: no licence asserted by the source |
| ndvi-timeseries | https://registry.opendata.aws/sentinel-2-l2a-cogs/ | Copernicus Sentinel data: free, full and open | Contains modified Copernicus Sentinel data 2017-2024 | Attribution required; redistribution permitted |
| northeast-pacific-sst | https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc%3AC00927 | NOAA: no constraints on data access or use; free redistribution | NOAA National Centers for Environmental Information; NOAA CoastWatch ERDDAP | REVIEW: custom terms; attribution required, confirm redistribution |
| ny-taxi-trips | https://github.com/visgl/deck.gl-data/tree/master/examples/trips | Sample data from the deck.gl-data repository (MIT) | deck.gl-data trips-v7 (NYC Taxi and Limousine Commission trips, sampled) | UNVERIFIED |
| openflights | https://openflights.org/data.php | Open Database License (ODbL) 1.0 | OpenFlights.org (airports.dat, routes.dat), ODbL | MUST RESOLVE BEFORE MERGE: ODbL / OpenStreetMap-derived (attribution and share-alike obligations) |
| plate-boundaries | https://github.com/fraxen/tectonicplates | ODC-BY 1.0 | Bird (2003), PB2002; GeoJSON conversion by Hugo Ahlenius, Nordpil (github.com/fraxen/tectonicplates) | Attribution required; redistribution permitted |
| poopdeck-adsb-paths | https://opensky-network.org/data/impala | OpenSky Network data terms: research and non-commercial use with attribution (the licence text could not be fetched when this entry was written, so treat the data as NON-COMMERCIAL until checked). Not CC0 or CC BY: do not redistribute commercially. | Data from the OpenSky Network, https://opensky-network.org. Matthias Schafer, Martin Strohmeier, Vincent Lenders, Ivan Martinovic and Matthias Wilhelm, "Bringing Up OpenSky: A Large-scale ADS-B Sensor Network for Research", IPSN 2014. Prepared via the poopdeck.gl archives. | MUST RESOLVE BEFORE MERGE: non-commercial terms |
| poopdeck-ais-us | https://hub.marinecadastre.gov/pages/vesseltraffic | Public domain (US Government work) | NOAA Office for Coastal Management / BOEM / U.S. Coast Guard, Marine Cadastre AIS data (via the poopdeck.gl ais-all-us archive) | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| poopdeck-animals | https://www.gbif.org/dataset/66e0553e-75f6-49de-b614-22efd9fbf6e9 | CC0 1.0 Universal (all five source datasets, checked in the GBIF dataset API) | INBO LifeWatch Belgium via GBIF: MH_WATERLAND (doi:10.5281/zenodo.10053583), H_GRONINGEN (doi:10.5281/zenodo.10053658), MH_ANTWERPEN (doi:10.5281/zenodo.10054153), BOP_RODENT (doi:10.5281/zenodo.17310324), SPOONBILL_VLAANDEREN (doi:10.5281/zenodo.23103735). Folded tracks from the poopdeck.gl animals archive. | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| poopdeck-bixi-rides | https://tiles.poopdeck.gl/data/bixi-points/manifest.json | ODbL 1.0 (OSRM routes on OpenStreetMap) and Creative Commons Attribution (BIXI Montreal trip history) | BIXI Montréal open data (CC BY); routes via OSRM on © OpenStreetMap contributors (ODbL); poopdeck.gl | MUST RESOLVE BEFORE MERGE: ODbL / OpenStreetMap-derived (attribution and share-alike obligations) |
| poopdeck-drifters | https://www.aoml.noaa.gov/phod/gdp/ | Public domain (NOAA waives copyright on GDP data) | NOAA Global Drifter Program (AOML / PMEL), 6-hourly interpolated drifter data, via the poopdeck.gl archive | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| poopdeck-earthquakes | https://earthquake.usgs.gov/fdsnws/event/1/ | Public domain (US Government work) | USGS Earthquake Catalog (ComCat), via poopdeck.gl | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| poopdeck-ecco-currents | https://ecco-group.org/ | NASA Earth science data, open and free to use | ECCO Consortium, NASA/JPL ECCO V4r4 ocean state estimate (PO.DAAC), particle archive from poopdeck.gl; mean field derived by this showcase | REVIEW: custom terms; attribution required, confirm redistribution |
| poopdeck-goes-glm-lightning | https://registry.opendata.aws/noaa-goes/ | Public domain (US Government work) | NOAA GOES-R GLM Level 2 (LCFA), via poopdeck.gl | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| poopdeck-gtfs-nl | https://gtfs.ovapi.nl/nl/gtfs-nl.zip | CC0 1.0 Universal (public domain) | OVapi / NDOV national GTFS feed, via poopdeck.gl (gtfs-nl) | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| poopdeck-mrms-precip-tracks | https://tiles.poopdeck.gl/data/mrms-precip-tracks/manifest.json | Public domain (US Government work) | NOAA MRMS (Multi-Radar Multi-Sensor), via poopdeck.gl | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| poopdeck-mrms-storm3d-outages | https://figshare.com/articles/dataset/24237376 | CC BY 4.0 | DOE / Oak Ridge National Laboratory EAGLE-I power outage data (CC BY 4.0), via poopdeck.gl | Attribution required; redistribution permitted |
| poopdeck-mrms-storm3d-reports | https://www.spc.noaa.gov/climo/reports/ | Public domain (US Government work) | NOAA Storm Prediction Center preliminary local storm reports, via poopdeck.gl | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| poopdeck-mrms-storm3d-warnings | https://mesonet.agron.iastate.edu/ | Public domain (US Government work) | NOAA National Weather Service warnings via the Iowa Environmental Mesonet, via poopdeck.gl | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| poopdeck-nyc-taxi | https://tiles.poopdeck.gl/data/nyc-taxi-paths/manifest.json | NYC Open Data terms of use (NYC TLC trip records, no restrictions); route endpoints snapped by OSRM on OpenStreetMap data (ODbL) | NYC Taxi & Limousine Commission trip records via NYC Open Data; poopdeck.gl nyc-taxi-paths archive; © OpenStreetMap contributors | MUST RESOLVE BEFORE MERGE: ODbL / OpenStreetMap-derived (attribution and share-alike obligations) |
| poopdeck-nyc-taxi-paths | https://tiles.poopdeck.gl/data/nyc-taxi-paths/manifest.json | NYC Open Data terms of use (NYC TLC trip records, no restrictions); OSRM routes on OpenStreetMap data (ODbL) | NYC Taxi & Limousine Commission trip records via NYC Open Data; poopdeck.gl nyc-taxi-paths archive; routes by OSRM on © OpenStreetMap contributors (ODbL) | MUST RESOLVE BEFORE MERGE: ODbL / OpenStreetMap-derived (attribution and share-alike obligations) |
| poopdeck-osm-nyc | https://tiles.poopdeck.gl/data/osm-nyc-nodes/manifest.json | ODbL 1.0 (Open Database License) | © OpenStreetMap contributors | MUST RESOLVE BEFORE MERGE: ODbL / OpenStreetMap-derived (attribution and share-alike obligations) |
| poopdeck-wildfires | https://data-nifc.opendata.arcgis.com/ | Public domain (US Government work, NIFC) | Fire perimeters: National Interagency Fire Center (NIFC) Open Data, public domain; packaged by poopdeck.gl | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| sentinel-cloud-repair | https://element84.com/earth-search/ | Copernicus Sentinel data legal notice; free, full and open access with attribution | Contains modified Copernicus Sentinel data (2026) | REVIEW: custom terms; attribution required, confirm redistribution |
| sf-bike-parking | https://data.sfgov.org/Transportation/Bicycle-Parking-Public-/hn4x-zwk7 | DataSF Open Data (PDDL), via the deck.gl-data repository | DataSF, bicycle parking (via deck.gl-data) | UNVERIFIED |
| swissimage-zermatt | https://www.swisstopo.admin.ch/en/orthoimage-swissimage-10 | swisstopo Open Government Data (OGD), free use with attribution | © swisstopo | REVIEW: custom terms; attribution required, confirm redistribution |
| us-airline-flows | https://www.transtats.bts.gov/ | US government public domain (BTS); airport coordinates ODbL 1.0 (OpenFlights) | U.S. Bureau of Transportation Statistics On-Time Reporting Carrier data; airport positions OpenFlights.org (ODbL) | MUST RESOLVE BEFORE MERGE: ODbL / OpenStreetMap-derived (attribution and share-alike obligations) |
| us-counties | https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html | US Government works (Census, CDC/ATSDR, CDC PLACES, USDA ERS): public domain | US Census Bureau cartographic boundaries 2022; CDC/ATSDR Social Vulnerability Index 2022; CDC PLACES 2024 release; USDA Economic Research Service | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| us-county-births | https://www.census.gov/programs-surveys/popest.html | US Government work (US Census Bureau): public domain | US Census Bureau, Population Estimates Program, Vintage 2023 | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| us-county-mortality | https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars | US Government work (NHTSA FARS): public domain | NHTSA Fatality Analysis Reporting System 2017-2023; US Census Bureau 2020 population | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| us-elections | https://doi.org/10.7910/DVN/VOQCHQ | CC0 1.0 (MIT Election Data and Science Lab) | MIT Election Data and Science Lab, County Presidential Election Returns 2000-2020 and 2024 precinct returns, Harvard Dataverse (CC0) | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| us-states | https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html | US Government work (US Census Bureau): public domain | US Census Bureau cartographic boundaries 2022 | Redistribution permitted (public domain / CC0); keep attribution as courtesy |
| wildfire-california-dem | https://registry.opendata.aws/terrain-tiles/ | Terrain Tiles on AWS: public domain and open source data, attribution required | Terrain Tiles on AWS (Mapzen, Tilezen): USGS NED, SRTM, GMTED2010, ETOPO1 and others; see https://github.com/tilezen/joerd/blob/master/docs/attribution.md | REVIEW: custom terms; attribution required, confirm redistribution |

`ny-taxi-trips` and `sf-bike-parking` have a dataset entry but no committed data directory (they load
remotely from deck.gl-data), so nothing is redistributed for them.

## Lens data

Location: `modules/lenses/src/computation-lenses/data/` (formerly
`modules/graph/src/computation-lenses/data/`). Small extracts of the showcase datasets above. The
README in that directory lists the source, licence and extraction script for each file; the licence of
an extract is that of its showcase dataset. Redistribution status follows the dataset: files
extracted from `chicago-nature`, `chicago-tracts` and OpenStreetMap-derived datasets are in the
"must resolve before merge" list above.

## Test oracle fixtures

- `modules/graph/test/oracles/upstream-golden/`: unmodified fixtures from Circuitscape.jl (MIT, licence
  retained in `Circuitscape.jl/LICENSE.md`), PDAL (BSD, per its LICENSE.txt; `PDAL/LICENSE.txt` retained) and OTB
  (Apache-2.0; `OTB/LICENSE` and `OTB/NOTICE` retained). Commit hashes and SHA-256 of every file are
  in `manifest.json` there.
- `modules/graph/test/oracles/fixtures/*.json` (746 files): generated by the neighbouring oracle
  scripts from synthetic inputs or from the reference libraries named in each script (for example
  shapely, PostGIS, PySAL, mgwr, GDAL, GRASS). Per-fixture licence provenance has not been audited
  here: UNVERIFIED for any fixture whose script reads external data; see the producing script.
