// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {DemoDataSource} from '../src/gpu/data/provenance';

/** Source credits used by the website. The preparation and sampling notes live beside the data. */
function source(
  id: string,
  label: string,
  url: string,
  credit: string,
  terms = 'Public domain',
  termsUrl?: string
): DemoDataSource {
  return {id, label, url, credit, terms, termsUrl};
}

const CC_BY = 'https://creativecommons.org/licenses/by/4.0/';
const SENTINEL_NOTICE =
  'https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice';

const sources = [
  source(
    'nadcon',
    'Midwest datum shifts',
    'https://www.ngs.noaa.gov/PC_PROD/NADCON/Readme.htm',
    'NOAA National Geodetic Survey, NADCON CONUS grid'
  ),
  source(
    'state-boundaries',
    'U.S. state boundaries',
    'https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html',
    'U.S. Census Bureau, cartographic boundaries'
  ),
  source(
    'amsterdam-buildings',
    'Amsterdam building footprints',
    'https://docs.3dbag.nl/en/copyright/',
    '© 3DBAG by tudelft3d and 3DGI. Modified footprint and height extract.',
    'CC BY 4.0',
    CC_BY
  ),
  source(
    'ghcn',
    'GHCN-Daily precipitation',
    'https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily',
    'NOAA National Centers for Environmental Information'
  ),
  source(
    'comcat-japan',
    'Japanese earthquake epicenters',
    'https://earthquake.usgs.gov/data/comcat/',
    'USGS ANSS Comprehensive Earthquake Catalog'
  ),
  source(
    'comcat-turkiye',
    'Türkiye earthquake sequence',
    'https://earthquake.usgs.gov/data/comcat/',
    'USGS ANSS Comprehensive Earthquake Catalog'
  ),
  source(
    'comcat-kilauea',
    'Kīlauea earthquake sequence',
    'https://earthquake.usgs.gov/data/comcat/',
    'USGS Hawaiian Volcano Observatory'
  ),
  source(
    'chicago-tracts',
    'Chicago census tract boundaries',
    'https://www.census.gov/geographies/mapping-files/2022/geo/carto-boundary-file.html',
    'U.S. Census Bureau, 2022 cartographic boundaries'
  ),
  source(
    'chicago-acs',
    'Chicago tract demographics',
    'https://atsdr.cdc.gov/place-health/php/svi/svi-data-documentation-download.html',
    'CDC/ATSDR Social Vulnerability Index 2022 and the U.S. Census Bureau ACS'
  ),
  source(
    'lodes',
    'Chicago employment counts',
    'https://lehd.ces.census.gov/data/',
    'U.S. Census Bureau, LEHD Origin-Destination Employment Statistics'
  ),
  source(
    'births',
    'Illinois county births and population',
    'https://www.census.gov/programs-surveys/popest.html',
    'U.S. Census Bureau, Population Estimates Program, Vintage 2023'
  ),
  source(
    'canyon',
    'Grand Canyon elevation',
    'https://data.usgs.gov/datacatalog/data/USGS%3A35f9c4d4-b113-4c8d-8691-47c428c29a5b',
    'U.S. Geological Survey, 3D Elevation Program'
  ),
  source(
    'reflectance',
    'Dixie Fire spectral bands',
    'https://registry.opendata.aws/sentinel-2-l2a-cogs/',
    'Contains modified Copernicus Sentinel data 2021',
    'Copernicus Sentinel Data Legal Notice',
    SENTINEL_NOTICE
  ),
  source(
    'ndvi',
    'Dixie Fire vegetation index',
    'https://registry.opendata.aws/sentinel-2-l2a-cogs/',
    'Contains modified Copernicus Sentinel data 2017–2024',
    'Copernicus Sentinel Data Legal Notice',
    SENTINEL_NOTICE
  ),
  source(
    'wind',
    'Hurricane Helene wind forecast',
    'https://registry.opendata.aws/noaa-gfs-bdp-pds/',
    'NOAA National Centers for Environmental Prediction, Global Forecast System'
  ),
  source(
    'streets',
    'San Francisco street network',
    'https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html',
    'U.S. Census Bureau, 2024 TIGER/Line'
  ),
  source(
    'vessels',
    'San Francisco Bay vessel tracks',
    'https://coast.noaa.gov/digitalcoast/data/vesseltraffic.html',
    'USCG, NOAA and BOEM, Marine Cadastre AIS'
  ),
  source(
    'natural-earth',
    'Natural Earth cartography',
    'https://www.naturalearthdata.com/about/terms-of-use/',
    'Made with Natural Earth'
  ),
  source(
    'worldcover',
    'Hyde Park land cover',
    'https://esa-worldcover.org/en/data-access',
    '© ESA WorldCover project 2021. Contains modified Copernicus Sentinel data 2021 processed by the WorldCover consortium. Land-cover classes are cropped and resampled.',
    'CC BY 4.0',
    CC_BY
  ),
  source(
    'lightning',
    'Midwest lightning flashes',
    'https://www.nesdis.noaa.gov/our-satellites/currently-flying/goes-east-west/geostationary-lightning-mapper-glm',
    'NOAA GOES-R Geostationary Lightning Mapper'
  ),
  source(
    'lidar',
    'King County lidar returns',
    'https://data.usgs.gov/datacatalog/data/USGS%3Ab7e353d2-325f-4fc6-8d95-01254705638a',
    'U.S. Geological Survey, 3D Elevation Program, King County 2021'
  ),
  source(
    'flights',
    'Atlanta departures',
    'https://www.transtats.bts.gov/ONTIME/',
    'U.S. Bureau of Transportation Statistics, Reporting Carrier On-Time Performance'
  )
];

export const DATA_SOURCES = Object.fromEntries(sources.map(record => [record.id, record]));

/** Paths relative to src/gpu/data. Register leaf assets, or single-source producer modules. */
export const DATA_FILES: Record<string, readonly string[]> = {
  'helene-rain.json': ['ghcn'],
  'midwest-nadcon.json': ['nadcon', 'state-boundaries'],
  'amsterdam-buildings.json': ['amsterdam-buildings'],
  'world-coastline.json': ['natural-earth'],
  'japan-coast.json': ['natural-earth'],
  'chicago-workplace-blocks.json': ['lodes'],
  'chicago-commute-lines.json': ['lodes'],
  'japan-earthquakes.json': ['comcat-japan'],
  'chicago-tracts.json': ['chicago-tracts', 'chicago-acs'],
  'kilauea-events.json': ['comcat-kilauea'],
  'chicago-workplace-jobs.json': ['lodes'],
  'illinois-births.json': ['births'],
  'rasters/grand-canyon.bin': ['canyon'],
  'rasters/dixie-spectral.bin': ['reflectance'],
  'rasters/dixie-ndvi.bin': ['ndvi'],
  'rasters/helene-wind.bin': ['wind'],
  'sf-market-streets.json': ['streets'],
  'sf-bay-ais.json': ['vessels'],
  'chicago-commutes.json': ['lodes'],
  'natural-earth-geometry.json': ['natural-earth'],
  'midwest-flashes.b64': ['lightning'],
  'king-county.b64': ['lidar'],
  'atlantadepartures.json': ['flights'],
  'recipes/hyde-park-cover.json': ['worldcover'],
  'recipes/hyde-park-tracts.json': ['chicago-tracts', 'chicago-acs'],
  'recipes/dixie-ndvi.json': ['ndvi'],
  'recipes/turkiye-quakes.json': ['comcat-turkiye']
};

/** A source can supply the geometry while the variable being analyzed is a scenario. */
export const SCENARIOS: Record<string, string> = {
  'spherical-arc-overlay':
    'The survey swath is modeled on the generalized Lake Superior shoreline.',
  'spherical-polygon-overlay': 'The survey corridor and its exclusion zone are constructed.',
  'spherical-convex-intersection':
    'The survey footprint is modeled. The lake operand is a convex envelope of the shoreline.',
  'spherical-convex-contains':
    'The search radius defines a hypothetical area around the observed epicenters.',
  'spherical-polygon-contains':
    'The lake shoreline and lightning locations are observed. The exclusion zone is modeled.',
  'local-variability-recipe':
    'The noise control imposes a spatial gradient in scatter on tract poverty estimates.',
  'spatial-regression-recipe':
    'Observed poverty and age 65+ shares are predictors. The response and spatial error dependence are modeled.',
  'grid-generator': 'The sampling grid is constructed over the rain-gauge study area.',
  'shape-generator': 'Search radii and sectors are modeled around observed vessel tracks.',
  'offset-expansion': 'The reader caps the number of observed fixes taken from each vessel.',
  'datum-grid-transform':
    'Test locations are constructed. The displayed correction vectors are magnified.',
  'local-morans-i':
    'The noise control adds simulated measurement error to observed station totals.',
  'seasonal-break-detection':
    'A monthly seasonal cycle is modeled from observed pre- and post-fire vegetation levels.',
  'raster-registration': 'The comparison image is shifted by a known offset.',
  'pointcloud-change': 'The displacement in the comparison cloud is simulated.',
  'pointcloud-local-positions':
    'Cloud offsets and scales are adjusted to demonstrate coordinate precision.',
  'cell-grid-path': 'Paths connect pairs of observed epicenters on the H3 grid.',
  'cell-compaction': 'The missing-cell mask is modeled around an observed epicenter.',
  's2-cap-covering': 'The search caps are hypothetical neighborhoods around measured epicenters.',
  'cell-topology':
    'The chosen grid distance defines a neighborhood around each observed epicenter.',
  'cell-table-compare': 'The comparison shifts epicenters eastward in a hypothetical scenario.',
  'cell-set-outline':
    'The spread control scales epicenter positions about the study-window center.',
  'convex-straight-skeleton':
    'Simplified convex lake hulls are stretched to exercise skeleton extraction.',
  'segregation-counterfactual-recipe':
    'The ACS baseline is observed. Growth and changes in group composition are simulated.',
  'segregation-decomposition-recipe':
    'The ACS baseline is observed. Changes in population and group composition are simulated.',
  'exposure-by-zone-recipe':
    'The hazard and population weights are modeled using observed built-up coverage.',
  'rate-cluster-map-recipe':
    'Case counts and population weights are modeled from built-up coverage.',
  'change-of-support-recipe':
    'Constructed reporting zones transfer weights derived from observed built-up coverage.',
  'dasymetric-redistribution-recipe':
    'Population weights are modeled from built-up coverage and redistributed across constructed zones.',
  'cell-flow-topology-recipe':
    'Tract pairs are assigned to display cells. The eastward weight is a scenario parameter.',
  'period-comparison-recipe': 'Event counts use a local display lattice for comparing periods.',
  'points-in-polygons-choropleth-recipe': 'Epicenters are counted in constructed reporting zones.',
  'zonal-distribution-recipe': 'Constructed reporting zones group the sampled events.',
  'pointcloud-icp': 'An imposed rotation and translation displace the comparison cloud.',
  'pointcloud-statistical-filter': 'Outliers are added to the observed returns.',
  'track-quality': 'The quality scenario introduces errors into observed AIS tracks.',
  'trajectory-kalman': 'Simulated position noise is added to the observed AIS fixes.',
  'regional-raster-change-recipe':
    'Vegetation change is summarized over constructed reporting zones.',
  'shortest-path': 'Travel costs and congestion are modeled on the observed street network.',
  'network-isochrones': 'Travel costs are modeled on the observed street network.',
  'delaunay-voronoi': 'The pointer moves a sample site for the tessellation experiment.'
};

/** Notes follow the input helper into every run that actually consumes it. */
export const MODEL_INPUTS: Record<string, string> = {
  'data/hydrography.ts#createShoreRing':
    'Lake shorelines are sampled radially and placed in constructed geometry.',
  'data/hydrography.ts#createRiverLine':
    'River reaches are resampled and rotated into local comparison panels.',
  'runs/entries/regression-weights/shared/regression.ts#createResponse':
    'Responses are simulated from tract covariates and specified coefficients.',
  'runs/entries/regression-weights/shared/panel.ts#definePanelRun':
    'Panel periods and responses are simulated from tract covariates.',
  'runs/entries/recipes-trajectories/lib/tracts.ts#getRateCounts':
    'Populations and event counts are modeled using tract poverty estimates.',
  'runs/entries/recipes-trajectories/lib/zones.ts#createZoneRectangles':
    'The vessel tracks are observed. Zone boundaries are constructed.',
  'runs/entries/spatial-statistics/shared/point-field.ts#mixNoise':
    'The noise control perturbs station observations.',
  'runs/entries/recipes-trajectories/lib/quads.ts#createQuadZones':
    'Reporting zones are constructed for the overlay.',
  'demos/street-network.ts#createStreetNetwork':
    'Intersections follow TIGER/Line geometry. Bidirectional links and travel costs are modeled.',
  'runs/inputs.ts#createSmoothField': 'The scenario includes a generated spatial field.'
};
