// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createDriveTimeBandsSample,
  createDriveTimeBandsSecondDataset,
  DRIVE_TIME_BANDS_SAMPLE
} from './samples/drive-time-bands-data';
import {
  createCellFlowTopologySample,
  createSecondCellFlowTopologySample,
  CELL_FLOW_TOPOLOGY_SAMPLE_LICENSE
} from './samples/cell-flow-topology';
import {
  createNetworkDisruptionSample,
  createNetworkDisruptionSecondDataset,
  NETWORK_DISRUPTION_SAMPLE
} from './samples/network-disruption';
import {
  createRegionalRasterChangeSample,
  REGIONAL_RASTER_CHANGE_SAMPLE_LICENSE
} from './samples/regional-raster-change';
import {createZonalExposureSample, createMappedZonalExposureSample} from './samples/zonal-exposure';
import {
  makeSurfacePreparationSample,
  makeSurfacePreparationSecondSample,
  SURFACE_PREPARATION_SAMPLE_LICENSE
} from './samples/surface-preparation';
import {
  createDrainageNetworkSample,
  createDrainageNetworkSecondDataset,
  DRAINAGE_NETWORK_SAMPLE
} from './samples/drainage-network';
import {
  createEmergingHotSpotsSample,
  EMERGING_HOT_SPOTS_SAMPLE_LICENSE,
  createTrackQualitySample,
  TRACK_QUALITY_SAMPLE_LICENSE,
  createLandscapeMetricsSample,
  LANDSCAPE_METRICS_SAMPLE_LICENSE,
  createFleetDwellSample,
  FLEET_DWELL_SAMPLE_LICENSE
} from './samples/index';

import type {CatalogFeature, CatalogFeatureCollection} from '../analysis/catalog-geojson';
import type {CatalogParameterValue} from '../analysis/catalog-binding-types';
import type {CatalogDataInput} from '../analysis/catalog-data-input';
import {
  createDasymetricRedistributionSample,
  DASYMETRIC_SAMPLE_LICENSE
} from './samples/dasymetric-redistribution';

// Repository-authored synthetic samples, with no third-party data. Seeded feature generators
// use geographic city coordinates or a declared planar lattice. Typed raster fixtures carry
// explicit grid units and fixed numerical expectations. Every factory is deterministic.

/** Longitude and latitude of the sample city centre. */
const CITY_CENTER_LONGITUDE = -87.65;
const CITY_CENTER_LATITUDE = 41.85;

/** Metres in one degree of latitude on the sphere used by the samples (rounded; the samples need no more). */
const METERS_PER_DEGREE_LATITUDE = 111_320;

/** The district lattice: 12 by 12 districts, each 0.008 degrees of longitude by 0.006 of latitude (about 660 m). */
const DISTRICT_COLUMNS = 12;
const DISTRICT_ROWS = 12;
const DISTRICT_LONGITUDE_STEP = 0.008;
const DISTRICT_LATITUDE_STEP = 0.006;
const DISTRICT_ORIGIN_LONGITUDE =
  CITY_CENTER_LONGITUDE - (DISTRICT_COLUMNS / 2) * DISTRICT_LONGITUDE_STEP;
const DISTRICT_ORIGIN_LATITUDE =
  CITY_CENTER_LATITUDE - (DISTRICT_ROWS / 2) * DISTRICT_LATITUDE_STEP;

/** Seeded feature samples, retaining each stream's existing sample entry. */
export const KEPLER_CURATED_FEATURE_ANALYSIS_SAMPLE_IDS = [
  'fleet-dwell',
  'emerging-hot-spots',
  'track-quality',
  'city-incidents',
  'city-district-rates',
  'city-district-incidents',
  'rain-gauges',
  'street-grid-facilities',
  'before-after-incidents',
  'facility-observations',
  'boundary-pair',
  'district-regression',
  'district-regression-omitted-variable',
  'district-smooth-surface'
] as const;

export type KeplerCuratedFeatureAnalysisSampleId =
  (typeof KEPLER_CURATED_FEATURE_ANALYSIS_SAMPLE_IDS)[number];

/** All discoverable synthetic samples, including typed non-feature inputs. */
export const KEPLER_CURATED_ANALYSIS_SAMPLE_IDS = [
  ...KEPLER_CURATED_FEATURE_ANALYSIS_SAMPLE_IDS,
  'dasymetric-fractional-support',
  'landscape-metrics',
  'drainage-network-slope',
  'surface-preparation-gaps',
  'zonal-exposure-counts',
  'regional-raster-change',
  'network-disruption-bridge',
  'cell-flow-topology',
  'drive-time-bands-streets'
] as const;

export type KeplerCuratedAnalysisSampleId = (typeof KEPLER_CURATED_ANALYSIS_SAMPLE_IDS)[number];

/** Shared sample provenance used by template discovery. */
export function getKeplerCuratedAnalysisSampleMetadata(sampleId: string) {
  if (sampleId === 'drainage-network-slope') {
    const {license, attribution, reference} = DRAINAGE_NETWORK_SAMPLE;
    return {id: sampleId, license, attribution, reference};
  }
  if (sampleId === 'surface-preparation-gaps') {
    const {license, attribution, reference} = SURFACE_PREPARATION_SAMPLE_LICENSE;
    return {id: sampleId, license, attribution, reference};
  }
  if (sampleId === 'zonal-exposure-counts') {
    const {license, attribution, reference} = createZonalExposureSample();
    return {id: sampleId, license, attribution, reference};
  }
  if (sampleId === 'regional-raster-change') {
    const {license, attribution, reference} = REGIONAL_RASTER_CHANGE_SAMPLE_LICENSE;
    return {id: sampleId, license, attribution, reference};
  }
  if (sampleId === 'network-disruption-bridge') {
    const {license, attribution, reference} = NETWORK_DISRUPTION_SAMPLE;
    return {id: sampleId, license, attribution, reference};
  }
  if (sampleId === 'cell-flow-topology') {
    const {license, attribution, reference} = CELL_FLOW_TOPOLOGY_SAMPLE_LICENSE;
    return {id: sampleId, license, attribution, reference};
  }
  if (sampleId === 'drive-time-bands-streets') {
    const {license, attribution, reference} = DRIVE_TIME_BANDS_SAMPLE;
    return {id: sampleId, license, attribution, reference};
  }
  if (sampleId === 'emerging-hot-spots')
    return {id: sampleId, ...EMERGING_HOT_SPOTS_SAMPLE_LICENSE};
  if (sampleId === 'fleet-dwell') return FLEET_DWELL_SAMPLE_LICENSE;
  if (sampleId === 'track-quality') return TRACK_QUALITY_SAMPLE_LICENSE;
  if (sampleId === 'landscape-metrics') return LANDSCAPE_METRICS_SAMPLE_LICENSE;
  if (sampleId === 'dasymetric-fractional-support') return DASYMETRIC_SAMPLE_LICENSE;
  if (
    KEPLER_CURATED_FEATURE_ANALYSIS_SAMPLE_IDS.includes(
      sampleId as KeplerCuratedFeatureAnalysisSampleId
    )
  ) {
    return {
      id: sampleId,
      license: 'MIT',
      attribution: 'vis.gl contributors',
      reference: 'modules/kepler/src/curated/curated-analysis-samples.ts'
    };
  }
  throw new Error(`getKeplerCuratedAnalysisSampleMetadata: unknown sampleId "${sampleId}"`);
}

/**
 * Column of each sample that fills a field role, keyed by `<slot>.<role>`. A curated analysis maps
 * its field roles to binding parameters; this table says which sample column to hand each role.
 */
export const KEPLER_CURATED_ANALYSIS_SAMPLE_COLUMNS: Readonly<
  Record<KeplerCuratedAnalysisSampleId, Readonly<Record<string, string>>>
> = {
  'fleet-dwell': {'tracks.track': 'track_id', 'tracks.time': 'timestamp'},
  'emerging-hot-spots': {'points.time': 'observed'},
  'track-quality': {'tracks.track': 'track_id', 'tracks.time': 'timestamp'},
  'city-incidents': {'points.value': 'severity', 'points.time': 'observed'},
  'city-district-rates': {
    'areas.value': 'rate',
    'areas.cases': 'cases',
    'areas.population': 'population'
  },
  'city-district-incidents': {'points.value': 'severity'},
  'rain-gauges': {'points.value': 'rainfall'},
  'street-grid-facilities': {'demand.weight': 'residents'},
  'before-after-incidents': {},
  'facility-observations': {'observations.value': 'waitMinutes'},
  'boundary-pair': {'sources.value': 'residents'},
  'district-regression': {'areas.response': 'outcome', 'areas.predictors': 'income,density'},
  'district-regression-omitted-variable': {
    'areas.response': 'outcome',
    'areas.predictors': 'income,density'
  },
  'district-smooth-surface': {'areas.value': 'value'},
  'dasymetric-fractional-support': {'sources.total': 'residents'},
  'landscape-metrics': {},
  'drainage-network-slope': {},
  'surface-preparation-gaps': {},
  'regional-raster-change': {},
  'zonal-exposure-counts': {
    'cells.cell-index': 'cell',
    'cells.zone': 'zone',
    'cells.population': 'people'
  },
  'network-disruption-bridge': NETWORK_DISRUPTION_SAMPLE.columns,
  'cell-flow-topology': {
    'flows.origin': 'origin',
    'flows.destination': 'destination',
    'flows.flow': 'flow'
  },
  'drive-time-bands-streets': DRIVE_TIME_BANDS_SAMPLE.columns
};

/** The planted structure of a sample, for the specification and for anyone reading the sample. */
export const CITY_INCIDENT_CLUSTERS = [
  {name: 'north-west', column: 2.5, row: 9.5, count: 300},
  {name: 'south-east', column: 9.5, row: 3.5, count: 220},
  {name: 'south-west', column: 3.5, row: 2.5, count: 180}
] as const;

/** Number of uniformly scattered background incidents, so the clusters stand out from noise. */
const CITY_BACKGROUND_COUNT = 160;

/** District aggregation examples keep their concentrations inside one roughly 660 m district. */
const DISTRICT_INCIDENT_CLUSTERS = CITY_INCIDENT_CLUSTERS.map((cluster, index) => ({
  ...cluster,
  count: [220, 130, 90][index]
}));
const DISTRICT_CLUSTER_SPREAD = 0.2;

/**
 * City concentrations span roughly 1.2 km on each side, with independently jittered locations
 * on a dense lattice. Centers are at least 4 km apart. At engine defaults the hot-spot
 * cells are several hundred metres wide, so each concentration covers several neighboring cells,
 * rather than splitting a 130 m concentration at a coarse cell corner. Lattice jitter is a quarter
 * of the point interval, so each planted group stays connected at the data-derived clustering
 * radius. Background points are stratified over a 10 by 16 lattice to avoid accidental tiny groups.
 */
const CITY_CLUSTER_WIDTH = 1.8;
const CITY_CLUSTER_JITTER = 0.25;
const CITY_BACKGROUND_COLUMNS = 10;
const CITY_BACKGROUND_ROWS = 16;
const CITY_BACKGROUND_JITTER = 0.15;

/**
 * Districts whose rate is planted high: a block of sixteen adjacent districts (4 by 4), and one lone district
 * whose eight neighbours are planted low, so that it is a high value among low ones (an outlier).
 */
export const CITY_ELEVATED_BLOCK = {
  firstColumn: 4,
  lastColumn: 7,
  firstRow: 4,
  lastRow: 7
} as const;
export const CITY_LONE_HIGH_DISTRICT = {column: 10, row: 2} as const;

/** Baseline cases per resident, the factor the planted districts are raised by and the factor of the low ring. */
const BASELINE_RATE = 0.01;
const ELEVATED_RATE_FACTOR = 3;
const LOW_RING_RATE_FACTOR = 0.4;

/**
 * The coefficients the regression samples are generated with: outcome = intercept + income *
 * income coefficient + density * density coefficient + noise (standard deviation 4). Income and
 * density are independent normal deviates in every district, so the residuals of the first sample
 * are independent; the second sample adds a smooth surface the model does not include.
 */
export const DISTRICT_REGRESSION_PLANTED = {
  intercept: 5,
  income: 0.8,
  density: -1.5,
  noiseDeviation: 4
} as const;

/** Amplitude of the omitted smooth surface added in the second regression sample. */
export const DISTRICT_OMITTED_SURFACE_AMPLITUDE = 8;

/** Seeded generator (mulberry32) with a normal deviate by the Box-Muller transform. */
type SeededGenerator = {next: () => number; nextGaussian: () => number};

function createSeededGenerator(seed: number): SeededGenerator {
  let state = seed >>> 0;
  const next = (): number => {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = state;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
  const nextGaussian = (): number => {
    const first = Math.max(next(), 1e-12);
    const second = next();
    return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * second);
  };
  return {next, nextGaussian};
}

function makePointFeature(
  longitude: number,
  latitude: number,
  properties: Record<string, unknown>
): CatalogFeature {
  return {
    type: 'Feature',
    geometry: {type: 'Point', coordinates: [longitude, latitude]},
    properties
  } as CatalogFeature;
}

function makeCollection(features: CatalogFeature[]): CatalogFeatureCollection {
  return {type: 'FeatureCollection', features};
}

/** Longitude and latitude of a position given in district units (column, row) from the south-west corner. */
function getDistrictPosition(column: number, row: number): [number, number] {
  return [
    DISTRICT_ORIGIN_LONGITUDE + column * DISTRICT_LONGITUDE_STEP,
    DISTRICT_ORIGIN_LATITUDE + row * DISTRICT_LATITUDE_STEP
  ];
}

/** An ISO timestamp from a fixed epoch plus a number of minutes; no clock is read. */
function formatObservedTime(minutesAfterStart: number): string {
  const startMilliseconds = Date.UTC(2025, 0, 1, 0, 0, 0);
  return new Date(startMilliseconds + Math.round(minutesAfterStart) * 60_000)
    .toISOString()
    .replace('.000Z', 'Z');
}

/** Incidents: three Gaussian clusters of different sizes over a uniform background. */
function createClusteredIncidents(
  generator: SeededGenerator,
  clusters: readonly {column: number; row: number; count: number}[],
  backgroundCount: number,
  timeOffsetMinutes: number
): CatalogFeature[] {
  const features: CatalogFeature[] = [];
  for (const cluster of clusters) {
    for (let index = 0; index < cluster.count; index++) {
      const column = cluster.column + generator.nextGaussian() * DISTRICT_CLUSTER_SPREAD;
      const row = cluster.row + generator.nextGaussian() * DISTRICT_CLUSTER_SPREAD;
      const [longitude, latitude] = getDistrictPosition(column, row);
      features.push(
        makePointFeature(longitude, latitude, {
          severity: Math.max(1, Math.min(10, Math.round(6 + generator.nextGaussian() * 1.5))),
          observed: formatObservedTime(timeOffsetMinutes + generator.next() * 60 * 24 * 30)
        })
      );
    }
  }
  for (let index = 0; index < backgroundCount; index++) {
    const [longitude, latitude] = getDistrictPosition(
      generator.next() * DISTRICT_COLUMNS,
      generator.next() * DISTRICT_ROWS
    );
    features.push(
      makePointFeature(longitude, latitude, {
        severity: Math.max(1, Math.min(10, Math.round(3 + generator.nextGaussian() * 1.5))),
        observed: formatObservedTime(timeOffsetMinutes + generator.next() * 60 * 24 * 30)
      })
    );
  }
  return features;
}

function createCityIncidents(seedOffset = 0, districtScale = false): CatalogFeatureCollection {
  const generator = createSeededGenerator(20250101 + seedOffset);
  if (districtScale) {
    return makeCollection(
      createClusteredIncidents(generator, DISTRICT_INCIDENT_CLUSTERS, CITY_BACKGROUND_COUNT, 0)
    );
  }
  const features: CatalogFeature[] = [];
  for (const cluster of CITY_INCIDENT_CLUSTERS) {
    const columns = Math.ceil(Math.sqrt(cluster.count));
    const rows = Math.ceil(cluster.count / columns);
    for (let index = 0; index < cluster.count; index++) {
      const column =
        cluster.column +
        (((index % columns) + 0.5 + (generator.next() - 0.5) * 2 * CITY_CLUSTER_JITTER) / columns -
          0.5) *
          CITY_CLUSTER_WIDTH;
      const row =
        cluster.row +
        ((Math.floor(index / columns) + 0.5 + (generator.next() - 0.5) * 2 * CITY_CLUSTER_JITTER) /
          rows -
          0.5) *
          CITY_CLUSTER_WIDTH;
      const [longitude, latitude] = getDistrictPosition(column, row);
      features.push(
        makePointFeature(longitude, latitude, {
          severity: Math.max(1, Math.min(10, Math.round(6 + generator.nextGaussian() * 1.5))),
          observed: formatObservedTime(generator.next() * 60 * 24 * 30)
        })
      );
    }
  }
  for (let row = 0; row < CITY_BACKGROUND_ROWS; row++) {
    for (let column = 0; column < CITY_BACKGROUND_COLUMNS; column++) {
      const [longitude, latitude] = getDistrictPosition(
        ((column + 0.5 + (generator.next() - 0.5) * 2 * CITY_BACKGROUND_JITTER) *
          DISTRICT_COLUMNS) /
          CITY_BACKGROUND_COLUMNS,
        ((row + 0.5 + (generator.next() - 0.5) * 2 * CITY_BACKGROUND_JITTER) * DISTRICT_ROWS) /
          CITY_BACKGROUND_ROWS
      );
      features.push(
        makePointFeature(longitude, latitude, {
          severity: Math.max(1, Math.min(10, Math.round(3 + generator.nextGaussian() * 1.5))),
          observed: formatObservedTime(generator.next() * 60 * 24 * 30)
        })
      );
    }
  }
  return makeCollection(features);
}

/** A rectangle polygon whose corners are computed from lattice indices, so neighbours share exact vertices. */
function makeLatticePolygon(
  getLongitude: (column: number) => number,
  getLatitude: (row: number) => number,
  column: number,
  row: number,
  properties: Record<string, unknown>
): CatalogFeature {
  const west = getLongitude(column);
  const east = getLongitude(column + 1);
  const south = getLatitude(row);
  const north = getLatitude(row + 1);
  return {
    type: 'Feature',
    geometry: {
      type: 'Polygon',
      coordinates: [
        [
          [west, south],
          [east, south],
          [east, north],
          [west, north],
          [west, south]
        ]
      ]
    },
    properties
  } as CatalogFeature;
}

/** True when the district lies in the planted high-rate block or is the lone high district. */
function isElevatedDistrict(column: number, row: number): boolean {
  const inBlock =
    column >= CITY_ELEVATED_BLOCK.firstColumn &&
    column <= CITY_ELEVATED_BLOCK.lastColumn &&
    row >= CITY_ELEVATED_BLOCK.firstRow &&
    row <= CITY_ELEVATED_BLOCK.lastRow;
  const isLone = column === CITY_LONE_HIGH_DISTRICT.column && row === CITY_LONE_HIGH_DISTRICT.row;
  return inBlock || isLone;
}

/** True when the district touches the lone high district, including diagonally. */
function isLowRingDistrict(column: number, row: number): boolean {
  return (
    Math.abs(column - CITY_LONE_HIGH_DISTRICT.column) <= 1 &&
    Math.abs(row - CITY_LONE_HIGH_DISTRICT.row) <= 1 &&
    !(column === CITY_LONE_HIGH_DISTRICT.column && row === CITY_LONE_HIGH_DISTRICT.row)
  );
}

/**
 * A Poisson deviate with the given mean, exact: a mean above 20 is split into independent Poisson
 * parts of at most 20 (a sum of independent Poisson variables is Poisson), each drawn by Knuth's
 * multiplication of uniforms.
 */
function drawPoisson(generator: SeededGenerator, mean: number): number {
  let total = 0;
  let remaining = mean;
  while (remaining > 0) {
    const part = Math.min(remaining, 20);
    remaining -= part;
    const limit = Math.exp(-part);
    let count = 0;
    let product = generator.next();
    while (product > limit) {
      count++;
      product *= generator.next();
    }
    total += count;
  }
  return total;
}

/** Populations of the ordinary districts run from 300 to 12,000 (log-uniform), so small districts have noisy rates. */
const DISTRICT_MINIMUM_POPULATION = 300;
const DISTRICT_MAXIMUM_POPULATION = 12_000;

/** The planted districts are urban cores of 3,000 to 12,000 residents, so the planted rate is not buried in small-count noise. */
const PLANTED_DISTRICT_MINIMUM_POPULATION = 3000;

/**
 * Districts with cases and residents. Cases are Poisson counts of population times rate, so the raw
 * rate of a district of a few hundred residents is visibly noisy (a handful of cases) and
 * empirical Bayes smoothing visibly helps. The planted districts have three times the baseline
 * rate, the low ring 0.4 times.
 */
function createCityDistrictRates(seedOffset = 0): CatalogFeatureCollection {
  const generator = createSeededGenerator(20250102 + seedOffset);
  const features: CatalogFeature[] = [];
  for (let row = 0; row < DISTRICT_ROWS; row++) {
    for (let column = 0; column < DISTRICT_COLUMNS; column++) {
      const isPlanted = isElevatedDistrict(column, row);
      const population = Math.round(
        isPlanted
          ? PLANTED_DISTRICT_MINIMUM_POPULATION +
              generator.next() * (DISTRICT_MAXIMUM_POPULATION - PLANTED_DISTRICT_MINIMUM_POPULATION)
          : Math.exp(
              Math.log(DISTRICT_MINIMUM_POPULATION) +
                generator.next() *
                  (Math.log(DISTRICT_MAXIMUM_POPULATION) - Math.log(DISTRICT_MINIMUM_POPULATION))
            )
      );
      const rate =
        BASELINE_RATE *
        (isPlanted
          ? ELEVATED_RATE_FACTOR
          : isLowRingDistrict(column, row)
            ? LOW_RING_RATE_FACTOR
            : 1);
      const cases = drawPoisson(generator, population * rate);
      features.push(
        makeLatticePolygon(
          column_ => DISTRICT_ORIGIN_LONGITUDE + column_ * DISTRICT_LONGITUDE_STEP,
          row_ => DISTRICT_ORIGIN_LATITUDE + row_ * DISTRICT_LATITUDE_STEP,
          column,
          row,
          {
            district: `D${String(row).padStart(2, '0')}-${String(column).padStart(2, '0')}`,
            districtColumn: column,
            districtRow: row,
            cases,
            population,
            rate: (cases / population) * 1000
          }
        )
      );
    }
  }
  return makeCollection(features);
}

/** A smooth surface over the district lattice, between about -1 and 1: one slow wave in each direction. */
function evaluateDistrictSurface(column: number, row: number): number {
  return Math.sin((column + 0.5) / 3) * Math.cos((row + 0.5) / 4);
}

/** Districts of the 12 by 12 lattice, each with the properties the callback returns for its column and row. */
function createDistrictLattice(
  getProperties: (column: number, row: number) => Record<string, unknown>
): CatalogFeatureCollection {
  const features: CatalogFeature[] = [];
  for (let row = 0; row < DISTRICT_ROWS; row++)
    for (let column = 0; column < DISTRICT_COLUMNS; column++)
      features.push(
        makeLatticePolygon(
          column_ => DISTRICT_ORIGIN_LONGITUDE + column_ * DISTRICT_LONGITUDE_STEP,
          row_ => DISTRICT_ORIGIN_LATITUDE + row_ * DISTRICT_LATITUDE_STEP,
          column,
          row,
          getProperties(column, row)
        )
      );
  return makeCollection(features);
}

/**
 * Districts with an outcome that depends on income and density, with or without a smooth omitted
 * surface. Both variants draw the same income, density and noise (one generator, the same order),
 * so they differ only by the omitted surface.
 */
function createDistrictRegression(
  withOmittedSurface: boolean,
  seedOffset = 0
): CatalogFeatureCollection {
  const generator = createSeededGenerator(20250108 + seedOffset);
  return createDistrictLattice((column, row) => {
    const income = 50 + generator.nextGaussian() * 10;
    const density = 20 + generator.nextGaussian() * 5;
    const noise = generator.nextGaussian() * DISTRICT_REGRESSION_PLANTED.noiseDeviation;
    const outcome =
      DISTRICT_REGRESSION_PLANTED.intercept +
      DISTRICT_REGRESSION_PLANTED.income * income +
      DISTRICT_REGRESSION_PLANTED.density * density +
      noise +
      (withOmittedSurface
        ? DISTRICT_OMITTED_SURFACE_AMPLITUDE * evaluateDistrictSurface(column, row)
        : 0);
    return {
      district: `D${String(row).padStart(2, '0')}-${String(column).padStart(2, '0')}`,
      outcome: Math.round(outcome * 1000) / 1000,
      income: Math.round(income * 100) / 100,
      density: Math.round(density * 100) / 100
    };
  });
}

/** Districts with a value that varies smoothly over space, plus a little noise: association that fades with the order of adjacency. */
function createDistrictSmoothSurface(seedOffset = 0): CatalogFeatureCollection {
  const generator = createSeededGenerator(20250110 + seedOffset);
  return createDistrictLattice((column, row) => ({
    district: `D${String(row).padStart(2, '0')}-${String(column).padStart(2, '0')}`,
    value:
      Math.round(
        (10 + 5 * evaluateDistrictSurface(column, row) + generator.nextGaussian() * 0.3) * 1000
      ) / 1000
  }));
}

/** Positions of the three planted gauge peaks are in metres from the sample centre; see the surface below. */
const GAUGE_EXTENT_METERS = 10_000;
const GAUGE_COUNT = 420;
const GAUGE_NOISE = 0.4;

function getMetersPerDegreeLongitude(latitude: number): number {
  return METERS_PER_DEGREE_LATITUDE * Math.cos((latitude * Math.PI) / 180);
}

/**
 * The smooth surface the gauges sample: a wave across the area plus one broad peak. Positions are
 * metres east and north of the city centre. The held-out check evaluates it at grid cell centres.
 */
export function evaluateKeplerCuratedGaugeSurface(longitude: number, latitude: number): number {
  const eastMeters =
    (longitude - CITY_CENTER_LONGITUDE) * getMetersPerDegreeLongitude(CITY_CENTER_LATITUDE);
  const northMeters = (latitude - CITY_CENTER_LATITUDE) * METERS_PER_DEGREE_LATITUDE;
  const wave =
    8 * Math.sin((2 * Math.PI * eastMeters) / 9000) * Math.cos((2 * Math.PI * northMeters) / 7000);
  const peak =
    6 * Math.exp(-((eastMeters - 1500) ** 2 + (northMeters + 1000) ** 2) / (2 * 1500 * 1500));
  return 12 + wave + peak;
}

/** Rain gauges scattered over a 10 km square, each reading the smooth surface plus small noise. */
function createRainGauges(seedOffset = 0): CatalogFeatureCollection {
  const generator = createSeededGenerator(20250103 + seedOffset);
  const features: CatalogFeature[] = [];
  for (let index = 0; index < GAUGE_COUNT; index++) {
    const eastMeters = (generator.next() - 0.5) * GAUGE_EXTENT_METERS;
    const northMeters = (generator.next() - 0.5) * GAUGE_EXTENT_METERS;
    const longitude =
      CITY_CENTER_LONGITUDE + eastMeters / getMetersPerDegreeLongitude(CITY_CENTER_LATITUDE);
    const latitude = CITY_CENTER_LATITUDE + northMeters / METERS_PER_DEGREE_LATITUDE;
    features.push(
      makePointFeature(longitude, latitude, {
        rainfall:
          Math.round(
            (evaluateKeplerCuratedGaugeSurface(longitude, latitude) +
              generator.nextGaussian() * GAUGE_NOISE) *
              100
          ) / 100
      })
    );
  }
  return makeCollection(features);
}

/** The street grid: 11 by 11 junctions 200 m apart. */
export const STREET_GRID = {size: 11, spacingMeters: 200} as const;

/** Junction indices of the three facilities; the first sits in the densest demand. */
export const STREET_GRID_FACILITIES = [
  {name: 'depot-a', column: 2, row: 2},
  {name: 'depot-b', column: 8, row: 3},
  {name: 'depot-c', column: 5, row: 9}
] as const;

/** Junctions within this many blocks (Manhattan) of the first facility carry ten times the residents. */
export const STREET_GRID_DENSE_BLOCKS = 2;

function getJunctionPosition(column: number, row: number): [number, number] {
  const halfSpan = ((STREET_GRID.size - 1) / 2) * STREET_GRID.spacingMeters;
  return [
    CITY_CENTER_LONGITUDE +
      (column * STREET_GRID.spacingMeters - halfSpan) /
        getMetersPerDegreeLongitude(CITY_CENTER_LATITUDE),
    CITY_CENTER_LATITUDE + (row * STREET_GRID.spacingMeters - halfSpan) / METERS_PER_DEGREE_LATITUDE
  ];
}

/** Residents at a junction: one hundred, or one thousand near the first facility. */
export function getStreetGridResidents(column: number, row: number): number {
  const distance =
    Math.abs(column - STREET_GRID_FACILITIES[0].column) +
    Math.abs(row - STREET_GRID_FACILITIES[0].row);
  return distance <= STREET_GRID_DENSE_BLOCKS ? 1000 : 100;
}

/** Streets as one two-vertex line per block, so neighbouring lines share exact junction coordinates. */
function createStreetGridFacilities(seedOffset = 0): Record<string, CatalogFeatureCollection> {
  const generator = createSeededGenerator(20250111 + seedOffset);
  const streets: CatalogFeature[] = [];
  for (let row = 0; row < STREET_GRID.size; row++) {
    for (let column = 0; column < STREET_GRID.size; column++) {
      if (column + 1 < STREET_GRID.size)
        streets.push({
          type: 'Feature',
          geometry: {
            type: 'LineString',
            coordinates: [getJunctionPosition(column, row), getJunctionPosition(column + 1, row)]
          },
          properties: {street: `east-west-${row}-${column}`}
        } as CatalogFeature);
      if (row + 1 < STREET_GRID.size)
        streets.push({
          type: 'Feature',
          geometry: {
            type: 'LineString',
            coordinates: [getJunctionPosition(column, row), getJunctionPosition(column, row + 1)]
          },
          properties: {street: `north-south-${row}-${column}`}
        } as CatalogFeature);
    }
  }
  const facilities = STREET_GRID_FACILITIES.map(facility => {
    const [longitude, latitude] = getJunctionPosition(facility.column, facility.row);
    return makePointFeature(longitude, latitude, {name: facility.name});
  });
  const demand: CatalogFeature[] = [];
  for (let row = 0; row < STREET_GRID.size; row++)
    for (let column = 0; column < STREET_GRID.size; column++) {
      const [longitude, latitude] = getJunctionPosition(column, row);
      demand.push(
        makePointFeature(longitude, latitude, {
          residents:
            seedOffset === 0
              ? getStreetGridResidents(column, row)
              : drawPoisson(generator, getStreetGridResidents(column, row)),
          junctionColumn: column,
          junctionRow: row
        })
      );
    }
  return {
    network: makeCollection(streets),
    facilities: makeCollection(facilities),
    demand: makeCollection(demand)
  };
}

/**
 * Two periods of incidents. Before: clusters in the north-west, south-east and south-west.
 * After: the south-west cluster is gone and a new one has appeared in the north-east.
 */
export const BEFORE_AFTER_CLUSTERS = {
  before: [
    {name: 'north-west', column: 2.5, row: 9.5, count: 150},
    {name: 'south-east', column: 9.5, row: 3.5, count: 150},
    {name: 'south-west', column: 3.5, row: 2.5, count: 150}
  ],
  after: [
    {name: 'north-west', column: 2.5, row: 9.5, count: 150},
    {name: 'south-east', column: 9.5, row: 3.5, count: 150},
    {name: 'north-east', column: 9.5, row: 9.5, count: 150}
  ]
} as const;

function createBeforeAfterIncidents(seedOffset = 0): Record<string, CatalogFeatureCollection> {
  const beforeGenerator = createSeededGenerator(20250104 + seedOffset);
  const afterGenerator = createSeededGenerator(20250105 + seedOffset);
  return {
    before: makeCollection(
      createClusteredIncidents(beforeGenerator, BEFORE_AFTER_CLUSTERS.before, 40, 0)
    ),
    after: makeCollection(
      createClusteredIncidents(afterGenerator, BEFORE_AFTER_CLUSTERS.after, 40, 60 * 24 * 365)
    )
  };
}

/**
 * Four facilities with observations around each, whose waiting time averages ten, twenty, thirty
 * and forty minutes by facility. Positions are metres east and north of the city centre.
 */
export const FACILITY_OBSERVATION_GROUPS = [
  {name: 'facility-a', eastMeters: -2500, northMeters: 2000, meanMinutes: 10},
  {name: 'facility-b', eastMeters: 2500, northMeters: 2000, meanMinutes: 20},
  {name: 'facility-c', eastMeters: -2500, northMeters: -2000, meanMinutes: 30},
  {name: 'facility-d', eastMeters: 2500, northMeters: -2000, meanMinutes: 40}
] as const;

const OBSERVATIONS_PER_FACILITY = 70;
const OBSERVATION_SPREAD_METERS = 500;

function createFacilityObservations(seedOffset = 0): Record<string, CatalogFeatureCollection> {
  const generator = createSeededGenerator(20250106 + seedOffset);
  const metersPerDegreeLongitude = getMetersPerDegreeLongitude(CITY_CENTER_LATITUDE);
  const facilities: CatalogFeature[] = [];
  const observations: CatalogFeature[] = [];
  for (const group of FACILITY_OBSERVATION_GROUPS) {
    facilities.push(
      makePointFeature(
        CITY_CENTER_LONGITUDE + group.eastMeters / metersPerDegreeLongitude,
        CITY_CENTER_LATITUDE + group.northMeters / METERS_PER_DEGREE_LATITUDE,
        {name: group.name}
      )
    );
    for (let index = 0; index < OBSERVATIONS_PER_FACILITY; index++) {
      const eastMeters = group.eastMeters + generator.nextGaussian() * OBSERVATION_SPREAD_METERS;
      const northMeters = group.northMeters + generator.nextGaussian() * OBSERVATION_SPREAD_METERS;
      observations.push(
        makePointFeature(
          CITY_CENTER_LONGITUDE + eastMeters / metersPerDegreeLongitude,
          CITY_CENTER_LATITUDE + northMeters / METERS_PER_DEGREE_LATITUDE,
          {waitMinutes: group.meanMinutes + generator.nextGaussian()}
        )
      );
    }
  }
  return {facilities: makeCollection(facilities), observations: makeCollection(observations)};
}

/**
 * Old boundaries (4 by 4 wards) and new ones (5 by 5 precincts) over the same rectangle. Residents
 * sit on the wards; the transfer to precincts should keep the total.
 */
export const BOUNDARY_PAIR = {sourceSize: 4, targetSize: 5} as const;

function createBoundaryPair(seedOffset = 0): Record<string, CatalogFeatureCollection> {
  const generator = createSeededGenerator(20250107 + seedOffset);
  const spanLongitude = DISTRICT_COLUMNS * DISTRICT_LONGITUDE_STEP;
  const spanLatitude = DISTRICT_ROWS * DISTRICT_LATITUDE_STEP;
  const makeLattice = (
    size: number,
    makeProperties: (column: number, row: number) => Record<string, unknown>
  ): CatalogFeatureCollection => {
    const features: CatalogFeature[] = [];
    for (let row = 0; row < size; row++)
      for (let column = 0; column < size; column++)
        features.push(
          makeLatticePolygon(
            index => DISTRICT_ORIGIN_LONGITUDE + (index / size) * spanLongitude,
            index => DISTRICT_ORIGIN_LATITUDE + (index / size) * spanLatitude,
            column,
            row,
            makeProperties(column, row)
          )
        );
    return makeCollection(features);
  };
  return {
    sources: makeLattice(BOUNDARY_PAIR.sourceSize, (column, row) => ({
      ward: `W${row}${column}`,
      residents: Math.round(8000 + generator.next() * 12000)
    })),
    targets: makeLattice(BOUNDARY_PAIR.targetSize, (column, row) => ({
      precinct: `P${row}${column}`
    }))
  };
}

/**
 * Parameter values that a sample needs because the data-derived default is known to be poor for
 * it, by analysis id. A user gets the derived defaults, so each entry here says what is wrong with
 * the default and what should fix it; the specification runs the analyses at the derived defaults
 * and uses these only where listed.
 */
export function getKeplerCuratedAnalysisSampleParameters(
  analysisId: string
): Readonly<Record<string, CatalogParameterValue>> {
  switch (analysisId) {
    case 'emerging-hot-spots':
      return {coordinateSystem: 'planar', cellSize: 1};
    case 'track-quality':
      return {coordinateSystem: 'planar'};
    case 'regional-raster-change':
      return {confirmComparable: true, threshold: 1};
    case 'surface-preparation':
      return {maximumDistance: 1, method: 'nearest'};
    case 'zonal-exposure':
      return createZonalExposureSample().parameters;
    case 'drive-time-bands':
      return {
        ...DRIVE_TIME_BANDS_SAMPLE.parameters,
        costBreaks: [...DRIVE_TIME_BANDS_SAMPLE.parameters.costBreaks]
      };
    case 'network-disruption':
      return {...NETWORK_DISRUPTION_SAMPLE.parameters};
    case 'dasymetric-redistribution':
      return createDasymetricRedistributionSample().parameters;
    default:
      return {};
  }
}

/** Builds a sample's datasets, keyed by the slot name of the curated analyses that use it. */
export function createKeplerCuratedAnalysisSample(
  sampleId: KeplerCuratedFeatureAnalysisSampleId,
  seedOffset?: number
): Readonly<Record<string, CatalogFeatureCollection>>;
export function createKeplerCuratedAnalysisSample(
  sampleId: string,
  seedOffset?: number
): Readonly<Record<string, CatalogDataInput>>;
export function createKeplerCuratedAnalysisSample(
  sampleId: string,
  /** Added to feature seeds; nonzero selects the independent fixture for typed samples. 0 is the bundled sample. */
  seedOffset = 0
): Readonly<Record<string, CatalogDataInput>> {
  switch (sampleId) {
    case 'drainage-network-slope':
      return seedOffset
        ? {elevation: createDrainageNetworkSecondDataset().survey}
        : createDrainageNetworkSample();
    case 'surface-preparation-gaps':
      return {
        surface: seedOffset ? makeSurfacePreparationSecondSample() : makeSurfacePreparationSample()
      };
    case 'regional-raster-change':
      return createRegionalRasterChangeSample(seedOffset !== 0);
    case 'zonal-exposure-counts':
      return seedOffset
        ? createMappedZonalExposureSample().inputs
        : createZonalExposureSample().inputs;
    case 'network-disruption-bridge':
      return seedOffset ? createNetworkDisruptionSecondDataset() : createNetworkDisruptionSample();
    case 'cell-flow-topology':
      return {
        flows: seedOffset ? createSecondCellFlowTopologySample() : createCellFlowTopologySample()
      };
    case 'drive-time-bands-streets':
      return seedOffset ? createDriveTimeBandsSecondDataset() : createDriveTimeBandsSample();
    case 'fleet-dwell':
      return createFleetDwellSample(seedOffset);
    case 'landscape-metrics':
      return createLandscapeMetricsSample(seedOffset ? 'second' : 'sample');
    case 'dasymetric-fractional-support':
      return createDasymetricRedistributionSample(seedOffset !== 0).inputs;
    case 'emerging-hot-spots':
      return {points: createEmergingHotSpotsSample(seedOffset)};
    case 'track-quality':
      return {tracks: createTrackQualitySample(seedOffset)};
    case 'city-incidents':
      return {points: createCityIncidents(seedOffset)};
    case 'city-district-rates':
      return {areas: createCityDistrictRates(seedOffset)};
    case 'city-district-incidents':
      return {
        points: createCityIncidents(seedOffset, true),
        areas: createCityDistrictRates(seedOffset)
      };
    case 'rain-gauges':
      return {points: createRainGauges(seedOffset)};
    case 'street-grid-facilities':
      return createStreetGridFacilities(seedOffset);
    case 'before-after-incidents':
      return createBeforeAfterIncidents(seedOffset);
    case 'facility-observations':
      return createFacilityObservations(seedOffset);
    case 'boundary-pair':
      return createBoundaryPair(seedOffset);
    case 'district-regression':
      return {areas: createDistrictRegression(false, seedOffset)};
    case 'district-regression-omitted-variable':
      return {areas: createDistrictRegression(true, seedOffset)};
    case 'district-smooth-surface':
      return {areas: createDistrictSmoothSurface(seedOffset)};
    default:
      throw new Error(
        `createKeplerCuratedAnalysisSample: unknown sampleId "${sampleId}"; accepted: ${KEPLER_CURATED_ANALYSIS_SAMPLE_IDS.join(', ')}`
      );
  }
}
