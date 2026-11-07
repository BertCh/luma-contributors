// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {toDegrees, toRadians} from '@math.gl/core';
import {EARTH_RADIUS_METERS} from '../analysis/catalog-geodesy';
import {KeplerAllocationError} from '../runtime/dataset-registry';
import {
  parseKeplerDorlingSettings,
  type KeplerDorlingSettings
} from '../analysis/dorling-cartogram';
import type {GPUVariableStandardizationMethod} from '@luma-contributors/graph/gpu-dataframe';

import {
  getKeplerPointWeightCoordinateKeys,
  parseKeplerPointWeightCoordinates,
  parseKeplerPointWeightRadius,
  type KeplerPointWeightCoordinates
} from '../analysis/point-weight-coordinates';

export type KeplerDatasetReference = Readonly<{datasetId: string; revision: number}>;
export type KeplerBackendPolicy = 'prefer-gpu' | 'require-gpu' | 'complementary';
export type KeplerAnalysisPopulation = 'all' | 'selected';
export type KeplerClassificationMethod = 'equal-interval' | 'quantile' | 'natural-breaks';
export type KeplerMultipleTestingMethod =
  | 'none'
  | 'bonferroni'
  | 'holm'
  | 'benjamini-hochberg'
  | 'benjamini-yekutieli';

/** Reproducible binary source-wide neighbors with explicit coordinate and metric declarations. */
export type KeplerPointWeightsSettings = Readonly<
  KeplerPointWeightCoordinates &
    (
      | {kind: 'knn'; k: number}
      | {
          kind: 'distance-band';
          radiusMeters: number;
          /** Total CSR slot capacity; an overflow result is incomplete and must not be published. */
          neighborCapacity: number;
          /**
           * Permutation scratch slots per row (64 to 512, default 64). Rows with more active
           * neighbors report inference overflow instead of being truncated.
           */
          permutationDegree?: number;
        }
    )
>;

export const DEFAULT_KEPLER_PERMUTATION_DEGREE = 64;
export const MAXIMUM_KEPLER_PERMUTATION_DEGREE = 512;

export const MAXIMUM_KEPLER_DISTANCE_BAND_CAPACITY = 2 ** 25;
/** uint32 row ids; a wrapping total slot count is detected by an exact 64-bit guard above 65536 rows. */
export const MAXIMUM_KEPLER_DISTANCE_BAND_ROWS = 2 ** 24;

export const MAXIMUM_KEPLER_POLYGON_ROWS = 4096;
export const MAXIMUM_KEPLER_POLYGON_VERTICES = 65536;
export const MAXIMUM_KEPLER_POLYGON_NEIGHBOR_CAPACITY = 262144;
export const MAXIMUM_KEPLER_POLYGON_PAIR_CAPACITY = 1048576;

export type KeplerPolygonWeightsSettings = Readonly<{
  kind: 'contiguity';
  geometryColumn: string;
  criterion: 'queen' | 'rook';
  equality: 'exact-source-xy';
  neighborCapacity: number;
  pairCapacity: number;
}>;

/** Bounds of the opt-in Local Moran conditional permutation adapter. */
export const MAXIMUM_KEPLER_LOCAL_MORAN_PERMUTATIONS = 9999;
export const MAXIMUM_KEPLER_LOCAL_MORAN_PERMUTATION_ROWS = 2 ** 24 - 1;

/** Explicit per-analysis work limit of the Kepler Quantile LISA adapter. */
export const MAXIMUM_KEPLER_QUANTILE_LISA_PERMUTATIONS = 9999;

export type KeplerQuantileLISASettings = Readonly<{
  quantileCount: number;
  selectedQuantile: number;
  seed: number;
  permutationCount: number;
  significanceLevel: number;
  inference: 'legacy-directed-pseudo';
  multipleTesting: 'none';
  islands: 'exclude';
  quantileMethod: 'type-5';
}>;

/** Geographic interpolation limits: uint32 rows and one device storage binding of cells. */
export const MAXIMUM_KEPLER_IDW_SOURCE_ROWS = 2 ** 24 - 1;
export const MAXIMUM_KEPLER_IDW_CELLS = 2 ** 24;
export const MAXIMUM_KEPLER_IDW_GRID_DIMENSION = 8192;

export type KeplerIDWSettings = Readonly<{
  longitudeColumn: string;
  latitudeColumn: string;
  coordinateSpace: 'longitude-latitude';
  metric: 'great-circle';
  units: 'meters';
  grid: Readonly<{
    bounds: readonly [number, number, number, number];
    width: number;
    height: number;
  }>;
  searchRadius: number;
  power: number;
  neighborCount: number;
  minimumNeighborCount: number;
  taper: 'none' | 'modified-shepard';
}>;

export const MAXIMUM_KEPLER_KRIGING_SOURCE_ROWS = 2 ** 24 - 1;
export const MAXIMUM_KEPLER_KRIGING_CELLS = 2 ** 24;
export const MAXIMUM_KEPLER_KRIGING_GRID_DIMENSION = 8192;
export const MAXIMUM_KEPLER_KRIGING_NEIGHBORS = 16;

export type KeplerKrigingSettings = Readonly<{
  longitudeColumn: string;
  latitudeColumn: string;
  coordinateSpace: 'longitude-latitude';
  metric: 'great-circle';
  units: 'meters';
  grid: Readonly<{
    bounds: readonly [number, number, number, number];
    width: number;
    height: number;
  }>;
  searchRadiusMeters: number;
  neighborCount: number;
  minimumNeighbors: number;
  variogram: Readonly<{
    model: 'spherical' | 'exponential' | 'gaussian';
    nugget: number;
    sill: number;
    rangeMeters: number;
  }>;
  /** Two-sided interval level for kriging-variance prediction intervals (0, 1). Optional. */
  confidenceLevel?: number;
  /** Also run leave-one-out cross-validation over the retained samples. Optional. */
  crossValidation?: boolean;
  /** Provenance of an automatic weighted-least-squares fit that produced `variogram`. Optional. */
  variogramFit?: Readonly<{
    method: 'weighted-least-squares';
    weighting: 'cressie';
    binCount: number;
    pairCount: number;
    maximumLagMeters: number;
    criterion: number;
  }>;
}>;

export const MAXIMUM_KEPLER_DBSCAN_SOURCE_ROWS = 4096;
export const MAXIMUM_KEPLER_DBSCAN_MINIMUM_SAMPLES = 256;

export type KeplerDBSCANSettings = Readonly<{
  longitudeColumn: string;
  latitudeColumn: string;
  coordinateSpace: 'longitude-latitude';
  metric: 'great-circle';
  units: 'meters';
  epsilon: number;
  /** Includes the focus point. */
  minimumSamples: number;
}>;

export const MAXIMUM_KEPLER_OLS_SOURCE_ROWS = 65536;
export const MAXIMUM_KEPLER_OLS_PREDICTORS = 15;

export type KeplerOLSSettings = Readonly<{
  responseColumn: string;
  predictorColumns: readonly string[];
  intercept: 'included';
  ridgeLambda: number;
  variance: 'degrees-of-freedom';
  missing: 'complete-cases';
}>;

export type KeplerPolygonAnalysisOperation =
  | (KeplerQuantileLISASettings & {
      kind: 'quantile-lisa-polygons';
      column: string;
      weights: KeplerPolygonWeightsSettings;
    })
  | {
      kind: 'spatial-rate-polygons';
      events: string;
      populations: string;
      weights: KeplerPolygonWeightsSettings;
      scale: number;
      islandPolicy: 'undefined' | 'own-rate';
    }
  | {
      kind: 'hot-spots-polygons';
      column: string;
      weights: KeplerPolygonWeightsSettings;
      minimumNeighbors: number;
      selfWeight: number;
      multipleTesting: KeplerMultipleTestingMethod;
      islands: 'exclude' | 'keep';
      inference: 'analytic-two-sided';
    }
  | {
      kind: 'local-moran-polygons';
      column: string;
      weights: KeplerPolygonWeightsSettings;
      minimumNeighbors: number;
      multipleTesting: KeplerMultipleTestingMethod;
      islands: 'exclude' | 'keep';
      significanceLevel: number;
      inference: 'analytic-conditional-normal-two-sided' | 'permutation-conditional-two-sided';
      permutationCount?: number;
      seed?: number;
    };

/** Save field names and scientific choices, never GPU handles or physical row positions. */
export type KeplerAnalysisOperation =
  | KeplerPolygonAnalysisOperation
  | (KeplerDorlingSettings & {kind: 'dorling-cartogram'})
  | (KeplerOLSSettings & {kind: 'ols'})
  | (KeplerDBSCANSettings & {kind: 'dbscan-points'})
  | (KeplerIDWSettings & {kind: 'idw-points'; column: string})
  | (KeplerKrigingSettings & {kind: 'kriging-points'; column: string})
  | {
      kind: 'standardize';
      column: string;
      method: GPUVariableStandardizationMethod;
      variance: 'sample' | 'population';
    }
  | {kind: 'excess-risk'; events: string; populations: string}
  | (KeplerQuantileLISASettings & {
      kind: 'quantile-lisa-points';
      column: string;
      weights: KeplerPointWeightsSettings;
    })
  | {
      kind: 'local-moran-points';
      column: string;
      /** Present for bivariate Local Moran: x is `column`, the weighted lag is taken over this y. */
      secondColumn?: string;
      weights: KeplerPointWeightsSettings;
      minimumNeighbors: number;
      islands: 'exclude' | 'keep';
      multipleTesting: KeplerMultipleTestingMethod;
      significanceLevel: number;
      inference: 'analytic-conditional-normal-two-sided' | 'permutation-conditional-two-sided';
      permutationCount?: number;
      seed?: number;
    }
  | {
      kind: 'multivariate-local-geary-points';
      /** Two to eight variables standardized jointly; row-major vectors are permuted whole. */
      columns: string[];
      weights: KeplerPointWeightsSettings;
      significanceLevel: number;
      permutationCount: number;
      seed: number;
      correction: 'none' | 'bonferroni';
      tail: 'equal-tail' | 'esda-min-tail';
    }
  | {
      kind: 'hot-spots-points';
      column: string;
      weights: KeplerPointWeightsSettings;
      minimumNeighbors: number;
      selfWeight: number;
      multipleTesting: KeplerMultipleTestingMethod;
      inference: 'analytic-two-sided';
      islands: 'exclude' | 'keep';
    }
  | {
      kind: 'spatial-rate-points';
      events: string;
      populations: string;
      weights: KeplerPointWeightsSettings;
      scale: number;
      islandPolicy: 'undefined' | 'own-rate';
    }
  | {
      kind: 'spatial-rate';
      events: string;
      populations: string;
      weightsId: string;
      weightsRevision: number;
      scale: number;
      islandPolicy: 'undefined' | 'own-rate';
    }
  | {
      kind: 'classify';
      column: string;
      method: KeplerClassificationMethod;
      classCount: number;
      naturalBreaksMode: 'approximate' | 'exact';
    }
  | {
      kind: 'hot-spots-lattice';
      column: string;
      width: number;
      height: number;
      criterion: 'rook' | 'queen';
      radius: number;
      neighborCapacity: number;
      minimumNeighbors: number;
      selfWeight: number;
      multipleTesting: KeplerMultipleTestingMethod;
      inference: 'analytic-two-sided';
    };

/**
 * Display-only georeferencing of a Dorling cartogram computed in a projected CRS that is not
 * EPSG:3857: the CRS unit size, plane coordinates of an anchor and the anchor's longitude and
 * latitude. It never changes computed values, so hosts keep it out of execution keys.
 */
export type KeplerCartogramDeclaration = Readonly<{
  metersPerUnit: number;
  planeOrigin: readonly [number, number];
  anchor: readonly [number, number];
}>;
export type KeplerAnalysisDisplay = Readonly<{cartogramDeclaration?: KeplerCartogramDeclaration}>;

export type KeplerAnalysisRecipe = Readonly<{
  schemaVersion: 1;
  algorithmVersion: 1;
  id: string;
  source: KeplerDatasetReference;
  population: KeplerAnalysisPopulation;
  /** Required for a selected population; identifies the reproducible mask snapshot. */
  selectionRevision?: number;
  backendPolicy: KeplerBackendPolicy;
  operation: KeplerAnalysisOperation;
  /** Presentation choices saved with the recipe; absent when there are none. */
  display?: KeplerAnalysisDisplay;
}>;

export type KeplerRecipeRestoreContext = {
  resolveDatasetRevision: (datasetId: string) => number | undefined;
  resolveSelectionRevision?: (datasetId: string) => number | undefined;
  resolveWeightsRevision?: (weightsId: string) => number | undefined;
};

/** Validates and detaches a saved recipe; unknown fields are rejected to avoid losing semantics. */
export function parseKeplerAnalysisRecipe(value: unknown): KeplerAnalysisRecipe {
  const recipe = readObject(value);
  checkKeys(recipe, [
    'schemaVersion',
    'algorithmVersion',
    'id',
    'source',
    'population',
    'selectionRevision',
    'backendPolicy',
    'operation',
    'display'
  ]);
  if (recipe['schemaVersion'] !== 1 || recipe['algorithmVersion'] !== 1)
    throw new Error('Unsupported Kepler analysis recipe version');
  const source = readObject(recipe['source']);
  checkKeys(source, ['datasetId', 'revision']);
  const population = readChoice(recipe['population'], ['all', 'selected'] as const);
  const selectionRevision =
    recipe['selectionRevision'] === undefined
      ? undefined
      : readRevision(recipe['selectionRevision']);
  if ((population === 'selected') !== (selectionRevision !== undefined))
    throw new Error('Selected recipes require exactly one selection revision');
  const operation = readObject(recipe['operation']);
  const kind = readChoice(operation['kind'], [
    'standardize',
    'excess-risk',
    'spatial-rate',
    'spatial-rate-points',
    'spatial-rate-polygons',
    'hot-spots-polygons',
    'local-moran-polygons',
    'classify',
    'hot-spots-lattice',
    'hot-spots-points',
    'local-moran-points',
    'multivariate-local-geary-points',
    'quantile-lisa-points',
    'quantile-lisa-polygons',
    'idw-points',
    'kriging-points',
    'dbscan-points',
    'dorling-cartogram',
    'ols'
  ] as const);
  let parsedOperation: KeplerAnalysisOperation;
  switch (kind) {
    case 'dorling-cartogram': {
      const {kind: ignoredKind, ...settings} = operation;
      parsedOperation = {kind, ...parseKeplerDorlingSettings(settings)};
      break;
    }
    case 'ols': {
      const {kind: ignoredKind, ...settings} = operation;
      parsedOperation = {kind, ...parseKeplerOLSSettings(settings)};
      break;
    }
    case 'dbscan-points': {
      const {kind: ignoredKind, ...settings} = operation;
      parsedOperation = {kind, ...parseKeplerDBSCANSettings(settings)};
      break;
    }
    case 'kriging-points': {
      const {kind: ignoredKind, column, ...settings} = operation;
      parsedOperation = {kind, column: readName(column), ...parseKeplerKrigingSettings(settings)};
      break;
    }
    case 'idw-points': {
      const {kind: ignoredKind, column, ...settings} = operation;
      parsedOperation = {kind, column: readName(column), ...parseKeplerIDWSettings(settings)};
      break;
    }
    case 'quantile-lisa-points': {
      checkKeys(operation, [
        'kind',
        'column',
        'weights',
        'quantileCount',
        'selectedQuantile',
        'seed',
        'permutationCount',
        'significanceLevel',
        'inference',
        'multipleTesting',
        'islands',
        'quantileMethod'
      ]);
      const quantileCount = readInteger(operation['quantileCount'], 2, 2 ** 24 - 2);
      const significanceLevel = operation['significanceLevel'];
      if (
        typeof significanceLevel !== 'number' ||
        !(
          significanceLevel > 0 &&
          significanceLevel < 1 &&
          Math.fround(significanceLevel) > 0 &&
          Math.fround(significanceLevel) < 1
        )
      )
        throw new Error('Quantile LISA significance must remain in (0, 1) in Float32');
      parsedOperation = {
        kind,
        column: readName(operation['column']),
        weights: parseKeplerPointWeightsSettings(operation['weights']),
        quantileCount,
        selectedQuantile: readInteger(operation['selectedQuantile'], 1, quantileCount),
        seed: readInteger(operation['seed'], 0, Number.MAX_SAFE_INTEGER),
        permutationCount: readInteger(
          operation['permutationCount'],
          1,
          MAXIMUM_KEPLER_QUANTILE_LISA_PERMUTATIONS
        ),
        significanceLevel,
        inference: readChoice(operation['inference'], ['legacy-directed-pseudo'] as const),
        multipleTesting: readChoice(operation['multipleTesting'], ['none'] as const),
        islands: readChoice(operation['islands'], ['exclude'] as const),
        quantileMethod: readChoice(operation['quantileMethod'], ['type-5'] as const)
      };
      break;
    }
    case 'quantile-lisa-polygons': {
      checkKeys(operation, [
        'kind',
        'column',
        'weights',
        'quantileCount',
        'selectedQuantile',
        'seed',
        'permutationCount',
        'significanceLevel',
        'inference',
        'multipleTesting',
        'islands',
        'quantileMethod'
      ]);
      const quantileCount = readInteger(operation['quantileCount'], 2, 2 ** 24 - 2);
      const significanceLevel = operation['significanceLevel'];
      if (
        typeof significanceLevel !== 'number' ||
        !(
          significanceLevel > 0 &&
          significanceLevel < 1 &&
          Math.fround(significanceLevel) > 0 &&
          Math.fround(significanceLevel) < 1
        )
      )
        throw new Error('Quantile LISA significance must remain in (0, 1) in Float32');
      parsedOperation = {
        kind,
        column: readName(operation['column']),
        weights: parseKeplerPolygonWeightsSettings(operation['weights']),
        quantileCount,
        selectedQuantile: readInteger(operation['selectedQuantile'], 1, quantileCount),
        seed: readInteger(operation['seed'], 0, Number.MAX_SAFE_INTEGER),
        permutationCount: readInteger(
          operation['permutationCount'],
          1,
          MAXIMUM_KEPLER_QUANTILE_LISA_PERMUTATIONS
        ),
        significanceLevel,
        inference: readChoice(operation['inference'], ['legacy-directed-pseudo'] as const),
        multipleTesting: readChoice(operation['multipleTesting'], ['none'] as const),
        islands: readChoice(operation['islands'], ['exclude'] as const),
        quantileMethod: readChoice(operation['quantileMethod'], ['type-5'] as const)
      };
      break;
    }
    case 'local-moran-points': {
      checkKeys(operation, [
        'kind',
        'column',
        'secondColumn',
        'weights',
        'minimumNeighbors',
        'islands',
        'multipleTesting',
        'significanceLevel',
        'inference',
        'permutationCount',
        'seed'
      ]);
      const significanceLevel = operation['significanceLevel'];
      if (
        typeof significanceLevel !== 'number' ||
        !(
          significanceLevel > 0 &&
          significanceLevel < 1 &&
          Math.fround(significanceLevel) > 0 &&
          Math.fround(significanceLevel) < 1
        )
      )
        throw new Error('Local Moran significance must remain in (0, 1) in Float32');
      const inference = readChoice(operation['inference'], [
        'analytic-conditional-normal-two-sided',
        'permutation-conditional-two-sided'
      ] as const);
      const minimumNeighbors = readInteger(operation['minimumNeighbors'], 1, 2 ** 24 - 1);
      const permutation = inference === 'permutation-conditional-two-sided';
      if (permutation && minimumNeighbors !== 1)
        throw new Error('Local Moran permutations require minimumNeighbors = 1');
      if (!permutation && ('permutationCount' in operation || 'seed' in operation))
        throw new Error('Analytic Local Moran does not accept permutation controls');
      // Bivariate Local Moran has no analytic variance; only conditional permutations exist.
      if (operation['secondColumn'] !== undefined && !permutation)
        throw new Error('Bivariate Local Moran requires permutation inference');
      parsedOperation = {
        kind,
        column: readName(operation['column']),
        ...(operation['secondColumn'] === undefined
          ? {}
          : {secondColumn: readName(operation['secondColumn'])}),
        weights: parseKeplerPointWeightsSettings(operation['weights']),
        minimumNeighbors,
        islands: readChoice(operation['islands'], ['exclude', 'keep'] as const),
        multipleTesting: readChoice(operation['multipleTesting'], [
          'none',
          'bonferroni',
          'holm',
          'benjamini-hochberg',
          'benjamini-yekutieli'
        ] as const),
        significanceLevel,
        inference,
        ...(permutation
          ? {
              permutationCount: readInteger(
                operation['permutationCount'],
                1,
                MAXIMUM_KEPLER_LOCAL_MORAN_PERMUTATIONS
              ),
              seed: readInteger(operation['seed'], 0, 2 ** 32 - 1)
            }
          : {})
      };
      break;
    }
    case 'multivariate-local-geary-points': {
      checkKeys(operation, [
        'kind',
        'columns',
        'weights',
        'significanceLevel',
        'permutationCount',
        'seed',
        'correction',
        'tail'
      ]);
      const significanceLevel = operation['significanceLevel'];
      if (
        typeof significanceLevel !== 'number' ||
        !(
          significanceLevel > 0 &&
          significanceLevel < 1 &&
          Math.fround(significanceLevel) > 0 &&
          Math.fround(significanceLevel) < 1
        )
      )
        throw new Error('Multivariate Local Geary significance must remain in (0, 1) in Float32');
      const columns = operation['columns'];
      // Whole-vector sampling needs distinct variables; the contributor accepts 1 to 16, Kepler 2 to 8.
      if (
        !Array.isArray(columns) ||
        columns.length < 2 ||
        columns.length > 8 ||
        new Set(columns).size !== columns.length
      )
        throw new Error('Multivariate Local Geary requires 2 to 8 distinct columns');
      parsedOperation = {
        kind,
        columns: columns.map(readName),
        weights: parseKeplerPointWeightsSettings(operation['weights']),
        significanceLevel,
        permutationCount: readInteger(operation['permutationCount'], 1, 9999),
        seed: readInteger(operation['seed'], 0, 2 ** 32 - 1),
        correction: readChoice(operation['correction'], ['none', 'bonferroni'] as const),
        tail: readChoice(operation['tail'], ['equal-tail', 'esda-min-tail'] as const)
      };
      break;
    }
    case 'local-moran-polygons': {
      checkKeys(operation, [
        'kind',
        'column',
        'weights',
        'minimumNeighbors',
        'islands',
        'multipleTesting',
        'significanceLevel',
        'inference',
        'permutationCount',
        'seed'
      ]);
      const significanceLevel = operation['significanceLevel'];
      if (
        typeof significanceLevel !== 'number' ||
        !(
          significanceLevel > 0 &&
          significanceLevel < 1 &&
          Math.fround(significanceLevel) > 0 &&
          Math.fround(significanceLevel) < 1
        )
      )
        throw new Error('Local Moran significance must remain in (0, 1) in Float32');
      const inference = readChoice(operation['inference'], [
        'analytic-conditional-normal-two-sided',
        'permutation-conditional-two-sided'
      ] as const);
      const minimumNeighbors = readInteger(operation['minimumNeighbors'], 1, 2 ** 24 - 1);
      const permutation = inference === 'permutation-conditional-two-sided';
      if (permutation && minimumNeighbors !== 1)
        throw new Error('Local Moran permutations require minimumNeighbors = 1');
      if (!permutation && ('permutationCount' in operation || 'seed' in operation))
        throw new Error('Analytic Local Moran does not accept permutation controls');
      parsedOperation = {
        kind,
        column: readName(operation['column']),
        weights: parseKeplerPolygonWeightsSettings(operation['weights']),
        minimumNeighbors,
        islands: readChoice(operation['islands'], ['exclude', 'keep'] as const),
        multipleTesting: readChoice(operation['multipleTesting'], [
          'none',
          'bonferroni',
          'holm',
          'benjamini-hochberg',
          'benjamini-yekutieli'
        ] as const),
        significanceLevel,
        inference,
        ...(permutation
          ? {
              permutationCount: readInteger(
                operation['permutationCount'],
                1,
                MAXIMUM_KEPLER_LOCAL_MORAN_PERMUTATIONS
              ),
              seed: readInteger(operation['seed'], 0, 2 ** 32 - 1)
            }
          : {})
      };
      break;
    }
    case 'hot-spots-points': {
      checkKeys(operation, [
        'kind',
        'column',
        'weights',
        'minimumNeighbors',
        'selfWeight',
        'multipleTesting',
        'inference',
        'islands'
      ]);
      const selfWeight = operation['selfWeight'];
      if (
        typeof selfWeight !== 'number' ||
        !Number.isFinite(Math.fround(selfWeight)) ||
        selfWeight < 0
      )
        throw new Error('Invalid hotspot self weight');
      parsedOperation = {
        kind,
        column: readName(operation['column']),
        weights: parseKeplerPointWeightsSettings(operation['weights']),
        minimumNeighbors: readInteger(operation['minimumNeighbors'], 1, 2 ** 24),
        selfWeight,
        multipleTesting: readChoice(operation['multipleTesting'], [
          'none',
          'bonferroni',
          'holm',
          'benjamini-hochberg',
          'benjamini-yekutieli'
        ] as const),
        inference: readChoice(operation['inference'], ['analytic-two-sided'] as const),
        islands: readChoice(operation['islands'], ['exclude', 'keep'] as const)
      };
      break;
    }
    case 'hot-spots-polygons': {
      checkKeys(operation, [
        'kind',
        'column',
        'weights',
        'minimumNeighbors',
        'selfWeight',
        'multipleTesting',
        'inference',
        'islands'
      ]);
      const selfWeight = operation['selfWeight'];
      if (
        typeof selfWeight !== 'number' ||
        !Number.isFinite(Math.fround(selfWeight)) ||
        selfWeight < 0
      )
        throw new Error('Invalid hotspot self weight');
      parsedOperation = {
        kind,
        column: readName(operation['column']),
        weights: parseKeplerPolygonWeightsSettings(operation['weights']),
        minimumNeighbors: readInteger(operation['minimumNeighbors'], 1, 2 ** 24),
        selfWeight,
        multipleTesting: readChoice(operation['multipleTesting'], [
          'none',
          'bonferroni',
          'holm',
          'benjamini-hochberg',
          'benjamini-yekutieli'
        ] as const),
        inference: readChoice(operation['inference'], ['analytic-two-sided'] as const),
        islands: readChoice(operation['islands'], ['exclude', 'keep'] as const)
      };
      break;
    }
    case 'hot-spots-lattice': {
      checkKeys(operation, [
        'kind',
        'column',
        'width',
        'height',
        'criterion',
        'radius',
        'neighborCapacity',
        'minimumNeighbors',
        'selfWeight',
        'multipleTesting',
        'inference'
      ]);
      const width = readInteger(operation['width'], 1, 2 ** 24);
      const height = readInteger(operation['height'], 1, 2 ** 24);
      if (width * height > 2 ** 24)
        throw new KeplerAllocationError('capacity', {
          message: 'Hotspot lattice exceeds row capacity'
        });
      const selfWeight = operation['selfWeight'];
      if (
        typeof selfWeight !== 'number' ||
        !Number.isFinite(Math.fround(selfWeight)) ||
        selfWeight < 0
      )
        throw new Error('Invalid hotspot self weight');
      parsedOperation = {
        kind,
        column: readName(operation['column']),
        width,
        height,
        criterion: readChoice(operation['criterion'], ['rook', 'queen'] as const),
        radius: readInteger(operation['radius'], 1, 32),
        neighborCapacity: readInteger(operation['neighborCapacity'], 0, 2 ** 31 - 1),
        minimumNeighbors: readInteger(operation['minimumNeighbors'], 0, 2 ** 24),
        selfWeight,
        multipleTesting: readChoice(operation['multipleTesting'], [
          'none',
          'bonferroni',
          'holm',
          'benjamini-hochberg',
          'benjamini-yekutieli'
        ] as const),
        inference: readChoice(operation['inference'], ['analytic-two-sided'] as const)
      };
      break;
    }
    case 'standardize':
      checkKeys(operation, ['kind', 'column', 'method', 'variance']);
      parsedOperation = {
        kind,
        column: readName(operation['column']),
        method: readChoice(operation['method'], [
          'deviation-from-mean',
          'z-score',
          'mad',
          'range-adjust',
          'range-standardize'
        ] as const),
        variance: readChoice(operation['variance'], ['sample', 'population'] as const)
      };
      break;
    case 'excess-risk':
      checkKeys(operation, ['kind', 'events', 'populations']);
      parsedOperation = {
        kind,
        events: readName(operation['events']),
        populations: readName(operation['populations'])
      };
      break;
    case 'spatial-rate-points': {
      checkKeys(operation, ['kind', 'events', 'populations', 'weights', 'scale', 'islandPolicy']);
      const scale = operation['scale'];
      if (typeof scale !== 'number' || !Number.isFinite(Math.fround(scale)) || scale < 0)
        throw new Error('Invalid rate scale');
      parsedOperation = {
        kind,
        events: readName(operation['events']),
        populations: readName(operation['populations']),
        weights: parseKeplerPointWeightsSettings(operation['weights']),
        scale,
        islandPolicy: readChoice(operation['islandPolicy'], ['undefined', 'own-rate'] as const)
      };
      break;
    }
    case 'spatial-rate-polygons': {
      checkKeys(operation, ['kind', 'events', 'populations', 'weights', 'scale', 'islandPolicy']);
      const scale = operation['scale'];
      if (typeof scale !== 'number' || !Number.isFinite(Math.fround(scale)) || scale < 0)
        throw new Error('Invalid rate scale');
      parsedOperation = {
        kind,
        events: readName(operation['events']),
        populations: readName(operation['populations']),
        weights: parseKeplerPolygonWeightsSettings(operation['weights']),
        scale,
        islandPolicy: readChoice(operation['islandPolicy'], ['undefined', 'own-rate'] as const)
      };
      break;
    }
    case 'spatial-rate': {
      checkKeys(operation, [
        'kind',
        'events',
        'populations',
        'weightsId',
        'weightsRevision',
        'scale',
        'islandPolicy'
      ]);
      const scale = operation['scale'];
      if (typeof scale !== 'number' || !Number.isFinite(Math.fround(scale)) || scale < 0)
        throw new Error('Invalid rate scale');
      parsedOperation = {
        kind,
        events: readName(operation['events']),
        populations: readName(operation['populations']),
        weightsId: readName(operation['weightsId']),
        weightsRevision: readRevision(operation['weightsRevision']),
        scale,
        islandPolicy: readChoice(operation['islandPolicy'], ['undefined', 'own-rate'] as const)
      };
      break;
    }
    case 'classify': {
      checkKeys(operation, ['kind', 'column', 'method', 'classCount', 'naturalBreaksMode']);
      const classCount = operation['classCount'];
      if (
        typeof classCount !== 'number' ||
        !Number.isInteger(classCount) ||
        classCount < 1 ||
        classCount > 64
      )
        throw new Error('Classification supports 1 to 64 classes');
      parsedOperation = {
        kind,
        column: readName(operation['column']),
        method: readChoice(operation['method'], [
          'equal-interval',
          'quantile',
          'natural-breaks'
        ] as const),
        classCount,
        naturalBreaksMode: readChoice(operation['naturalBreaksMode'], [
          'approximate',
          'exact'
        ] as const)
      };
      break;
    }
  }
  return Object.freeze({
    schemaVersion: 1,
    algorithmVersion: 1,
    id: readName(recipe['id']),
    source: Object.freeze({
      datasetId: readName(source['datasetId']),
      revision: readRevision(source['revision'])
    }),
    population,
    ...(selectionRevision === undefined ? {} : {selectionRevision}),
    backendPolicy: readChoice(recipe['backendPolicy'], [
      'prefer-gpu',
      'require-gpu',
      'complementary'
    ] as const),
    operation: Object.freeze(parsedOperation),
    ...(recipe['display'] === undefined
      ? {}
      : {display: parseAnalysisDisplay(recipe['display'], parsedOperation.kind)})
  });
}

function parseAnalysisDisplay(value: unknown, kind: KeplerAnalysisOperation['kind']) {
  const display = readObject(value);
  checkKeys(display, ['cartogramDeclaration']);
  if (display['cartogramDeclaration'] === undefined) return Object.freeze({});
  // A declaration georeferences cartogram output only.
  if (kind !== 'dorling-cartogram') throw new Error('Cartogram declaration needs a cartogram');
  const declaration = readObject(display['cartogramDeclaration']);
  checkKeys(declaration, ['metersPerUnit', 'planeOrigin', 'anchor']);
  const readPair = (pair: unknown, limits?: readonly [number, number]): [number, number] => {
    if (!Array.isArray(pair) || pair.length !== 2) throw new Error('Invalid cartogram declaration');
    const numbers = pair.map(item => {
      if (typeof item !== 'number' || !Number.isFinite(item))
        throw new Error('Invalid cartogram declaration');
      return item;
    });
    if (limits && (Math.abs(numbers[0]) > limits[0] || Math.abs(numbers[1]) > limits[1]))
      throw new Error('Invalid cartogram declaration');
    return [numbers[0], numbers[1]];
  };
  const metersPerUnit = declaration['metersPerUnit'];
  if (typeof metersPerUnit !== 'number' || !(metersPerUnit > 0) || !Number.isFinite(metersPerUnit))
    throw new Error('Invalid cartogram declaration');
  return Object.freeze({
    cartogramDeclaration: Object.freeze({
      metersPerUnit,
      planeOrigin: Object.freeze(readPair(declaration['planeOrigin'])),
      // Longitude within +/-180 and latitude within +/-90.
      anchor: Object.freeze(readPair(declaration['anchor'], [180, 90]))
    })
  });
}

export function serializeKeplerAnalysisRecipe(recipe: KeplerAnalysisRecipe): string {
  return JSON.stringify(parseKeplerAnalysisRecipe(recipe));
}

/** A saved mask/weight identifier is a reference; its actual source must still be available. */
export function restoreKeplerAnalysisRecipe(
  serialized: string,
  context: KeplerRecipeRestoreContext
): KeplerAnalysisRecipe {
  const recipe = parseKeplerAnalysisRecipe(JSON.parse(serialized));
  if (context.resolveDatasetRevision(recipe['source'].datasetId) !== recipe['source'].revision)
    throw new Error('Kepler recipe dataset revision is unavailable or stale');
  if (
    recipe['population'] === 'selected' &&
    context.resolveSelectionRevision?.(recipe['source'].datasetId) !== recipe['selectionRevision']
  )
    throw new Error('Kepler recipe selection revision is unavailable or stale');
  if (
    recipe['operation'].kind === 'spatial-rate' &&
    context.resolveWeightsRevision?.(recipe['operation'].weightsId) !==
      recipe['operation'].weightsRevision
  )
    throw new Error('Kepler recipe weights revision is unavailable or stale');
  return recipe;
}

/** Validate and detach point-weight settings before allocating graph resources. */
export function parseKeplerPointWeightsSettings(value: unknown): KeplerPointWeightsSettings {
  const settings = readObject(value);
  const kind = readChoice(settings['kind'], ['knn', 'distance-band'] as const);
  const coordinateSpace = readChoice(settings['coordinateSpace'], [
    'longitude-latitude',
    'planar'
  ] as const);
  const coordinateKeys = getKeplerPointWeightCoordinateKeys(coordinateSpace);
  checkKeys(settings, [
    'kind',
    ...coordinateKeys,
    ...(kind === 'knn' ? ['k'] : ['radiusMeters', 'neighborCapacity', 'permutationDegree'])
  ]);
  const coordinates = parseKeplerPointWeightCoordinates(
    Object.fromEntries(coordinateKeys.map(key => [key, settings[key]]))
  );
  if (kind === 'knn')
    return Object.freeze({kind, ...coordinates, k: readInteger(settings['k'], 1, 32)});
  return Object.freeze({
    kind,
    ...coordinates,
    radiusMeters: parseKeplerPointWeightRadius(settings['radiusMeters'], coordinates),
    neighborCapacity: readInteger(
      settings['neighborCapacity'],
      1,
      MAXIMUM_KEPLER_DISTANCE_BAND_CAPACITY
    ),
    ...(settings['permutationDegree'] === undefined
      ? {}
      : {
          permutationDegree: readInteger(
            settings['permutationDegree'],
            DEFAULT_KEPLER_PERMUTATION_DEGREE,
            MAXIMUM_KEPLER_PERMUTATION_DEGREE
          )
        })
  });
}

/** Exact source-XY equality topology; capacities include intermediate duplicate feature pairs. */
export function parseKeplerPolygonWeightsSettings(value: unknown): KeplerPolygonWeightsSettings {
  const settings = readObject(value);
  checkKeys(settings, [
    'kind',
    'geometryColumn',
    'criterion',
    'equality',
    'neighborCapacity',
    'pairCapacity'
  ]);
  return Object.freeze({
    kind: readChoice(settings['kind'], ['contiguity'] as const),
    geometryColumn: readName(settings['geometryColumn']),
    criterion: readChoice(settings['criterion'], ['queen', 'rook'] as const),
    equality: readChoice(settings['equality'], ['exact-source-xy'] as const),
    neighborCapacity: readInteger(
      settings['neighborCapacity'],
      1,
      MAXIMUM_KEPLER_POLYGON_NEIGHBOR_CAPACITY
    ),
    pairCapacity: readInteger(settings['pairCapacity'], 1, MAXIMUM_KEPLER_POLYGON_PAIR_CAPACITY)
  });
}

/** Validate geographic support and detach settings before creating GPU resources. */
export function parseKeplerIDWSettings(value: unknown): KeplerIDWSettings {
  const settings = readObject(value);
  checkKeys(settings, [
    'longitudeColumn',
    'latitudeColumn',
    'coordinateSpace',
    'metric',
    'units',
    'grid',
    'searchRadius',
    'power',
    'neighborCount',
    'minimumNeighborCount',
    'taper'
  ]);
  const {grid, searchRadius} = parseGeographicInterpolationGrid(
    settings['grid'],
    settings['searchRadius'],
    MAXIMUM_KEPLER_IDW_GRID_DIMENSION,
    MAXIMUM_KEPLER_IDW_CELLS
  );
  const power = settings['power'];
  if (typeof power !== 'number' || !Number.isFinite(power) || power < 0 || power > 32)
    throw new Error('Invalid IDW power');
  const neighborCount = readInteger(settings['neighborCount'], 1, 64);
  return Object.freeze({
    longitudeColumn: readName(settings['longitudeColumn']),
    latitudeColumn: readName(settings['latitudeColumn']),
    coordinateSpace: readChoice(settings['coordinateSpace'], ['longitude-latitude'] as const),
    metric: readChoice(settings['metric'], ['great-circle'] as const),
    units: readChoice(settings['units'], ['meters'] as const),
    grid,
    searchRadius,
    power: Math.fround(power),
    neighborCount,
    minimumNeighborCount: readInteger(settings['minimumNeighborCount'], 1, neighborCount),
    taper: readChoice(settings['taper'], ['none', 'modified-shepard'] as const)
  });
}

/** Ordinary geographic kriging with explicit manual variogram and no silent capacity clamping. */
export function parseKeplerKrigingSettings(value: unknown): KeplerKrigingSettings {
  const settings = readObject(value);
  checkKeys(settings, [
    'longitudeColumn',
    'latitudeColumn',
    'coordinateSpace',
    'metric',
    'units',
    'grid',
    'searchRadiusMeters',
    'neighborCount',
    'minimumNeighbors',
    'variogram',
    'confidenceLevel',
    'crossValidation',
    'variogramFit'
  ]);
  const {grid, searchRadius} = parseGeographicInterpolationGrid(
    settings['grid'],
    settings['searchRadiusMeters'],
    MAXIMUM_KEPLER_KRIGING_GRID_DIMENSION,
    MAXIMUM_KEPLER_KRIGING_CELLS
  );
  const neighborCount = readInteger(settings['neighborCount'], 1, MAXIMUM_KEPLER_KRIGING_NEIGHBORS);
  const variogram = readObject(settings['variogram']);
  checkKeys(variogram, ['model', 'nugget', 'sill', 'rangeMeters']);
  const nugget = readNonnegativeFloat32(variogram['nugget']);
  const sill = readNonnegativeFloat32(variogram['sill']);
  const rangeMeters = readNonnegativeFloat32(variogram['rangeMeters']);
  const total = Math.fround(nugget + sill);
  if (!(total > 0) || !Number.isFinite(total) || !(rangeMeters > 0))
    throw new Error('Kriging requires a finite positive Float32 total sill and range');
  return Object.freeze({
    longitudeColumn: readName(settings['longitudeColumn']),
    latitudeColumn: readName(settings['latitudeColumn']),
    coordinateSpace: readChoice(settings['coordinateSpace'], ['longitude-latitude'] as const),
    metric: readChoice(settings['metric'], ['great-circle'] as const),
    units: readChoice(settings['units'], ['meters'] as const),
    grid,
    searchRadiusMeters: searchRadius,
    neighborCount,
    minimumNeighbors: readInteger(settings['minimumNeighbors'], 1, neighborCount),
    variogram: Object.freeze({
      model: readChoice(variogram['model'], ['spherical', 'exponential', 'gaussian'] as const),
      nugget,
      sill,
      rangeMeters
    }),
    ...parseKrigingDiagnosticSettings(settings)
  });
}

function parseKrigingDiagnosticSettings(settings: Record<string, unknown>) {
  const result: {
    confidenceLevel?: number;
    crossValidation?: boolean;
    variogramFit?: NonNullable<KeplerKrigingSettings['variogramFit']>;
  } = {};
  const confidenceLevel = settings['confidenceLevel'];
  if (confidenceLevel !== undefined) {
    if (typeof confidenceLevel !== 'number' || !(confidenceLevel > 0 && confidenceLevel < 1))
      throw new Error('Kriging confidence level must be between 0 and 1');
    result.confidenceLevel = confidenceLevel;
  }
  const crossValidation = settings['crossValidation'];
  if (crossValidation !== undefined) {
    if (typeof crossValidation !== 'boolean') throw new Error('Invalid kriging cross-validation');
    result.crossValidation = crossValidation;
  }
  const fit = settings['variogramFit'];
  if (fit !== undefined) {
    const fitObject = readObject(fit);
    checkKeys(fitObject, [
      'method',
      'weighting',
      'binCount',
      'pairCount',
      'maximumLagMeters',
      'criterion'
    ]);
    const maximumLagMeters = fitObject['maximumLagMeters'];
    const criterion = fitObject['criterion'];
    if (
      typeof maximumLagMeters !== 'number' ||
      !(maximumLagMeters > 0) ||
      !Number.isFinite(maximumLagMeters) ||
      typeof criterion !== 'number' ||
      !Number.isFinite(criterion)
    )
      throw new Error('Invalid kriging variogram fit provenance');
    result.variogramFit = Object.freeze({
      method: readChoice(fitObject['method'], ['weighted-least-squares'] as const),
      weighting: readChoice(fitObject['weighting'], ['cressie'] as const),
      binCount: readInteger(fitObject['binCount'], 2, 512),
      pairCount: readInteger(fitObject['pairCount'], 0, Number.MAX_SAFE_INTEGER),
      maximumLagMeters,
      criterion
    });
  }
  return result;
}

function parseGeographicInterpolationGrid(
  gridValue: unknown,
  radiusValue: unknown,
  maximumDimension: number,
  maximumCells: number
) {
  const grid = readObject(gridValue);
  checkKeys(grid, ['bounds', 'width', 'height']);
  const width = readInteger(grid['width'], 1, maximumDimension);
  const height = readInteger(grid['height'], 1, maximumDimension);
  if (width * height > maximumCells)
    throw new KeplerAllocationError('capacity', {
      message: 'Interpolation grid exceeds cell capacity'
    });
  const bounds = grid['bounds'];
  if (
    !Array.isArray(bounds) ||
    bounds.length !== 4 ||
    !bounds.every(value => typeof value === 'number' && Number.isFinite(value))
  )
    throw new Error('Interpolation requires finite geographic bounds');
  const [west, south, east, north] = bounds.map(Math.fround);
  if (!(west >= -180 && west < east && east <= 180 && south > -90 && south < north && north < 90))
    throw new Error('Interpolation requires ordered nonwrapping geographic bounds');
  if (
    typeof radiusValue !== 'number' ||
    !Number.isFinite(Math.fround(radiusValue)) ||
    Math.fround(radiusValue) <= 0
  )
    throw new Error('Interpolation radius must be positive finite Float32 meters');
  const searchRadius = Math.fround(radiusValue);
  // Match both contributors' conservative query box, including Float32 coordinate rounding.
  const latitudeSpan = toDegrees(searchRadius / EARTH_RADIUS_METERS) * 1.01;
  const highestLatitude = Math.max(Math.abs(south), Math.abs(north)) + latitudeSpan;
  const longitudeSpan = latitudeSpan / Math.cos(toRadians(highestLatitude));
  if (highestLatitude >= 89.9 || west - longitudeSpan <= -180 || east + longitudeSpan >= 180)
    throw new Error('Interpolation search domain must not reach poles or cross the antimeridian');
  return {
    grid: Object.freeze({
      width,
      height,
      bounds: Object.freeze([west, south, east, north] as const)
    }),
    searchRadius
  };
}

/** Detach exact geographic clustering controls; no implicit projection or value column. */
export function parseKeplerDBSCANSettings(value: unknown): KeplerDBSCANSettings {
  const settings = readObject(value);
  checkKeys(settings, [
    'longitudeColumn',
    'latitudeColumn',
    'coordinateSpace',
    'metric',
    'units',
    'epsilon',
    'minimumSamples'
  ]);
  const epsilon = settings['epsilon'];
  if (
    typeof epsilon !== 'number' ||
    !Number.isFinite(Math.fround(epsilon)) ||
    Math.fround(epsilon) <= 0
  )
    throw new Error('DBSCAN epsilon must be positive finite Float32 meters');
  return Object.freeze({
    longitudeColumn: readName(settings['longitudeColumn']),
    latitudeColumn: readName(settings['latitudeColumn']),
    coordinateSpace: readChoice(settings['coordinateSpace'], ['longitude-latitude'] as const),
    metric: readChoice(settings['metric'], ['great-circle'] as const),
    units: readChoice(settings['units'], ['meters'] as const),
    epsilon: Math.fround(epsilon),
    minimumSamples: readInteger(
      settings['minimumSamples'],
      1,
      MAXIMUM_KEPLER_DBSCAN_MINIMUM_SAMPLES
    )
  });
}

/** Validate a complete-case model with an explicit ordered design and fixed intercept policy. */
export function parseKeplerOLSSettings(value: unknown): KeplerOLSSettings {
  const settings = readObject(value);
  checkKeys(settings, [
    'responseColumn',
    'predictorColumns',
    'intercept',
    'ridgeLambda',
    'variance',
    'missing'
  ]);
  const responseColumn = readName(settings['responseColumn']);
  const predictors = settings['predictorColumns'];
  if (
    !Array.isArray(predictors) ||
    predictors.length < 1 ||
    predictors.length > MAXIMUM_KEPLER_OLS_PREDICTORS
  )
    throw new Error('OLS requires 1 to 15 predictor columns');
  const predictorColumns = predictors.map(readName);
  if (
    new Set(predictorColumns).size !== predictorColumns.length ||
    predictorColumns.includes(responseColumn)
  )
    throw new Error('OLS predictors must be unique and distinct from the response');
  const ridgeLambda = settings['ridgeLambda'];
  if (
    typeof ridgeLambda !== 'number' ||
    ridgeLambda < 0 ||
    !Number.isFinite(Math.fround(ridgeLambda)) ||
    (ridgeLambda > 0 && Math.fround(ridgeLambda) === 0)
  )
    throw new Error('OLS ridge penalty must be nonnegative finite Float32');
  return Object.freeze({
    responseColumn,
    predictorColumns: Object.freeze(predictorColumns),
    intercept: readChoice(settings['intercept'], ['included'] as const),
    ridgeLambda: Math.fround(ridgeLambda),
    variance: readChoice(settings['variance'], ['degrees-of-freedom'] as const),
    missing: readChoice(settings['missing'], ['complete-cases'] as const)
  });
}

function readNonnegativeFloat32(value: unknown): number {
  if (
    typeof value !== 'number' ||
    value < 0 ||
    !Number.isFinite(Math.fround(value)) ||
    (value > 0 && Math.fround(value) === 0)
  )
    throw new Error('Expected a nonnegative finite Float32 parameter');
  return Math.fround(value);
}

function readObject(value: unknown): Record<string, unknown> {
  if (
    !value ||
    typeof value !== 'object' ||
    Array.isArray(value) ||
    Object.getPrototypeOf(value) !== Object.prototype
  )
    throw new Error('Expected a plain recipe object');
  return value as Record<string, unknown>;
}
function readName(value: unknown): string {
  if (typeof value !== 'string' || !value.length || value.length > 1024)
    throw new Error('Invalid recipe identifier');
  return value;
}
function readRevision(value: unknown): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1)
    throw new Error('Invalid recipe revision');
  return value;
}
function readInteger(value: unknown, minimum: number, maximum: number): number {
  if (
    typeof value !== 'number' ||
    !Number.isSafeInteger(value) ||
    value < minimum ||
    value > maximum
  )
    throw new Error('Invalid recipe integer');
  return value;
}
function readChoice<const Choice extends string>(
  value: unknown,
  choices: readonly Choice[]
): Choice {
  if (typeof value !== 'string' || !choices.includes(value as Choice))
    throw new Error('Unsupported recipe option');
  return value as Choice;
}
function checkKeys(object: Record<string, unknown>, keys: readonly string[]): void {
  if (Object.keys(object).some(key => !keys.includes(key)))
    throw new Error('Unknown recipe property');
}
