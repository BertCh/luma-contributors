// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  parseKeplerIDWSettings,
  parseKeplerKrigingSettings,
  MAXIMUM_KEPLER_KRIGING_SOURCE_ROWS,
  parseKeplerAnalysisRecipe,
  parseKeplerDorlingSettings,
  MAXIMUM_KEPLER_POLYGON_ROWS,
  type KeplerPolygonAnalysisOperation,
  parseKeplerDBSCANSettings,
  parseKeplerOLSSettings,
  MAXIMUM_KEPLER_OLS_SOURCE_ROWS,
  MAXIMUM_KEPLER_DBSCAN_SOURCE_ROWS,
  MAXIMUM_KEPLER_IDW_SOURCE_ROWS,
  type KeplerAnalysisRecipe
} from '@luma-contributors/kepler';
import type {AnalysisSettings} from './analysis-runtime';
import {normalizeHotspotSettings} from './analysis-hotspot-settings';
import {
  getPointTopologySettings,
  normalizePointWeightsSettings
} from './analysis-point-weights-settings';
import {normalizePointHotspotSettings} from './analysis-point-hotspot-settings';
import {normalizeLocalMoranSettings} from './analysis-local-moran-settings';
import {normalizeQuantileLisaSettings} from './analysis-quantile-lisa-settings';
import {getPointCoordinateColumns} from './analysis-point-coordinate-settings';
import {normalizeCartogramSettings} from './cartogram-settings';

export const MAXIMUM_EXACT_NATURAL_BREAKS_ROWS = 1024;

export function isPolygonAnalysisOperation(
  operation: KeplerAnalysisRecipe['operation']
): operation is KeplerPolygonAnalysisOperation {
  return (
    operation.kind === 'spatial-rate-polygons' ||
    operation.kind === 'hot-spots-polygons' ||
    operation.kind === 'local-moran-polygons' ||
    operation.kind === 'quantile-lisa-polygons'
  );
}

/** Detach and validate every nested field before host state or GPU ownership changes. */
export function normalizePolygonAnalysisOperation(
  operation: KeplerPolygonAnalysisOperation
): KeplerPolygonAnalysisOperation {
  const parsed = parseKeplerAnalysisRecipe({
    schemaVersion: 1,
    algorithmVersion: 1,
    id: 'polygon-settings',
    source: {datasetId: 'polygon-settings', revision: 1},
    population: 'all',
    backendPolicy: 'require-gpu',
    operation
  }).operation;
  if (!isPolygonAnalysisOperation(parsed)) throw new Error('Expected polygon analysis settings');
  return parsed;
}

/** Validate source capacity before allocating work or mutating restored host state. */
export function validateAnalysisRecipeCapacity(
  recipe: KeplerAnalysisRecipe,
  rowCount: number
): void {
  if (isPolygonAnalysisOperation(recipe.operation)) {
    normalizePolygonAnalysisOperation(recipe.operation);
    if (!Number.isSafeInteger(rowCount) || rowCount < 1 || rowCount > MAXIMUM_KEPLER_POLYGON_ROWS)
      throw new Error(`Polygon analysis supports 1 to ${MAXIMUM_KEPLER_POLYGON_ROWS} source rows`);
    if (recipe.operation.kind === 'quantile-lisa-polygons') {
      if (rowCount < 3) throw new Error('Quantile LISA requires at least 3 source rows');
      if (recipe.operation.quantileCount >= rowCount)
        throw new Error('Quantile count must be smaller than the source row count');
    }
  }
  if (recipe.operation.kind === 'dorling-cartogram') {
    const {kind, ...settings} = recipe.operation;
    normalizeCartogramSettings(settings, rowCount);
  }
  if (recipe.operation.kind === 'ols') {
    const {kind, ...settings} = recipe.operation;
    parseKeplerOLSSettings(settings);
    if (rowCount < 1 || rowCount > MAXIMUM_KEPLER_OLS_SOURCE_ROWS)
      throw new Error(`OLS supports 1 to ${MAXIMUM_KEPLER_OLS_SOURCE_ROWS} total source rows`);
  }
  if (recipe.operation.kind === 'dbscan-points') {
    const {kind, ...settings} = recipe.operation;
    parseKeplerDBSCANSettings(settings);
    if (rowCount < 1 || rowCount > MAXIMUM_KEPLER_DBSCAN_SOURCE_ROWS)
      throw new Error(
        `DBSCAN supports 1 to ${MAXIMUM_KEPLER_DBSCAN_SOURCE_ROWS} total source rows`
      );
  }
  if (recipe.operation.kind === 'kriging-points') {
    const {kind, column, ...settings} = recipe.operation;
    parseKeplerKrigingSettings(settings);
    if (rowCount < 1 || rowCount > MAXIMUM_KEPLER_KRIGING_SOURCE_ROWS)
      throw new Error(
        `Kriging supports 1 to ${MAXIMUM_KEPLER_KRIGING_SOURCE_ROWS} total source rows`
      );
  }
  if (recipe.operation.kind === 'idw-points') {
    const {kind, column, ...settings} = recipe.operation;
    parseKeplerIDWSettings(settings);
    if (rowCount < 1 || rowCount > MAXIMUM_KEPLER_IDW_SOURCE_ROWS)
      throw new Error(
        `Interpolation supports 1 to ${MAXIMUM_KEPLER_IDW_SOURCE_ROWS} total source rows`
      );
  }
  if (recipe.operation.kind === 'quantile-lisa-points')
    normalizeQuantileLisaSettings(
      {
        ...getPointTopologySettings(recipe.operation.weights),
        quantileCount: recipe.operation.quantileCount,
        selectedQuantile: recipe.operation.selectedQuantile,
        seed: recipe.operation.seed,
        permutationCount: recipe.operation.permutationCount,
        significanceLevel: recipe.operation.significanceLevel
      },
      rowCount
    );
  if (recipe.operation.kind === 'multivariate-local-geary-points')
    normalizeLocalMoranSettings(
      {
        ...getPointTopologySettings(recipe.operation.weights),
        variableColumns: [...recipe.operation.columns],
        minimumNeighbors: 1,
        multipleTesting: recipe.operation.correction,
        islands: 'exclude',
        significanceLevel: recipe.operation.significanceLevel,
        inference: 'permutation-conditional-two-sided',
        permutationCount: recipe.operation.permutationCount,
        seed: recipe.operation.seed
      },
      rowCount
    );
  if (recipe.operation.kind === 'local-moran-points')
    normalizeLocalMoranSettings(
      {
        ...getPointTopologySettings(recipe.operation.weights),
        ...(recipe.operation.secondColumn === undefined
          ? {}
          : {secondColumn: recipe.operation.secondColumn}),
        minimumNeighbors: recipe.operation.minimumNeighbors,
        multipleTesting: recipe.operation.multipleTesting,
        islands: recipe.operation.islands,
        significanceLevel: recipe.operation.significanceLevel,
        inference: recipe.operation.inference,
        ...(recipe.operation.inference === 'permutation-conditional-two-sided'
          ? {permutationCount: recipe.operation.permutationCount, seed: recipe.operation.seed}
          : {})
      },
      rowCount
    );
  if (recipe.operation.kind === 'hot-spots-lattice')
    normalizeHotspotSettings(recipe.operation, rowCount);
  if (recipe.operation.kind === 'hot-spots-points')
    normalizePointHotspotSettings(
      {
        ...getPointTopologySettings(recipe.operation.weights),
        minimumNeighbors: recipe.operation.minimumNeighbors,
        selfWeight: recipe.operation.selfWeight,
        multipleTesting: recipe.operation.multipleTesting,
        islands: recipe.operation.islands
      },
      rowCount
    );
  if (recipe.operation.kind === 'spatial-rate-points')
    normalizePointWeightsSettings(
      {
        ...getPointTopologySettings(recipe.operation.weights),
        scale: recipe.operation.scale,
        islandPolicy: recipe.operation.islandPolicy
      },
      rowCount
    );
  if (
    recipe.operation.kind === 'classify' &&
    recipe.operation.method === 'natural-breaks' &&
    recipe.operation.naturalBreaksMode === 'exact' &&
    rowCount > MAXIMUM_EXACT_NATURAL_BREAKS_ROWS
  )
    throw new Error(
      `Exact natural breaks supports at most ${MAXIMUM_EXACT_NATURAL_BREAKS_ROWS} source rows`
    );
}

/** Shared by replay preflight and execution so unsupported recipes cannot mutate native state. */
export function getAnalysisRecipeSettings(
  recipe: KeplerAnalysisRecipe
): Pick<AnalysisSettings, 'mode' | 'population'> &
  Partial<
    Pick<
      AnalysisSettings,
      | 'column'
      | 'classCount'
      | 'variance'
      | 'naturalBreaksMode'
      | 'denominatorColumn'
      | 'hotspot'
      | 'pointWeights'
      | 'pointHotspot'
      | 'localMoran'
      | 'quantileLisa'
      | 'idw'
      | 'kriging'
      | 'dbscan'
      | 'ols'
      | 'polygon'
      | 'cartogram'
    >
  > {
  const operation = recipe.operation;
  if (operation.kind === 'dorling-cartogram') {
    const {kind, ...settings} = operation;
    const cartogram = parseKeplerDorlingSettings(settings);
    return {mode: kind, population: recipe.population, column: cartogram.valueColumn, cartogram};
  }
  if (isPolygonAnalysisOperation(operation)) {
    const polygon = normalizePolygonAnalysisOperation(operation);
    return {
      mode: polygon.kind,
      population: recipe.population,
      column: polygon.kind === 'spatial-rate-polygons' ? polygon.events : polygon.column,
      ...(polygon.kind === 'spatial-rate-polygons' ? {denominatorColumn: polygon.populations} : {}),
      polygon
    };
  }
  if (operation.kind === 'ols') {
    const {kind, ...settings} = operation;
    return {
      mode: 'ols',
      population: recipe.population,
      column: operation.responseColumn,
      ols: parseKeplerOLSSettings(settings)
    };
  }
  if (operation.kind === 'dbscan-points') {
    const {kind, ...settings} = operation;
    return {
      mode: 'dbscan-points',
      population: recipe.population,
      dbscan: parseKeplerDBSCANSettings(settings)
    };
  }
  if (operation.kind === 'kriging-points') {
    const {kind, column, ...settings} = operation;
    return {
      mode: kind,
      population: recipe.population,
      column,
      kriging: parseKeplerKrigingSettings(settings)
    };
  }
  if (operation.kind === 'idw-points') {
    const {kind, column, ...settings} = operation;
    return {
      mode: 'idw-points',
      population: recipe.population,
      column,
      idw: parseKeplerIDWSettings(settings)
    };
  }
  if (operation.kind === 'quantile-lisa-points')
    return {
      mode: 'quantile-lisa-points',
      population: recipe.population,
      column: operation.column,
      quantileLisa: {
        ...getPointTopologySettings(operation.weights),
        quantileCount: operation.quantileCount,
        selectedQuantile: operation.selectedQuantile,
        seed: operation.seed,
        permutationCount: operation.permutationCount,
        significanceLevel: operation.significanceLevel
      }
    };
  if (operation.kind === 'multivariate-local-geary-points')
    return {
      mode: 'local-moran-points',
      population: recipe.population,
      column: operation.columns[0],
      localMoran: {
        ...getPointTopologySettings(operation.weights),
        variableColumns: [...operation.columns],
        minimumNeighbors: 1,
        multipleTesting: operation.correction,
        islands: 'exclude',
        significanceLevel: operation.significanceLevel,
        inference: 'permutation-conditional-two-sided',
        permutationCount: operation.permutationCount,
        seed: operation.seed
      }
    };
  if (operation.kind === 'local-moran-points')
    return {
      mode: 'local-moran-points',
      population: recipe.population,
      column: operation.column,
      localMoran: {
        ...getPointTopologySettings(operation.weights),
        ...(operation.secondColumn === undefined ? {} : {secondColumn: operation.secondColumn}),
        minimumNeighbors: operation.minimumNeighbors,
        multipleTesting: operation.multipleTesting,
        islands: operation.islands,
        significanceLevel: operation.significanceLevel,
        inference: operation.inference,
        ...(operation.inference === 'permutation-conditional-two-sided'
          ? {permutationCount: operation.permutationCount, seed: operation.seed}
          : {})
      }
    };
  if (operation.kind === 'hot-spots-points')
    return {
      mode: 'hot-spots-points',
      population: recipe.population,
      column: operation.column,
      pointHotspot: {
        ...getPointTopologySettings(operation.weights),
        minimumNeighbors: operation.minimumNeighbors,
        selfWeight: operation.selfWeight,
        multipleTesting: operation.multipleTesting,
        islands: operation.islands
      }
    };
  if (operation.kind === 'spatial-rate-points')
    return {
      mode: 'spatial-rate-points',
      population: recipe.population,
      column: operation.events,
      denominatorColumn: operation.populations,
      pointWeights: {
        ...getPointTopologySettings(operation.weights),
        scale: operation.scale,
        islandPolicy: operation.islandPolicy
      }
    };
  if (operation.kind === 'hot-spots-lattice')
    return {
      mode: 'hot-spots-lattice',
      population: recipe.population,
      column: operation.column,
      hotspot: {
        width: operation.width,
        height: operation.height,
        criterion: operation.criterion,
        radius: operation.radius,
        neighborCapacity: operation.neighborCapacity,
        minimumNeighbors: operation.minimumNeighbors,
        selfWeight: operation.selfWeight,
        multipleTesting: operation.multipleTesting
      }
    };
  if (operation.kind === 'excess-risk')
    return {
      mode: 'excess-risk',
      population: recipe.population,
      column: operation.events,
      denominatorColumn: operation.populations
    };
  if (operation.kind === 'standardize')
    return {
      mode: operation.method,
      population: recipe.population,
      column: operation.column,
      variance: operation.variance
    };
  if (
    operation.kind === 'classify' &&
    (operation.method === 'quantile' ||
      operation.method === 'equal-interval' ||
      operation.method === 'natural-breaks') &&
    operation.classCount >= 2 &&
    operation.classCount <= 8
  )
    return {
      mode: operation.method,
      column: operation.column,
      classCount: operation.classCount,
      naturalBreaksMode: operation.naturalBreaksMode,
      population: recipe.population
    };
  throw new Error('Recipe is not supported by this example panel');
}

/** All source fields required by a supported host recipe, before any restoration mutates state. */
export function getAnalysisRecipeColumns(recipe: KeplerAnalysisRecipe): readonly string[] {
  const settings = getAnalysisRecipeSettings(recipe);
  return [
    ...new Set([
      ...(settings.column ? [settings.column] : []),
      ...(settings.cartogram ? [settings.cartogram.xColumn, settings.cartogram.yColumn] : []),
      ...(settings.polygon ? [settings.polygon.weights.geometryColumn] : []),
      ...(settings.denominatorColumn ? [settings.denominatorColumn] : []),
      ...(settings.pointWeights ? getPointCoordinateColumns(settings.pointWeights) : []),
      ...(settings.ols ? [settings.ols.responseColumn, ...settings.ols.predictorColumns] : []),
      ...(settings.dbscan ? [settings.dbscan.longitudeColumn, settings.dbscan.latitudeColumn] : []),
      ...(settings.idw ? [settings.idw.longitudeColumn, settings.idw.latitudeColumn] : []),
      ...(settings.kriging
        ? [settings.kriging.longitudeColumn, settings.kriging.latitudeColumn]
        : []),
      ...(settings.quantileLisa ? getPointCoordinateColumns(settings.quantileLisa) : []),
      ...(settings.localMoran ? getPointCoordinateColumns(settings.localMoran) : []),
      ...(settings.localMoran?.secondColumn ? [settings.localMoran.secondColumn] : []),
      ...(settings.localMoran?.variableColumns ?? []),
      ...(settings.pointHotspot ? getPointCoordinateColumns(settings.pointHotspot) : [])
    ])
  ];
}
