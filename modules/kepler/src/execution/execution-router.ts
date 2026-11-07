// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {MAXIMUM_KEPLER_DORLING_SOURCE_ROWS} from '../analysis/dorling-cartogram';
import {MAXIMUM_KEPLER_MULTIVARIATE_GEARY_ROWS} from '../analysis/point-multivariate-local-geary';
import {getKeplerPointWeightCoordinateColumns} from '../analysis/point-weight-coordinates';
import {
  parseKeplerAnalysisRecipe,
  MAXIMUM_KEPLER_DISTANCE_BAND_ROWS,
  MAXIMUM_KEPLER_POLYGON_ROWS,
  MAXIMUM_KEPLER_IDW_SOURCE_ROWS,
  MAXIMUM_KEPLER_KRIGING_SOURCE_ROWS,
  MAXIMUM_KEPLER_DBSCAN_SOURCE_ROWS,
  MAXIMUM_KEPLER_OLS_SOURCE_ROWS,
  MAXIMUM_KEPLER_LOCAL_MORAN_PERMUTATION_ROWS,
  type KeplerAnalysisRecipe,
  type KeplerBackendPolicy
} from '../recipes/analysis-recipe';

export type KeplerExecutionRequest =
  | {kind: 'sql'; sql: string; backendPolicy?: KeplerBackendPolicy}
  | {
      kind: 'analysis';
      recipe: KeplerAnalysisRecipe;
      /** Actual prepared input metadata, after explicit type normalization. */
      input: {rowCount: number; chunkCount: number; formats: Readonly<Record<string, string>>};
      /** Estimate includes outputs/scratch; missing estimates cannot establish memory admission. */
      estimatedBytes: number;
    };
export type KeplerExecutionCapabilities = {
  webgpu: boolean;
  /** Available budget after source/resident allocations and outstanding reservations. */
  availableBytes: number;
  duckdb: boolean;
  analysisBackend: boolean;
};
export type KeplerExecutionRoute = {
  backend: 'luma' | 'duckdb' | 'analysis-backend' | 'unavailable';
  reason:
    | 'supported'
    | 'general-sql'
    | 'explicit-complementary'
    | 'webgpu-unavailable'
    | 'precision'
    | 'capacity'
    | 'packing-required'
    | 'memory-budget'
    | 'backend-unavailable';
};

/** Capability routing is CPU-only and never silently changes precision, population or batch layout. */
export function selectKeplerExecutionRoute(
  request: KeplerExecutionRequest,
  capabilities: KeplerExecutionCapabilities
): KeplerExecutionRoute {
  if (!Number.isSafeInteger(capabilities.availableBytes) || capabilities.availableBytes < 0)
    throw new Error('Invalid available memory budget');
  if (request.kind === 'sql') {
    if (!request.sql.trim()) throw new Error('SQL must not be empty');
    if (request.backendPolicy === 'require-gpu')
      return {backend: 'unavailable', reason: 'general-sql'};
    return {
      backend: capabilities.duckdb ? 'duckdb' : 'unavailable',
      reason: capabilities.duckdb ? 'general-sql' : 'backend-unavailable'
    };
  }
  const recipe = parseKeplerAnalysisRecipe(request.recipe);
  const {rowCount, chunkCount, formats} = request.input;
  if (
    !Number.isSafeInteger(rowCount) ||
    rowCount < 0 ||
    !Number.isSafeInteger(chunkCount) ||
    chunkCount < 0 ||
    (rowCount > 0 && chunkCount === 0)
  )
    throw new Error('Invalid analysis input shape');
  if (!Number.isSafeInteger(request.estimatedBytes) || request.estimatedBytes < 0)
    throw new Error('Invalid analysis memory estimate');
  const operation = recipe.operation;
  const fields =
    operation.kind === 'dorling-cartogram'
      ? [operation.xColumn, operation.yColumn, operation.valueColumn]
      : operation.kind === 'ols'
        ? [operation.responseColumn, ...operation.predictorColumns]
        : operation.kind === 'dbscan-points'
          ? [operation.longitudeColumn, operation.latitudeColumn]
          : operation.kind === 'multivariate-local-geary-points'
            ? [...operation.columns]
            : 'column' in operation
              ? operation.kind === 'local-moran-points' && operation.secondColumn
                ? [operation.column, operation.secondColumn]
                : [operation.column]
              : [operation.events, operation.populations];
  if ('weights' in operation && operation.weights.kind !== 'contiguity')
    fields.push(...getKeplerPointWeightCoordinateColumns(operation.weights));
  if (operation.kind === 'idw-points' || operation.kind === 'kriging-points')
    fields.push(operation.longitudeColumn, operation.latitudeColumn);
  let reason: KeplerExecutionRoute['reason'] = 'supported';
  if (recipe.backendPolicy === 'complementary') reason = 'explicit-complementary';
  else if (!capabilities.webgpu) reason = 'webgpu-unavailable';
  else if (fields.some(field => formats[field] !== 'float32')) reason = 'precision';
  else if (
    rowCount > 2 ** 24 ||
    ('weights' in operation &&
      operation.weights.kind === 'contiguity' &&
      (rowCount < 1 || rowCount > MAXIMUM_KEPLER_POLYGON_ROWS)) ||
    ('weights' in operation &&
      operation.weights.kind === 'distance-band' &&
      rowCount > MAXIMUM_KEPLER_DISTANCE_BAND_ROWS) ||
    (operation.kind === 'local-moran-points' &&
      operation.inference === 'permutation-conditional-two-sided' &&
      rowCount > MAXIMUM_KEPLER_LOCAL_MORAN_PERMUTATION_ROWS) ||
    (operation.kind === 'multivariate-local-geary-points' &&
      (rowCount < 2 || rowCount > MAXIMUM_KEPLER_MULTIVARIATE_GEARY_ROWS)) ||
    (operation.kind === 'dorling-cartogram' &&
      (rowCount < 1 || rowCount > MAXIMUM_KEPLER_DORLING_SOURCE_ROWS)) ||
    (operation.kind === 'ols' && (rowCount < 1 || rowCount > MAXIMUM_KEPLER_OLS_SOURCE_ROWS)) ||
    (operation.kind === 'dbscan-points' &&
      (rowCount < 1 || rowCount > MAXIMUM_KEPLER_DBSCAN_SOURCE_ROWS)) ||
    (operation.kind === 'kriging-points' &&
      (rowCount < 1 || rowCount > MAXIMUM_KEPLER_KRIGING_SOURCE_ROWS)) ||
    (operation.kind === 'idw-points' &&
      (rowCount < 1 || rowCount > MAXIMUM_KEPLER_IDW_SOURCE_ROWS)) ||
    ((operation.kind === 'hot-spots-points' ||
      operation.kind === 'local-moran-points' ||
      operation.kind === 'quantile-lisa-points') &&
      rowCount >= 2 ** 24) ||
    ((operation.kind === 'spatial-rate-points' ||
      operation.kind === 'hot-spots-points' ||
      operation.kind === 'local-moran-points') &&
      rowCount === 0) ||
    ((operation.kind === 'quantile-lisa-points' || operation.kind === 'quantile-lisa-polygons') &&
      (rowCount < 3 || operation.quantileCount >= rowCount)) ||
    (operation.kind === 'classify' &&
      operation.method === 'natural-breaks' &&
      operation.naturalBreaksMode === 'exact' &&
      rowCount > 1024)
  )
    reason = 'capacity';
  else if (operation.kind !== 'standardize' && chunkCount !== 1) reason = 'packing-required';
  else if (request.estimatedBytes > capabilities.availableBytes) reason = 'memory-budget';
  if (
    reason === 'supported' &&
    operation.kind === 'hot-spots-lattice' &&
    operation.width * operation.height !== rowCount
  )
    throw new Error('Hotspot lattice shape does not match source rows');
  if (reason === 'supported') return {backend: 'luma', reason};
  // Statistical operations require an actual statistical executor, not a generic SQL fallback.
  return {
    backend:
      recipe.backendPolicy !== 'require-gpu' && capabilities.analysisBackend
        ? 'analysis-backend'
        : 'unavailable',
    reason
  };
}

export type KeplerExecutionExecutors<Result> = {
  /** Releases a completed result that cancellation prevents publishing. */
  discardResult: (result: Result) => void | Promise<void>;
  luma?: (
    request: Extract<KeplerExecutionRequest, {kind: 'analysis'}>,
    signal?: AbortSignal
  ) => Promise<Result>;
  duckdb?: (
    request: Extract<KeplerExecutionRequest, {kind: 'sql'}>,
    signal?: AbortSignal
  ) => Promise<Result>;
  analysisBackend?: (
    request: Extract<KeplerExecutionRequest, {kind: 'analysis'}>,
    signal?: AbortSignal
  ) => Promise<Result>;
};

/** Exceptions stay on the selected route; dispatch failures do not trigger an unannounced rerun. */
export async function executeKeplerRequest<Result>(
  request: KeplerExecutionRequest,
  capabilities: KeplerExecutionCapabilities,
  executors: KeplerExecutionExecutors<Result>,
  signal?: AbortSignal
): Promise<{route: KeplerExecutionRoute; result: Result}> {
  signal?.throwIfAborted();
  const route = selectKeplerExecutionRoute(request, capabilities);
  let result: Result;
  if (route.backend === 'duckdb' && request.kind === 'sql' && executors.duckdb)
    result = await executors.duckdb(request, signal);
  else if (route.backend === 'luma' && request.kind === 'analysis' && executors.luma)
    result = await executors.luma(request, signal);
  else if (
    route.backend === 'analysis-backend' &&
    request.kind === 'analysis' &&
    executors.analysisBackend
  )
    result = await executors.analysisBackend(request, signal);
  else throw new Error(`Kepler execution unavailable: ${route.reason}`);
  // A cancellation may arrive while submitted GPU/SQL work finishes; never publish its stale result.
  if (signal?.aborted) {
    await executors.discardResult(result);
    signal.throwIfAborted();
  }
  return {route, result};
}
