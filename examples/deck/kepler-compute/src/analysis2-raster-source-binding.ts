// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {compileBandMath} from './raster-band-math';
import {readNumericGeoTIFF} from './raster-geotiff-reader';
import type {NumericRasterSource} from './raster-tile-ingest';
import {
  computeRasterZonalStatistics,
  type RasterZonalPolygon,
  type RasterZonalStatistics
} from './raster-zonal';
import {
  computeRasterTerrainZonalStatistics,
  createTerrainRasterSource,
  type RasterTerrainOptions
} from './analysis2-raster-terrain';

/**
 * Binding of the TILES NumericRasterSource (tile layers, bitmaps, GeoTIFF bytes) to raster
 * analysis inputs: zonal statistics over a band or band-math expression, and terrain derivatives.
 * Adapter only; ingestion stays in raster-tile-ingest.ts and raster-geotiff-reader.ts.
 */
export type RasterAnalysisRequest = Readonly<
  | {kind: 'zonal'; bandIndex?: number; expression?: string; maximumPixelCount?: number}
  | {kind: 'terrain-zonal'; terrain: RasterTerrainOptions; maximumPixelCount?: number}
  | {kind: 'terrain-source'; terrain: RasterTerrainOptions}
>;

export type RasterAnalysisResult =
  | {kind: 'statistics'; statistics: RasterZonalStatistics}
  | {kind: 'source'; source: NumericRasterSource};

export function readRasterAnalysisSourceFromGeoTIFF(
  bytes: Uint8Array,
  id?: string
): NumericRasterSource {
  return readNumericGeoTIFF(bytes, id);
}

export function runRasterAnalysis(
  source: NumericRasterSource,
  polygons: readonly RasterZonalPolygon[],
  request: RasterAnalysisRequest
): RasterAnalysisResult {
  if (request.kind === 'terrain-source')
    return {kind: 'source', source: createTerrainRasterSource(source, request.terrain)};
  if (request.kind === 'terrain-zonal')
    return {
      kind: 'statistics',
      statistics: computeRasterTerrainZonalStatistics(source, polygons, request.terrain, {
        maximumPixelCount: request.maximumPixelCount
      })
    };
  return {
    kind: 'statistics',
    statistics: computeRasterZonalStatistics(source, polygons, {
      bandIndex: request.bandIndex,
      expression: request.expression ? compileBandMath(request.expression) : undefined,
      maximumPixelCount: request.maximumPixelCount
    })
  };
}
