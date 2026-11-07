// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type * as arrow from 'apache-arrow';
import type {RasterLiveController} from './raster-live-controller';
import {
  appendZonalColumns,
  computeZonalTable,
  decodeZonalPolygons,
  type RasterLiveZonalResult
} from './raster-live-zonal';
import {computeRasterZonalStatisticsGpu} from './raster-gpu-zonal';
import type {RasterZonalStatistics} from './raster-zonal';

export type RasterLiveZonalGpuResult = RasterLiveZonalResult &
  Readonly<{
    /** Which path produced the statistics. */
    engine: 'gpu' | 'cpu-fallback';
    fallbackReason?: string;
    residentTileCount?: number;
  }>;

/**
 * GPU zonal statistics for every row of a polygon table (one zone per row), appended as `zonal_*`
 * columns. Falls back to the CPU reference when zones overlap, no device exists, or the GPU path
 * throws, and says so in `engine`/`fallbackReason`.
 */
export async function computeZonalTableGpu(options: {
  controller: RasterLiveController;
  device: Device | undefined;
  rasterLayerId: string;
  table: arrow.Table;
  geometryColumn: string;
  bandIndex?: number;
}): Promise<RasterLiveZonalGpuResult> {
  const {controller, device, rasterLayerId, table, geometryColumn} = options;
  const fallback = (fallbackReason: string): RasterLiveZonalGpuResult => ({
    ...computeZonalTable({
      controller,
      rasterLayerId,
      table,
      geometryColumn,
      bandIndex: options.bandIndex
    }),
    engine: 'cpu-fallback',
    fallbackReason
  });
  if (!device) return fallback('No GPU device');
  const geometry = table.getChild(geometryColumn);
  const source = controller.refreshSource(rasterLayerId);
  if (!geometry || !source || source.tiles.length === 0) return fallback('No loaded raster tiles');
  const zonePolygons = Array.from(
    {length: table.numRows},
    (_, rowIndex) => decodeZonalPolygons(geometry.get(rowIndex)) ?? null
  );
  try {
    const result = await computeRasterZonalStatisticsGpu({
      device,
      source,
      zonePolygons,
      bandIndex: options.bandIndex,
      getResource: (tileId, bandIndex) =>
        controller.getResidentBandResource(rasterLayerId, tileId, bandIndex)
    });
    if (result.declinedReason) return fallback(result.declinedReason);
    const statistics = result.statistics as (RasterZonalStatistics | null)[];
    const skippedRowCount = statistics.filter(stats => !stats).length;
    if (skippedRowCount === statistics.length)
      return {
        table,
        statistics,
        skippedRowCount,
        reason: 'No polygon row intersects a loaded raster tile',
        engine: 'gpu'
      };
    return {
      table: appendZonalColumns(table, statistics),
      statistics,
      skippedRowCount,
      engine: 'gpu',
      residentTileCount: result.residentTileCount
    };
  } catch (error) {
    return fallback(`GPU path failed: ${error instanceof Error ? error.message : String(error)}`);
  }
}
