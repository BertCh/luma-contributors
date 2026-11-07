// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type * as arrow from 'apache-arrow';
import type {RasterTerrainOptions} from './analysis2-raster-terrain';
import {appendZonalColumns, decodeZonalPolygons} from './raster-live-zonal';
import {computeRasterTerrainZonalStatistics} from './raster-terrain-zonal-gpu';
import type {NumericRasterSource} from './raster-tile-ingest';

export type RasterTerrainZonalTableResult = Readonly<{
  table?: arrow.Table;
  skippedRowCount: number;
  engine: 'gpu' | 'cpu-reference';
  engineNote: string;
  fallbackReason?: string;
  readbackByteLength: number;
  reason?: string;
}>;

/**
 * Zonal statistics of a terrain derivative for every polygon row, appended as `zonal_*` columns.
 * On the GPU the derived grid stays on the device and only the zone columns are read back.
 */
export async function computeTerrainZonalTable(options: {
  source: NumericRasterSource;
  device: Device | undefined;
  table: arrow.Table;
  geometryColumn: string;
  terrain: RasterTerrainOptions;
}): Promise<RasterTerrainZonalTableResult> {
  const geometry = options.table.getChild(options.geometryColumn);
  if (!geometry || options.source.tiles.length === 0)
    return {
      skippedRowCount: options.table.numRows,
      engine: 'cpu-reference',
      engineNote: 'Terrain zonal: nothing to scan',
      readbackByteLength: 0,
      reason: 'No loaded raster tiles'
    };
  const zonePolygons = Array.from(
    {length: options.table.numRows},
    (_, rowIndex) => decodeZonalPolygons(geometry.get(rowIndex)) ?? null
  );
  const result = await computeRasterTerrainZonalStatistics({
    source: options.source,
    device: options.device,
    zonePolygons,
    terrain: options.terrain
  });
  const skippedRowCount = result.statistics.filter(statistics => !statistics).length;
  const base = {
    skippedRowCount,
    engine: result.engine,
    engineNote: result.engineNote,
    fallbackReason: result.fallbackReason,
    readbackByteLength: result.readbackByteLength
  };
  if (skippedRowCount === result.statistics.length)
    return {...base, reason: 'No polygon row intersects a loaded raster tile'};
  return {...base, table: appendZonalColumns(options.table, result.statistics)};
}
