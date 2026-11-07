// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CompensatedSum, estimateRasterZoneSumErrorBound} from './raster-gpu-zonal-precision';
import {isPointInPolygon, type RasterZonalPolygon} from './raster-zonal';
import {planRasterReprojection} from './raster-reprojection';
import type {NumericRasterCrs, NumericRasterTile} from './raster-tile-ingest';

/** Zone value for cells outside every polygon; equals `GPU_RASTER_ZONAL_STATISTICS_NO_ZONE`. */
export const RASTER_ZONE_NONE = 0xffffffff;

export type RasterZoneRaster = Readonly<{
  zones: Uint32Array;
  /** True when a cell lies inside two or more zones; one zone id per cell cannot express that. */
  hasOverlap: boolean;
  /** Number of cells assigned to a zone. */
  assignedCount: number;
}>;

/**
 * Rasterizes zone polygons (each zone a list of polygons) onto a tile grid by pixel center, the
 * same sampling the CPU reference uses, so the GPU reduction sees the identical cell set.
 */
export function rasterizeZoneIds(
  tile: NumericRasterTile,
  crs: NumericRasterCrs,
  zonePolygons: readonly (readonly RasterZonalPolygon[] | null)[]
): RasterZoneRaster {
  const plan = planRasterReprojection(crs, 'EPSG:4326');
  const zones = new Uint32Array(tile.width * tile.height).fill(RASTER_ZONE_NONE);
  const [west, south, east, north] = tile.extent;
  let hasOverlap = false;
  let assignedCount = 0;
  zonePolygons.forEach((polygons, zoneIndex) => {
    if (!polygons?.length) return;
    let lngWest = Infinity;
    let lngEast = -Infinity;
    let latSouth = Infinity;
    let latNorth = -Infinity;
    for (const polygon of polygons)
      for (const ring of polygon)
        for (const [lng, lat] of ring) {
          lngWest = Math.min(lngWest, lng);
          lngEast = Math.max(lngEast, lng);
          latSouth = Math.min(latSouth, lat);
          latNorth = Math.max(latNorth, lat);
        }
    const [sourceWest, sourceSouth] = plan.transformTargetToSource(lngWest, latSouth);
    const [sourceEast, sourceNorth] = plan.transformTargetToSource(lngEast, latNorth);
    if (sourceEast < west || sourceWest > east || sourceNorth < south || sourceSouth > north)
      return;
    const columnStart = Math.max(0, Math.floor(((sourceWest - west) / (east - west)) * tile.width));
    const columnEnd = Math.min(
      tile.width - 1,
      Math.ceil(((sourceEast - west) / (east - west)) * tile.width)
    );
    const rowStart = Math.max(
      0,
      Math.floor(((north - sourceNorth) / (north - south)) * tile.height)
    );
    const rowEnd = Math.min(
      tile.height - 1,
      Math.ceil(((north - sourceSouth) / (north - south)) * tile.height)
    );
    for (let row = rowStart; row <= rowEnd; row++)
      for (let column = columnStart; column <= columnEnd; column++) {
        const x = west + ((column + 0.5) / tile.width) * (east - west);
        const y = north - ((row + 0.5) / tile.height) * (north - south);
        const [lng, lat] = plan.transformSourceToTarget(x, y);
        if (!polygons.some(polygon => isPointInPolygon(lng, lat, polygon))) continue;
        const cellIndex = row * tile.width + column;
        if (zones[cellIndex] !== RASTER_ZONE_NONE) hasOverlap = true;
        else assignedCount++;
        zones[cellIndex] = zoneIndex;
      }
  });
  return {zones, hasOverlap, assignedCount};
}

export type RasterZoneColumns = Readonly<{
  cellCounts: ArrayLike<number>;
  valueCounts: ArrayLike<number>;
  sums: ArrayLike<number>;
  minimums: ArrayLike<number>;
  maximums: ArrayLike<number>;
}>;

export type RasterGpuZoneStatistics = Readonly<{
  count: number;
  sum: number;
  mean: number | null;
  minimum: number | null;
  maximum: number | null;
  nodataCount: number;
  scannedCount: number;
  tileCount: number;
  isTruncated: false;
  /** Worst-case float32 serial-sum error of `sum` (see estimateRasterZoneSumErrorBound). */
  sumErrorBound?: number;
}>;

/** Merges per-tile GPU zone columns (one entry per tile) into per-zone statistics. */
export function mergeRasterZoneColumns(
  zoneCount: number,
  tileColumns: readonly RasterZoneColumns[]
): (RasterGpuZoneStatistics | null)[] {
  return Array.from({length: zoneCount}, (_, zoneIndex) => {
    let scannedCount = 0;
    let count = 0;
    const sumAccumulator = new CompensatedSum();
    let maximumTileValueCount = 0;
    let minimum = Infinity;
    let maximum = -Infinity;
    for (const columns of tileColumns) {
      scannedCount += columns.cellCounts[zoneIndex];
      const valueCount = columns.valueCounts[zoneIndex];
      count += valueCount;
      if (valueCount > 0) {
        sumAccumulator.add(columns.sums[zoneIndex]);
        maximumTileValueCount = Math.max(maximumTileValueCount, valueCount);
        minimum = Math.min(minimum, columns.minimums[zoneIndex]);
        maximum = Math.max(maximum, columns.maximums[zoneIndex]);
      }
    }
    if (scannedCount === 0) return null;
    const sum = sumAccumulator.value;
    return {
      count,
      sum,
      mean: count ? sum / count : null,
      minimum: count ? minimum : null,
      maximum: count ? maximum : null,
      nodataCount: scannedCount - count,
      scannedCount,
      tileCount: tileColumns.length,
      isTruncated: false as const,
      sumErrorBound: count
        ? estimateRasterZoneSumErrorBound(
            maximumTileValueCount,
            Math.max(Math.abs(minimum), Math.abs(maximum))
          ) * Math.ceil(count / Math.max(maximumTileValueCount, 1))
        : 0
    };
  });
}
