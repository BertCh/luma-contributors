// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {evaluateBandMath, type CompiledBandMath} from './raster-band-math';
import {planRasterReprojection} from './raster-reprojection';
import type {NumericRasterSource, NumericRasterTile} from './raster-tile-ingest';

/** Polygon as GeoJSON-style rings of [lng, lat]: outer ring first, then holes. */
export type RasterZonalPolygon = readonly (readonly (readonly [number, number])[])[];

export type RasterZonalStatistics = Readonly<{
  count: number;
  sum: number;
  mean: number | null;
  minimum: number | null;
  maximum: number | null;
  nodataCount: number;
  scannedCount: number;
  tileCount: number;
  /** True when the pixel cap stopped the scan; statistics then cover a prefix only. */
  isTruncated: boolean;
}>;

export type RasterZonalOptions = Readonly<{
  bandIndex?: number;
  expression?: CompiledBandMath;
  /** Kepler uses 2,000,000. */
  maximumPixelCount?: number;
  isTileAvailable?: (tile: NumericRasterTile) => boolean;
}>;

export const KEPLER_MAXIMUM_ZONAL_PIXELS = 2000000;

/** Even-odd ray cast over all rings, so holes subtract. Boundary points are not guaranteed either way. */
export function isPointInPolygon(lng: number, lat: number, polygon: RasterZonalPolygon): boolean {
  let isInside = false;
  for (const ring of polygon)
    for (let current = 0, previous = ring.length - 1; current < ring.length; previous = current++) {
      const [currentX, currentY] = ring[current];
      const [previousX, previousY] = ring[previous];
      if (
        currentY > lat !== previousY > lat &&
        lng < ((previousX - currentX) * (lat - currentY)) / (previousY - currentY) + currentX
      )
        isInside = !isInside;
    }
  return isInside;
}

function getPolygonBounds(polygons: readonly RasterZonalPolygon[]) {
  let west = Infinity;
  let south = Infinity;
  let east = -Infinity;
  let north = -Infinity;
  for (const polygon of polygons)
    for (const ring of polygon)
      for (const [lng, lat] of ring) {
        west = Math.min(west, lng);
        east = Math.max(east, lng);
        south = Math.min(south, lat);
        north = Math.max(north, lat);
      }
  return {west, south, east, north};
}

/**
 * CPU reference zonal statistics. Pixel centers inside any polygon contribute; only tiles at
 * the highest zoom intersecting the polygon are scanned (Kepler behavior), so overlapping
 * lower-zoom parents never double count. Sums use Neumaier compensation.
 */
export function computeRasterZonalStatistics(
  source: NumericRasterSource,
  polygons: readonly RasterZonalPolygon[],
  options: RasterZonalOptions = {}
): RasterZonalStatistics {
  const bounds = getPolygonBounds(polygons);
  const plan = planRasterReprojection(source.crs, 'EPSG:4326');
  const [sourceWest, sourceSouth] = plan.transformTargetToSource(bounds.west, bounds.south);
  const [sourceEast, sourceNorth] = plan.transformTargetToSource(bounds.east, bounds.north);
  const maximumPixelCount = options.maximumPixelCount ?? KEPLER_MAXIMUM_ZONAL_PIXELS;
  const intersecting = source.tiles.filter(
    tile =>
      (!options.isTileAvailable || options.isTileAvailable(tile)) &&
      !(
        tile.extent[2] < sourceWest ||
        tile.extent[0] > sourceEast ||
        tile.extent[3] < sourceSouth ||
        tile.extent[1] > sourceNorth
      )
  );
  const maximumZoom = Math.max(...intersecting.map(tile => tile.zoom));
  const scanTiles = intersecting.filter(tile => tile.zoom === maximumZoom);
  let count = 0;
  let sum = 0;
  let compensation = 0;
  let minimum = Infinity;
  let maximum = -Infinity;
  let nodataCount = 0;
  let scannedCount = 0;
  let isTruncated = false;
  for (const tile of scanTiles) {
    const [west, south, east, north] = tile.extent;
    let values: ArrayLike<number>;
    let validity: Uint8Array;
    if (options.expression) {
      const derived = evaluateBandMath(tile, options.expression);
      values = derived.values;
      validity = derived.validity;
    } else {
      const band = tile.bands.find(candidate => candidate.index === (options.bandIndex ?? 1));
      // Binding error: the requested band is not part of this source.
      if (!band) throw new Error(`Raster tile ${tile.id} has no band ${options.bandIndex ?? 1}`);
      values = band.values;
      validity = band.validity;
    }
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
    for (let row = rowStart; row <= rowEnd && !isTruncated; row++)
      for (let column = columnStart; column <= columnEnd; column++) {
        const x = west + ((column + 0.5) / tile.width) * (east - west);
        const y = north - ((row + 0.5) / tile.height) * (north - south);
        const [lng, lat] = plan.transformSourceToTarget(x, y);
        if (!polygons.some(polygon => isPointInPolygon(lng, lat, polygon))) continue;
        if (scannedCount >= maximumPixelCount) {
          isTruncated = true;
          break;
        }
        scannedCount++;
        const sampleIndex = row * tile.width + column;
        if (!validity[sampleIndex]) {
          nodataCount++;
          continue;
        }
        const value = values[sampleIndex];
        const total = sum + value;
        compensation +=
          Math.abs(sum) >= Math.abs(value) ? sum - total + value : value - total + sum;
        sum = total;
        count++;
        if (value < minimum) minimum = value;
        if (value > maximum) maximum = value;
      }
  }
  const exactSum = sum + compensation;
  return {
    count,
    sum: exactSum,
    mean: count ? exactSum / count : null,
    minimum: count ? minimum : null,
    maximum: count ? maximum : null,
    nodataCount,
    scannedCount,
    tileCount: scanTiles.length,
    isTruncated
  };
}
