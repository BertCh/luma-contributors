// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {evaluateBandMath, type CompiledBandMath} from './raster-band-math';
import {planRasterReprojection} from './raster-reprojection';
import type {NumericRasterSource, NumericRasterTile} from './raster-tile-ingest';

export type RasterIdentifyStatus = 'value' | 'nodata' | 'outside' | 'evicted';

export type RasterIdentifyResult = Readonly<{
  status: RasterIdentifyStatus;
  tileId?: string;
  column?: number;
  row?: number;
  /** One entry per requested band; null where the sample is nodata. */
  bandValues?: readonly (number | null)[];
  /** Band math result when an expression was supplied; null for nodata. */
  derivedValue?: number | null;
}>;

export type RasterIdentifyOptions = Readonly<{
  /** One-based band indexes to report; default all bands. */
  bandIndexes?: readonly number[];
  expression?: CompiledBandMath;
  /**
   * Residency predicate (for example `tile => residency.isTileResident(tile)`).
   * When the best covering tile is not resident the next lower-zoom resident tile is used,
   * like Kepler's loaded-tile lookup; if none is resident the result is `evicted`.
   */
  isTileAvailable?: (tile: NumericRasterTile) => boolean;
}>;

/**
 * Finds the pixel owning a point in a tile. Ownership is half-open: west and north edges belong
 * to the tile, east and south edges to the neighbour. When `isClosed` is set the east and south
 * edges are also accepted, which lets the outer boundary of a mosaic still be identified.
 */
export function getRasterTilePixel(
  tile: NumericRasterTile,
  x: number,
  y: number,
  isClosed = false
): {column: number; row: number} | null {
  const [west, south, east, north] = tile.extent;
  if (x < west || y > north) return null;
  if (isClosed ? x > east || y < south : x >= east || y <= south) {
    // The south-edge rule: a point exactly on the south edge belongs to the southern neighbour.
    return null;
  }
  const column = Math.min(tile.width - 1, Math.floor(((x - west) / (east - west)) * tile.width));
  const row = Math.min(tile.height - 1, Math.floor(((north - y) / (north - south)) * tile.height));
  return {column, row};
}

function findCandidates(
  source: NumericRasterSource,
  x: number,
  y: number
): {tile: NumericRasterTile; column: number; row: number}[] {
  for (const isClosed of [false, true]) {
    const candidates: {tile: NumericRasterTile; column: number; row: number}[] = [];
    for (const tile of source.tiles) {
      const pixel = getRasterTilePixel(tile, x, y, isClosed);
      if (pixel) candidates.push({tile, ...pixel});
    }
    if (candidates.length)
      return candidates.sort((first, second) => second.tile.zoom - first.tile.zoom);
  }
  return [];
}

/** Pure CPU reference identify at a lng/lat. */
export function identifyRasterValue(
  source: NumericRasterSource,
  lng: number,
  lat: number,
  options: RasterIdentifyOptions = {}
): RasterIdentifyResult {
  const [x, y] = planRasterReprojection(source.crs, 'EPSG:4326').transformTargetToSource(lng, lat);
  const candidates = findCandidates(source, x, y);
  if (!candidates.length) return {status: 'outside'};
  const available = options.isTileAvailable
    ? candidates.find(candidate => options.isTileAvailable?.(candidate.tile))
    : candidates[0];
  if (!available) return {status: 'evicted', tileId: candidates[0].tile.id};
  const {tile, column, row} = available;
  const sampleIndex = row * tile.width + column;
  const bandIndexes = options.bandIndexes ?? tile.bands.map(band => band.index);
  const bandValues = bandIndexes.map(bandIndex => {
    const band = tile.bands.find(candidate => candidate.index === bandIndex);
    // Asking for a band the source lacks is a binding error.
    if (!band) throw new Error(`Raster tile ${tile.id} has no band ${bandIndex}`);
    return band.validity[sampleIndex] ? band.values[sampleIndex] : null;
  });
  let derivedValue: number | null | undefined;
  if (options.expression) {
    const derivedBand = evaluateBandMath(singlePixelTile(tile, sampleIndex), options.expression);
    derivedValue = derivedBand.validity[0] ? derivedBand.values[0] : null;
  }
  const isNoData = options.expression
    ? derivedValue === null
    : bandValues.every(value => value === null);
  return {
    status: isNoData ? 'nodata' : 'value',
    tileId: tile.id,
    column,
    row,
    bandValues,
    derivedValue
  };
}

function singlePixelTile(tile: NumericRasterTile, sampleIndex: number): NumericRasterTile {
  return {
    ...tile,
    width: 1,
    height: 1,
    bands: tile.bands.map(band => ({
      ...band,
      values: band.values.slice(sampleIndex, sampleIndex + 1) as typeof band.values,
      validity: band.validity.slice(sampleIndex, sampleIndex + 1)
    }))
  };
}

export type RasterPixelReadbackPlan = Readonly<{
  tileId: string;
  bandIndex: number;
  origin: Readonly<{x: number; y: number}>;
  size: Readonly<{width: 1; height: 1}>;
  /** WebGPU texture-to-buffer copies require 256-byte aligned rows. */
  bytesPerRow: 256;
  /** Staging buffer bytes for the value copy plus the validity copy. */
  readbackByteLength: 512;
}>;

/**
 * Bounded GPU identify design: one 1x1 copy from the r32float value texture and one from the
 * r8unorm validity texture into a 512-byte staging buffer, regardless of tile size. The caller
 * maps the staging buffer and passes the bytes to `decodeRasterPixelReadback`.
 */
export function planRasterPixelReadback(
  tile: NumericRasterTile,
  bandIndex: number,
  column: number,
  row: number
): RasterPixelReadbackPlan {
  // Out-of-tile coordinates would read undefined texture memory.
  if (column < 0 || row < 0 || column >= tile.width || row >= tile.height)
    throw new Error('Pixel readback is outside the tile');
  return {
    tileId: tile.id,
    bandIndex,
    origin: {x: column, y: row},
    size: {width: 1, height: 1},
    bytesPerRow: 256,
    readbackByteLength: 512
  };
}

export function decodeRasterPixelReadback(readback: Uint8Array): number | null {
  const valueBytes = new DataView(readback.buffer, readback.byteOffset, readback.byteLength);
  return readback[256] === 0 ? null : valueBytes.getFloat32(0, true);
}
