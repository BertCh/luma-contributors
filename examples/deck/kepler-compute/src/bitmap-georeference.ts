// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readNumericGeoTIFF} from './raster-geotiff-reader';
import type {NumericRasterSource} from './raster-tile-ingest';

import {clamp} from '@math.gl/core';

export type BitmapBounds = readonly [west: number, south: number, east: number, north: number];
export type BitmapCorners = readonly [
  readonly [number, number],
  readonly [number, number],
  readonly [number, number],
  readonly [number, number]
];

/** Kepler `visConfig` fields written by the bitmap bounds editor. */
export type KeplerBitmapVisConfig = Readonly<{
  boundsWest?: number;
  boundsSouth?: number;
  boundsEast?: number;
  boundsNorth?: number;
  opacity?: number;
}>;

export type BitmapGeoreference = Readonly<{
  bounds: BitmapBounds;
  /** deck.gl BitmapLayer `bounds`: a box or four [lng, lat] corners (bottom-left, top-left, top-right, bottom-right). */
  deckBounds: BitmapBounds | BitmapCorners;
  opacity: number;
  isRotated: boolean;
}>;

/**
 * Normalizes Kepler bitmap config into deck.gl bounds. Accepts the editor's four bound fields,
 * a `[west, south, east, north]` array, or four corners. The editor stores min/max already;
 * arrays and corners are normalized the same way (axis-aligned corners collapse to a box,
 * rotated corners are preserved for deck.gl and flagged `isRotated`).
 */
export function resolveBitmapGeoreference(
  input: KeplerBitmapVisConfig | BitmapBounds | BitmapCorners
): BitmapGeoreference {
  let opacity = 1;
  let bounds: BitmapBounds;
  let deckBounds: BitmapBounds | BitmapCorners;
  let isRotated = false;
  if (Array.isArray(input) && Array.isArray(input[0])) {
    const corners = input as BitmapCorners;
    const longitudes = corners.map(corner => corner[0]);
    const latitudes = corners.map(corner => corner[1]);
    bounds = [
      Math.min(...longitudes),
      Math.min(...latitudes),
      Math.max(...longitudes),
      Math.max(...latitudes)
    ];
    isRotated = !(
      corners[0][0] === corners[1][0] &&
      corners[2][0] === corners[3][0] &&
      corners[0][1] === corners[3][1] &&
      corners[1][1] === corners[2][1]
    );
    deckBounds = isRotated ? corners : bounds;
  } else if (Array.isArray(input)) {
    const [west, south, east, north] = input as BitmapBounds;
    bounds = [
      Math.min(west, east),
      Math.min(south, north),
      Math.max(west, east),
      Math.max(south, north)
    ];
    deckBounds = bounds;
  } else {
    const config = input as KeplerBitmapVisConfig;
    // The editor always writes all four fields; a partial config is unusable.
    if (
      [config.boundsWest, config.boundsSouth, config.boundsEast, config.boundsNorth].some(
        value => value === undefined
      )
    )
      throw new Error('Bitmap bounds need west, south, east and north');
    const {boundsWest, boundsSouth, boundsEast, boundsNorth} =
      config as Required<KeplerBitmapVisConfig>;
    bounds = [
      Math.min(boundsWest, boundsEast),
      Math.min(boundsSouth, boundsNorth),
      Math.max(boundsWest, boundsEast),
      Math.max(boundsSouth, boundsNorth)
    ];
    deckBounds = bounds;
    opacity = config.opacity ?? 1;
  }
  if (
    !bounds.every(Number.isFinite) ||
    bounds[0] >= bounds[2] ||
    bounds[1] >= bounds[3] ||
    bounds[0] < -180 ||
    bounds[2] > 180 ||
    bounds[1] < -90 ||
    bounds[3] > 90
  )
    throw new Error('Bitmap bounds must be a non-empty geographic box');
  return {bounds, deckBounds, opacity: clamp(opacity, 0, 1), isRotated};
}

export type BitmapNumericInput = Readonly<{
  georeference: BitmapGeoreference;
  source: NumericRasterSource;
}>;

/**
 * Numeric path: a GeoTIFF supplies both the georeference (its extent) and analysis values.
 * A plain image supplies neither, so callers must not derive values from image pixels.
 */
export function readBitmapNumericInput(
  bytes: Uint8Array,
  id: string,
  opacity = 1
): BitmapNumericInput {
  const source = readNumericGeoTIFF(bytes, id);
  const [west, south, east, north] = source.tiles[0].extent;
  // EPSG:3857 extents are meters; the deck bitmap bounds must be geographic.
  if (source.crs !== 'EPSG:4326')
    throw new Error('Bitmap numeric input requires an EPSG:4326 GeoTIFF');
  return {
    georeference: resolveBitmapGeoreference({
      boundsWest: west,
      boundsSouth: south,
      boundsEast: east,
      boundsNorth: north,
      opacity
    }),
    source
  };
}

/** Writes edited bounds back into Kepler visConfig field names, preserving other fields. */
export function writeBitmapGeoreference<Config extends KeplerBitmapVisConfig>(
  visConfig: Config,
  georeference: BitmapGeoreference
): Config {
  const [west, south, east, north] = georeference.bounds;
  return {
    ...visConfig,
    boundsWest: west,
    boundsSouth: south,
    boundsEast: east,
    boundsNorth: north,
    opacity: georeference.opacity
  };
}

/**
 * Validates edited bounds/opacity and returns the Kepler visConfig patch (field names of the
 * native bounds editor). Throws when the box is empty or outside geographic range.
 */
export function makeBitmapBoundsPatch(
  bounds: BitmapBounds,
  opacity = 1
): Pick<
  Required<KeplerBitmapVisConfig>,
  'boundsWest' | 'boundsSouth' | 'boundsEast' | 'boundsNorth' | 'opacity'
> {
  const georeference = resolveBitmapGeoreference({
    boundsWest: bounds[0],
    boundsSouth: bounds[1],
    boundsEast: bounds[2],
    boundsNorth: bounds[3],
    opacity
  });
  const [boundsWest, boundsSouth, boundsEast, boundsNorth] = georeference.bounds;
  return {boundsWest, boundsSouth, boundsEast, boundsNorth, opacity: georeference.opacity};
}

/** Cleanup contract for a bitmap overlay's image resources (object URL, ImageBitmap, GPU textures). */
export function createBitmapResourceScope(): {
  track: (release: () => void) => void;
  destroy: () => number;
} {
  const releases: (() => void)[] = [];
  return {
    track: release => void releases.push(release),
    destroy: () => {
      const count = releases.length;
      for (const release of releases.splice(0).reverse()) release();
      return count;
    }
  };
}

/** True when pixels of a plain bitmap may feed analysis; only a numeric source may. */
export function canBitmapSupplyAnalysisValues(input: BitmapNumericInput | undefined): boolean {
  return input !== undefined;
}

/** ESRI world file lines (pixel size, rotation terms, upper-left pixel center) for exported bitmaps. */
export function encodeBitmapWorldFile(bounds: BitmapBounds, width: number, height: number): string {
  const pixelWidth = (bounds[2] - bounds[0]) / width;
  const pixelHeight = (bounds[3] - bounds[1]) / height;
  return [
    pixelWidth,
    0,
    0,
    -pixelHeight,
    bounds[0] + pixelWidth / 2,
    bounds[3] - pixelHeight / 2
  ].join('\n');
}
