// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {toDegrees, toRadians} from '@math.gl/core';

/**
 * Tiled PNG capture. The existing single-pass export keeps its 8,192 pixels per side / 16 megapixel
 * bound (see validateImageDimensions in native-image-capture.ts). This path renders a larger virtual
 * viewport as a grid of tiles. Every tile is a normal viewport no larger than MAXIMUM_TILE_DIMENSION
 * whose centre is shifted by the tile's pixel offset, then tiles are stitched on one canvas.
 *
 * Bounds: 32,768 pixels per side and 128 megapixels in total (512 MiB of RGBA in the stitch canvas).
 * Browsers may refuse smaller canvases than this; that failure is reported, not hidden.
 */
export const MAXIMUM_TILED_DIMENSION = 32768;
export const MAXIMUM_TILED_PIXELS = 134217728;
export const MAXIMUM_TILE_DIMENSION = 4096;
export const MAXIMUM_TILE_COUNT = 1024;

export type ImageTile = {x: number; y: number; width: number; height: number};

export type TileViewState = {
  longitude: number;
  latitude: number;
  zoom: number;
  bearing?: number;
  pitch?: number;
};

export function validateTiledImageDimensions(width: number, height: number): void {
  if (
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    width < 1 ||
    height < 1 ||
    width > MAXIMUM_TILED_DIMENSION ||
    height > MAXIMUM_TILED_DIMENSION ||
    width * height > MAXIMUM_TILED_PIXELS
  )
    // Choose up to 32768 pixels per side and 128 megapixels in total.
    throw new Error('Choose a tiled image up to 32768 pixels per side and 128 megapixels.');
}

/** Row-major grid of tiles. Edge tiles keep the nominal tile size; stitching clips them. */
export function planImageTiles(
  width: number,
  height: number,
  tileWidth: number,
  tileHeight: number
): ImageTile[] {
  validateTiledImageDimensions(width, height);
  // Tile sizes must be positive integers within the single-pass renderer limits.
  if (
    !Number.isInteger(tileWidth) ||
    !Number.isInteger(tileHeight) ||
    tileWidth < 1 ||
    tileHeight < 1 ||
    tileWidth > MAXIMUM_TILE_DIMENSION ||
    tileHeight > MAXIMUM_TILE_DIMENSION
  )
    throw new Error('Tile size is outside the supported range.');
  const columns = Math.ceil(width / tileWidth);
  const rows = Math.ceil(height / tileHeight);
  if (columns * rows > MAXIMUM_TILE_COUNT)
    throw new Error(`The image needs more than ${MAXIMUM_TILE_COUNT} tiles.`);
  const tiles: ImageTile[] = [];
  for (let row = 0; row < rows; row++)
    for (let column = 0; column < columns; column++)
      tiles.push({
        x: column * tileWidth,
        y: row * tileHeight,
        width: tileWidth,
        height: tileHeight
      });
  return tiles;
}

const WORLD_TILE_SIZE = 512;

function lngLatToWorld(longitude: number, latitude: number, worldSize: number): [number, number] {
  const sine = Math.sin(toRadians(latitude));
  return [
    ((longitude + 180) / 360) * worldSize,
    (0.5 - Math.log((1 + sine) / (1 - sine)) / (4 * Math.PI)) * worldSize
  ];
}

function worldToLngLat(x: number, y: number, worldSize: number): [number, number] {
  const longitude = (x / worldSize) * 360 - 180;
  const latitude = toDegrees(2 * Math.atan(Math.exp((0.5 - y / worldSize) * 2 * Math.PI))) - 90;
  return [longitude, latitude];
}

/**
 * View state for one tile. `viewState` describes the full virtual viewport (fullWidth x fullHeight
 * at the already scaled zoom); the result describes a viewport of tileWidth x tileHeight pixels whose
 * pixels coincide with the sub-rectangle `tile` of the full viewport. Only pitch 0 is supported
 * because a sub-rectangle of a perspective frustum is not a translated Web Mercator viewport.
 */
export function getTileViewState(
  viewState: TileViewState,
  tile: ImageTile,
  fullWidth: number,
  fullHeight: number
): TileViewState {
  // Perspective tiles would need an off-axis projection; refuse rather than seam.
  if (viewState.pitch) throw new Error('Tiled capture requires a top-down view (pitch 0).');
  const worldSize = WORLD_TILE_SIZE * 2 ** viewState.zoom;
  const offsetX = tile.x + tile.width / 2 - fullWidth / 2;
  const offsetY = tile.y + tile.height / 2 - fullHeight / 2;
  const bearing = toRadians(viewState.bearing ?? 0);
  const east = offsetX * Math.cos(bearing) - offsetY * Math.sin(bearing);
  const south = offsetX * Math.sin(bearing) + offsetY * Math.cos(bearing);
  const [centerX, centerY] = lngLatToWorld(viewState.longitude, viewState.latitude, worldSize);
  const [longitude, latitude] = worldToLngLat(centerX + east, centerY + south, worldSize);
  return {...viewState, longitude, latitude};
}

export class TiledCaptureCancelledError extends Error {
  override name = 'TiledCaptureCancelledError';
  constructor() {
    super('Image export was cancelled.');
  }
}

export type TiledCaptureOptions = {
  width: number;
  height: number;
  tileWidth: number;
  tileHeight: number;
  /** Render one tile and return a canvas (or bitmap) at least tile.width x tile.height pixels. */
  renderTile(tile: ImageTile, index: number, count: number): Promise<CanvasImageSource>;
  isCurrent(): boolean;
  /** Optional abort signal; aborting is equivalent to isCurrent() returning false. */
  signal?: AbortSignal;
  onProgress?(completed: number, count: number): void;
};

function makeStitchCanvas(width: number, height: number): OffscreenCanvas | HTMLCanvasElement {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(width, height);
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

/** Stitch rendered tiles into one PNG blob. Cancellation is checked between tiles. */
export async function captureTiledImage(options: TiledCaptureOptions): Promise<Blob> {
  const tiles = planImageTiles(
    options.width,
    options.height,
    options.tileWidth,
    options.tileHeight
  );
  const isCancelled = () => options.signal?.aborted || !options.isCurrent();
  if (isCancelled()) throw new TiledCaptureCancelledError();
  const canvas = makeStitchCanvas(options.width, options.height);
  const context = canvas.getContext('2d') as CanvasRenderingContext2D | null;
  // The browser could not allocate a canvas this large.
  if (!context) throw new Error('The browser could not allocate the stitched image.');
  for (let index = 0; index < tiles.length; index++) {
    if (isCancelled()) throw new TiledCaptureCancelledError();
    const tile = tiles[index];
    const source = await options.renderTile(tile, index, tiles.length);
    if (isCancelled()) throw new TiledCaptureCancelledError();
    const clippedWidth = Math.min(tile.width, options.width - tile.x);
    const clippedHeight = Math.min(tile.height, options.height - tile.y);
    context.drawImage(
      source,
      0,
      0,
      clippedWidth,
      clippedHeight,
      tile.x,
      tile.y,
      clippedWidth,
      clippedHeight
    );
    options.onProgress?.(index + 1, tiles.length);
  }
  if (canvas instanceof HTMLCanvasElement)
    return new Promise((resolve, reject) =>
      canvas.toBlob(
        blob => (blob ? resolve(blob) : reject(new Error('PNG encoding failed.'))),
        'image/png'
      )
    );
  return canvas.convertToBlob({type: 'image/png'});
}
