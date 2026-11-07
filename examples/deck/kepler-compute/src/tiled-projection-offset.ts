// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  MAXIMUM_TILED_DIMENSION,
  MAXIMUM_TILED_PIXELS,
  type ImageTile
} from './native-tiled-capture';

/**
 * Off-axis projection for tiles of a pitched (perspective) view. A pitch-0 tile is a translated
 * Web Mercator viewport (see getTileViewState); under pitch a sub-rectangle of the frustum is not,
 * so the tile must keep the full view's camera and instead post-multiply the projection by a clip
 * space scale and shift: x' = scaleX * x + offsetX * w, y' = scaleY * y + offsetY * w.
 *
 * This module is the exact, unit-tested math. Applying it needs a viewport `projectionMatrix`
 * override on Deck's MapView, which Kepler's map-state path does not expose, so the live host still
 * refuses pitched tiling and falls back to single-pass export (see native-export-image-tiled.tsx).
 */
export type TileProjectionOffset = {
  scaleX: number;
  scaleY: number;
  offsetX: number;
  offsetY: number;
};

export function getTileProjectionOffset(
  tile: ImageTile,
  fullWidth: number,
  fullHeight: number
): TileProjectionOffset {
  return {
    scaleX: fullWidth / tile.width,
    scaleY: fullHeight / tile.height,
    offsetX: (fullWidth - tile.width - 2 * tile.x) / tile.width,
    offsetY: (tile.height - fullHeight + 2 * tile.y) / tile.height
  };
}

/** Apply the offset to a column-major 4x4 projection matrix (as Deck and gl-matrix store them). */
export function applyTileProjectionOffset(
  projection: ArrayLike<number>,
  offset: TileProjectionOffset
): number[] {
  const result = Array.from(projection);
  for (let column = 0; column < 4; column++) {
    result[column * 4] =
      offset.scaleX * projection[column * 4] + offset.offsetX * projection[column * 4 + 3];
    result[column * 4 + 1] =
      offset.scaleY * projection[column * 4 + 1] + offset.offsetY * projection[column * 4 + 3];
  }
  return result;
}

const TILED_SCALES = [2, 4, 8, 16] as const;

/** Largest allowed tiled scale for a map of the given size (null when even 2x does not fit). */
export function getAllowedTiledScales(mapWidth: number, mapHeight: number): number[] {
  return TILED_SCALES.filter(scale => {
    const width = Math.round(mapWidth * scale);
    const height = Math.round(mapHeight * scale);
    return (
      width <= MAXIMUM_TILED_DIMENSION &&
      height <= MAXIMUM_TILED_DIMENSION &&
      width * height <= MAXIMUM_TILED_PIXELS
    );
  });
}
