// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  readRasterParameterOrigin,
  validateParameterTarget,
  validateRasterParameterExtent
} from '../raster-band-utils';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';

/** Number of float32 elements in a {@link GPUIsolines} parameter view. */
export const GPU_ISOLINES_PARAMETER_LENGTH = 8;

// Accepted rasters have at most 2^31 - 1 samples and at least two per dimension. Their largest
// dimension is below 2^30; rounding its sample anchor to float32 may reach 2^30. Keeping the
// anchor in this range makes both the anchor and the signed sample difference safe in WGSL i32.
const MAXIMUM_SAMPLE_ANCHOR = 0x40000000;

/** Per-frame settings of {@link GPUIsolines}. */
export type GPUIsolinesSettings = {
  /** Number of active leading rows of the `levels` view. Clamped by the contributor to its maximum. */
  levelCount: number;
  /**
   * World extent `[minX, minY, maxX, maxY]` of the raster. Sample `(column, row)` sits at the
   * centre of its cell, `minX + (column + 0.5) * cellWidth`; row 0 is at `minY`.
   */
  extent: readonly [number, number, number, number];
  /**
   * Float64 point that segment and polyline vertices are written relative to; add it back on the
   * CPU or as the layer's coordinate origin. Defaults to `[0, 0]` (absolute vertices). Pass the
   * raster or tile corner `[minX, minY]` for projected rasters: absolute float32 vertices are
   * quantized to 0.5 m at a UTM northing of 4.6e6 m, origin-relative ones to about `1.2e-7` of
   * their distance from the origin (1.2 cm at 100 km).
   */
  localOrigin?: readonly [number, number];
};

/**
 * Packs per-frame {@link GPUIsolines} parameters.
 *
 * Layout (float32): `[levelCount, anchorX, anchorY, cellWidth, cellHeight, columnAnchor,
 * rowAnchor, 0]`. The sample anchors are float32-exact integers near `localOrigin`, clamped to
 * the domain before float32 rounding. Their world positions `anchorX`/`anchorY` are relative to `localOrigin`, computed
 * here in float64. Shaders subtract the sample anchors in integer arithmetic before converting
 * sample coordinates to float32, so large domain offsets do not destroy nearby coordinates.
 * Without an explicit `localOrigin`, sample anchors stay zero, retaining the original layout.
 * Legacy manually packed parameters with both sample anchors zero are still interpreted as before.
 *
 * @param settings Active level count, world extent and optional local origin.
 * @param grid Frame raster grid; its width and height give the cell sizes and must match the
 * contributor's unless `GPUIsolinesProps.tile` is set: then pass the DOMAIN grid and extent,
 * and reuse identical parameters (including localOrigin) for all tiles. Tile offsets are applied
 * in domain sample coordinates by the contributor, before interpolation.
 * @param target Optional destination of at least 8 elements.
 */
export function getGPUIsolinesParameterValues(
  settings: GPUIsolinesSettings,
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_ISOLINES_PARAMETER_LENGTH)
): Float32Array {
  validateParameterTarget('getGPUIsolinesParameterValues', target, GPU_ISOLINES_PARAMETER_LENGTH);
  const {width, height} = grid;
  const {levelCount, extent} = settings;
  for (const [name, value] of [
    ['width', width],
    ['height', height]
  ] as const) {
    if (!Number.isSafeInteger(value) || value < 2) {
      throw new Error(
        `getGPUIsolinesParameterValues grid ${name} must be an integer of at least 2`
      );
    }
  }
  if (!Number.isSafeInteger(levelCount) || levelCount < 0) {
    throw new Error('Isolines levelCount must be a non-negative integer');
  }
  validateRasterParameterExtent('Isolines', extent);
  const [minX, minY, maxX, maxY] = extent;
  const [originX, originY] = readRasterParameterOrigin('Isolines', settings.localOrigin);
  target.fill(0, 0, GPU_ISOLINES_PARAMETER_LENGTH);
  target[0] = levelCount;
  const cellWidth = (maxX - minX) / width;
  const cellHeight = (maxY - minY) / height;
  const columnAnchor = settings.localOrigin ? getSampleAnchor(minX, originX, cellWidth, width) : 0;
  const rowAnchor = settings.localOrigin ? getSampleAnchor(minY, originY, cellHeight, height) : 0;
  target[1] = minX - originX + columnAnchor * cellWidth;
  target[2] = minY - originY + rowAnchor * cellHeight;
  target[3] = cellWidth;
  target[4] = cellHeight;
  target[5] = columnAnchor;
  target[6] = rowAnchor;
  return target;
}

/** Choose an exactly representable integer sample anchor near the requested world origin. */
function getSampleAnchor(
  minimum: number,
  origin: number,
  cellSize: number,
  sampleCount: number
): number {
  return Math.fround(
    Math.round(
      Math.max(0, Math.min(sampleCount - 1, MAXIMUM_SAMPLE_ANCHOR, (origin - minimum) / cellSize))
    )
  );
}
