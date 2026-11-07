// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared helpers for the raster runs: wrap uploaded buffers as calibrated bands and draw rasters.

import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import type {RunBuilder, RunBuffer, RasterRunLayer} from '../../../define-run';
import type {RasterInput} from '../../../inputs';

/** The grid of a raster input, as metadata the analysis operations take. */
export function makeGridOfRaster(raster: RasterInput) {
  return makeGPURasterMetadata({
    width: raster.columns,
    height: raster.rows,
    cellSize: raster.cellSize,
    origin: [0, raster.rows * raster.cellSize]
  });
}

/** Uploads `values` and wraps them as a band; returns the buffer for drawing too. */
export function uploadBand(
  builder: RunBuilder,
  raster: RasterInput,
  id: string,
  values: Float32Array
) {
  const buffer = builder.upload(id, 'float32', values);
  const band = makeGPURasterBand(buffer.view, makeGridOfRaster(raster), {id});
  return {buffer, band};
}

/** A raster layer descriptor for a buffer on the input's grid. */
export function drawRaster(
  raster: {columns: number; rows: number},
  values: RunBuffer,
  look: Partial<RasterRunLayer> = {}
): RasterRunLayer {
  return {layer: 'raster', values, columns: raster.columns, rows: raster.rows, ...look};
}

/** Grid of `columns` square-ish cells over the world, row 0 at the top (y = world height). */
export function makeWorldGrid(world: {width: number; height: number}, columns: number) {
  const rows = Math.max(1, Math.round(columns / world.width));
  const cellSize: [number, number] = [world.width / columns, world.height / rows];
  const grid = makeGPURasterMetadata({
    width: columns,
    height: rows,
    cellSize,
    origin: [0, world.height]
  });
  return {grid, columns, rows, cellSize, cellCount: columns * rows};
}
