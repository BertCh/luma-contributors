// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A binary-shape scene shared by fill-holes, medial-axis and skeleton: a smooth landmass that fills
// the middle of the frame, with lakes inside it. A run thresholds `values` to get the mask.

import {createSmoothField, type RasterInput, type Random, type World} from '../../../inputs';

/** Score raster in [0, 1]: high in the middle, broken up by a smooth field. Row 0 is the top. */
export function createShapeRaster(
  random: Random,
  world: World,
  rows = 72,
  options: {fieldWeight?: number; stretch?: number; radius?: number} = {}
): RasterInput {
  const fieldWeight = options.fieldWeight ?? 0.45;
  const stretch = options.stretch ?? 1;
  const radius = options.radius ?? 0.55;
  const columns = Math.max(16, Math.round((rows * world.width) / world.height));
  const field = createSmoothField(random, 4, 3);
  const values = new Float32Array(columns * rows);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const unitX = (column + 0.5) / columns;
      const unitY = (row + 0.5) / rows;
      const offsetX = ((unitX - 0.5) * world.width) / stretch;
      const offsetY = (unitY - 0.5) * world.height;
      const distance = Math.min(1, Math.hypot(offsetX, offsetY) / radius);
      values[row * columns + column] =
        fieldWeight * field(unitX, unitY) + (1 - fieldWeight) * (1 - distance * distance);
    }
  }
  return {columns, rows, cellSize: world.height / rows, values};
}

/** Marks cells within `radius` cells of (column, row) in `mask`; `ringWidth` keeps only an outline. */
export function stampDisc(
  mask: Uint32Array,
  columns: number,
  rows: number,
  column: number,
  row: number,
  radius: number,
  ringWidth = Infinity
): void {
  for (
    let rowIndex = Math.max(0, row - radius - 1);
    rowIndex <= Math.min(rows - 1, row + radius + 1);
    rowIndex++
  ) {
    for (
      let columnIndex = Math.max(0, column - radius - 1);
      columnIndex <= Math.min(columns - 1, column + radius + 1);
      columnIndex++
    ) {
      const distance = Math.hypot(columnIndex - column, rowIndex - row);
      if (distance <= radius && distance > radius - ringWidth)
        mask[rowIndex * columns + columnIndex] = 1;
    }
  }
}
