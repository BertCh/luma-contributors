// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A raster grid over the drawing world (row 0 at the top, y = world.height), as the operations
// and the raster layer both read it.

import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';

export type RunGrid = {
  columns: number;
  rows: number;
  cellSize: number;
  grid: ReturnType<typeof makeGPURasterMetadata>;
  /** [minX, minY, maxX, maxY] of the whole world. */
  window: readonly [number, number, number, number];
  /** Cell center positions (x, y) in row-major order, row 0 at the top. */
  centers: Float32Array;
};

export function createRunGrid(world: {width: number; height: number}, rows: number): RunGrid {
  const cellSize = world.height / rows;
  const columns = Math.round(world.width / cellSize);
  const centers = new Float32Array(columns * rows * 2);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      centers[(row * columns + column) * 2] = (column + 0.5) * cellSize;
      centers[(row * columns + column) * 2 + 1] = world.height - (row + 0.5) * cellSize;
    }
  }
  return {
    columns,
    rows,
    cellSize,
    grid: makeGPURasterMetadata({
      width: columns,
      height: rows,
      cellSize,
      origin: [0, world.height]
    }),
    window: [0, 0, columns * cellSize, world.height],
    centers
  };
}
