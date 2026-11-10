// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A NOAA GFS wind field near Hurricane Helene for the flow-texture runs. Cells are laid out
// like a raster (row 0 first), field coordinates equal the drawing world, row 0 has the smallest y.

import {sampleHeleneWind} from '../../../../data/wind';
import type {RunInputContext} from '../../../define-run';
import {getRasterCell} from '../../../pick';

export type VectorFieldInputs = {
  columns: number;
  rows: number;
  /** Cell size in world units. */
  cellSize: number;
  /** (u, v) per cell in world units per second. */
  velocities: Float32Array;
  /** Speed per cell in m/s, for a context raster (row 0 at the top). */
  speeds: Float32Array;
  /** Speed per cell in the operation's units (the length of its u, v), same layout as `speeds`. */
  operationSpeeds: Float32Array;
};

/** The index into `speeds` and `operationSpeeds` of the cell under a drawing-world position. */
export function getFieldCellIndex(field: VectorFieldInputs, x: number, y: number): number {
  return getRasterCell(
    field.columns,
    field.rows,
    [0, 0, field.columns * field.cellSize, field.rows * field.cellSize],
    x,
    y
  ).index;
}

export function createWindField(
  context: RunInputContext,
  coordinates: 'world' | 'raster' = 'world'
): VectorFieldInputs {
  const rows = 40;
  const {world} = context;
  const columns = Math.max(24, Math.round(rows * world.width));
  const cellSize = world.height / rows;
  const {east, north} = sampleHeleneWind(columns, rows);
  const velocities = new Float32Array(columns * rows * 2);
  const speeds = new Float32Array(columns * rows);
  const operationSpeeds = new Float32Array(columns * rows);
  // Advection uses y increasing north, while the raster layer draws row zero at the top.
  // Scale the winds into interactive world units, correcting longitude spacing at 28N.
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const source = row * columns + column;
      const destination = coordinates === 'raster' ? source : (rows - row - 1) * columns + column;
      const u = east[source] / (100 * Math.cos((28 * Math.PI) / 180));
      const v = north[source] / 100;
      velocities[2 * destination] = u;
      velocities[2 * destination + 1] = coordinates === 'raster' ? -v : v;
      speeds[source] = Math.hypot(east[source], north[source]);
      operationSpeeds[source] = Math.hypot(u, v);
    }
  }
  return {columns, rows, cellSize, velocities, speeds, operationSpeeds};
}
