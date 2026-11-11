// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// NOAA GFS 10 m wind near Hurricane Helene (2024-09-27 02 UTC) for the Lagrangian runs. Field
// coordinates equal the drawing world: cell (i, j) is centred at ((i + 0.5), (j + 0.5)) * cellSize
// and row 0 has the smallest y. Winds are scaled to world units per time unit (1 m/s = 0.01).
// One snapshot only, so both time slices of the flow map hold the same (steady) field.

import {sampleHeleneWind} from '../../../../data/wind';

export type WindField = {
  columns: number;
  rows: number;
  cellSize: number;
  /** (u, v) per cell, row 0 at the smallest y. */
  velocities: Float32Array;
  /** Speed in m/s, row 0 at the top, for a context raster. */
  speeds: Float32Array;
};

export function createWind(aspect: number, rows = 40): WindField {
  const columns = Math.max(24, Math.round(rows * aspect));
  const cellSize = 1 / rows;
  const {east, north} = sampleHeleneWind(columns, rows);
  const velocities = new Float32Array(columns * rows * 2);
  const speeds = new Float32Array(columns * rows);
  const longitudeScale = 1 / Math.cos((28 * Math.PI) / 180);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const source = row * columns + column;
      const destination = (rows - row - 1) * columns + column;
      velocities[2 * destination] = (east[source] / 100) * longitudeScale;
      velocities[2 * destination + 1] = north[source] / 100;
      speeds[source] = Math.hypot(east[source], north[source]);
    }
  }
  return {columns, rows, cellSize, velocities, speeds};
}
