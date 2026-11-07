// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A seeded 2D vector field for the flow-texture runs: a drift plus a few swirls. Cells are laid out
// like a raster (row 0 first), field coordinates equal the drawing world, row 0 has the smallest y.

import type {RunInputContext} from '../../../define-run';

export type VectorFieldInputs = {
  columns: number;
  rows: number;
  /** Cell size in world units. */
  cellSize: number;
  /** (u, v) per cell in world units per second. */
  velocities: Float32Array;
  /** Speed per cell, for a context raster. */
  speeds: Float32Array;
};

export function createSwirlField(context: RunInputContext, rows = 40): VectorFieldInputs {
  const {random, world} = context;
  const columns = Math.max(24, Math.round(rows * world.width));
  const cellSize = world.height / rows;
  const swirls = Array.from({length: 4}, (_, index) => ({
    x: world.width * (0.15 + 0.7 * random()),
    y: 0.2 + 0.6 * random(),
    strength: (index % 2 === 0 ? 1 : -1) * (0.05 + 0.04 * random()),
    radius: 0.14 + 0.08 * random()
  }));
  const velocities = new Float32Array(columns * rows * 2);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const x = (column + 0.5) * cellSize;
      const y = (row + 0.5) * cellSize;
      let u = 0.12;
      let v = 0.03 * Math.sin(x * 5);
      for (const swirl of swirls) {
        const deltaX = x - swirl.x;
        const deltaY = y - swirl.y;
        const weight = swirl.strength * Math.exp(-(deltaX ** 2 + deltaY ** 2) / swirl.radius ** 2);
        u += -deltaY * weight * 20;
        v += deltaX * weight * 20;
      }
      velocities[2 * (row * columns + column)] = u;
      velocities[2 * (row * columns + column) + 1] = v;
    }
  }
  const speeds = new Float32Array(columns * rows);
  for (let cell = 0; cell < speeds.length; cell++) {
    speeds[cell] = Math.hypot(velocities[2 * cell], velocities[2 * cell + 1]);
  }
  return {columns, rows, cellSize, velocities, speeds};
}
