// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Seeded time-series stacks (cell-major: `cell * sliceCount + slice`) for the temporal runs.

import {createSmoothField, type Random} from '../../../inputs';

export type StackInput = {
  columns: number;
  rows: number;
  cellCount: number;
  sliceCount: number;
  /** Smooth 0..1 field per cell: where the change happens. */
  region: Float32Array;
  /** A second smooth field per cell (baseline level). */
  level: Float32Array;
  /** Noise per (cell, slice) in [-0.5, 0.5]. */
  noise: Float32Array;
  /** Cloud draw per (cell, slice) in [0, 1). */
  clouds: Float32Array;
};

export function createStackInput(
  random: Random,
  aspect: number,
  rows: number,
  sliceCount: number
): StackInput {
  const columns = Math.max(8, Math.round(rows * aspect));
  const cellCount = columns * rows;
  const regionField = createSmoothField(random, 3, 3);
  const levelField = createSmoothField(random, 3, 4);
  const region = new Float32Array(cellCount);
  const level = new Float32Array(cellCount);
  for (let cell = 0; cell < cellCount; cell++) {
    const x = ((cell % columns) + 0.5) / columns;
    const y = (Math.floor(cell / columns) + 0.5) / rows;
    region[cell] = regionField(x, y);
    level[cell] = levelField(x, y);
  }
  const noise = Float32Array.from({length: cellCount * sliceCount}, () => random() - 0.5);
  const clouds = Float32Array.from({length: cellCount * sliceCount}, () => random());
  return {columns, rows, cellCount, sliceCount, region, level, noise, clouds};
}
