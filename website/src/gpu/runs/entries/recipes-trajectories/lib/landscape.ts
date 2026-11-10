// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A categorical landscape raster with its connected patches, for the landscape recipes.

import {getCoverClass} from './land-cover';

export type Landscape = {
  columns: number;
  rows: number;
  classCount: number;
  cellSizeMeters: number;
  /** Class id per cell. */
  classes: Uint32Array;
  /** Patch id per cell (4-connected components of equal class). */
  patches: Uint32Array;
  /** Class id per patch. */
  patchClasses: Uint32Array;
  patchCount: number;
};

/** Four-connected components of a generalized WorldCover land-cover map. */
export function createLandscape(
  columns: number,
  rows: number,
  grain: number,
  classCount = 4
): Landscape {
  // Merge land-cover classes into a selectable number of groups. Coarsen the sampling
  // grain before finding connected components so the same landscape can be inspected at scale.
  const groupCodes = [50, 80, 10, 30, 60];
  const classes = new Uint32Array(columns * rows);
  for (let cell = 0; cell < classes.length; cell++) {
    const column = Math.floor((cell % columns) / grain) * grain;
    const row = Math.floor(Math.floor(cell / columns) / grain) * grain;
    // The native source window is approximately 4 × 4 km. Center-crop to preserve
    // square cells at the wider display aspect.
    const code = getCoverClass(
      (column + grain / 2) / columns,
      0.5 + (rows / 2 - row - grain / 2) / columns
    );
    const category = groupCodes.indexOf(code);
    classes[cell] = Math.min(classCount - 1, category < 0 ? groupCodes.length : category);
  }
  const patches = new Uint32Array(columns * rows).fill(0xffffffff);
  const patchClasses: number[] = [];
  for (let start = 0; start < classes.length; start++) {
    if (patches[start] !== 0xffffffff) continue;
    const patch = patchClasses.length;
    patchClasses.push(classes[start]);
    const stack = [start];
    patches[start] = patch;
    while (stack.length > 0) {
      const cell = stack.pop() as number;
      const column = cell % columns;
      const row = Math.floor(cell / columns);
      for (const [dx, dy] of [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1]
      ]) {
        const x = column + dx;
        const y = row + dy;
        if (x < 0 || y < 0 || x >= columns || y >= rows) continue;
        const next = y * columns + x;
        if (patches[next] === 0xffffffff && classes[next] === classes[start]) {
          patches[next] = patch;
          stack.push(next);
        }
      }
    }
  }
  return {
    columns,
    rows,
    classCount,
    cellSizeMeters: 4000 / columns,
    classes,
    patches,
    patchClasses: Uint32Array.from(patchClasses),
    patchCount: patchClasses.length
  };
}
