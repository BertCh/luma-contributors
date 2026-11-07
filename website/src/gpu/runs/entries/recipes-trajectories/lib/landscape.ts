// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A categorical landscape raster with its connected patches, for the landscape recipes.

import {createSmoothField} from '../../../inputs';

export type Landscape = {
  columns: number;
  rows: number;
  classCount: number;
  /** Class id per cell. */
  classes: Uint32Array;
  /** Patch id per cell (4-connected components of equal class). */
  patches: Uint32Array;
  /** Class id per patch. */
  patchClasses: Uint32Array;
  patchCount: number;
};

/** Classes from thresholds of a smooth field; `frequency` controls how fragmented the patches are. */
export function createLandscape(
  random: () => number,
  columns: number,
  rows: number,
  frequency: number,
  classCount = 4
): Landscape {
  const field = createSmoothField(random, 3, Math.max(2, Math.round(frequency)));
  const classes = new Uint32Array(columns * rows);
  for (let cell = 0; cell < classes.length; cell++) {
    const value = field(
      ((cell % columns) + 0.5) / columns,
      (Math.floor(cell / columns) + 0.5) / rows
    );
    classes[cell] = Math.min(classCount - 1, Math.floor(value * classCount));
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
    classes,
    patches,
    patchClasses: Uint32Array.from(patchClasses),
    patchCount: patchClasses.length
  };
}

/** A small seeded generator, so a rebuild with the same seed makes the same landscape. */
export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}
