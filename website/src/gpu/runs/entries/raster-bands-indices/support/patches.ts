// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Dense patch labels for the patch runs: a weighted Voronoi segmentation of a raster grid, label 0
// is background and labels 1.. identify patches, the contract of GPURasterPatchLabels.

import type {Random} from '../../../inputs';

export type PatchSeeds = {
  /** (column, row, weight) per potential patch. */
  seeds: Float32Array;
  columns: number;
  rows: number;
};

export function createPatchSeeds(
  random: Random,
  columns: number,
  rows: number,
  maximumCount: number
): PatchSeeds {
  const seeds = new Float32Array(maximumCount * 3);
  for (let index = 0; index < maximumCount; index++) {
    seeds[index * 3] = random() * columns;
    seeds[index * 3 + 1] = random() * rows;
    seeds[index * 3 + 2] = 0.15 + random() * random() * 3;
  }
  return {seeds, columns, rows};
}

/** Labels from the first `count` seeds; the top-left corner stays background. */
export function getPatchLabels(patchSeeds: PatchSeeds, count: number): Uint32Array {
  const {seeds, columns, rows} = patchSeeds;
  const labels = new Uint32Array(columns * rows);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      let best = 0;
      let bestDistance = Infinity;
      for (let seed = 0; seed < count; seed++) {
        const distance =
          ((column - seeds[seed * 3]) ** 2 + (row - seeds[seed * 3 + 1]) ** 2) /
          seeds[seed * 3 + 2];
        if (distance < bestDistance) {
          bestDistance = distance;
          best = seed;
        }
      }
      labels[row * columns + column] = best + 1;
    }
  }
  return labels;
}
