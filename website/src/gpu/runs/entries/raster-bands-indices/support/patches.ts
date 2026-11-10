// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Dense patch labels for the patch runs: a spatial/surface segmentation of measured canyon elevations, label 0
// is background and labels 1.. identify patches, the contract of GPURasterPatchLabels.

import type {Random} from '../../../inputs';

export type PatchSeeds = {
  /** (column, row, weight) per potential patch. */
  seeds: Float32Array;
  columns: number;
  rows: number;
  elevations: Float32Array;
};

export function createPatchSeeds(
  random: Random,
  columns: number,
  rows: number,
  maximumCount: number,
  elevations: Float32Array
): PatchSeeds {
  const seeds = new Float32Array(maximumCount * 3);
  for (let index = 0; index < maximumCount; index++) {
    seeds[index * 3] = random() * columns;
    seeds[index * 3 + 1] = random() * rows;
    seeds[index * 3 + 2] = 0.15 + random() * random() * 3;
  }
  return {seeds, columns, rows, elevations};
}

/** Labels from the first `count` seeds. */
export function getPatchLabels(patchSeeds: PatchSeeds, count: number): Uint32Array {
  const {seeds, columns, rows, elevations} = patchSeeds;
  const labels = new Uint32Array(columns * rows);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      let best = 0;
      let bestDistance = Infinity;
      for (let seed = 0; seed < count; seed++) {
        const site = Math.floor(seeds[seed * 3 + 1]) * columns + Math.floor(seeds[seed * 3]);
        const reliefDifference = elevations[row * columns + column] - elevations[site];
        const distance =
          rows * rows * 0.25 * reliefDifference ** 2 +
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
