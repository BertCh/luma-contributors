// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Inputs shared by the regionalization runs: contiguous tracts with a two-column attribute table
// (two smooth fields with an edge between "provinces"), and rook weights.

import {createSmoothField, createTracts, type TractsInput} from '../../../inputs';
import type {RunBuffer} from '../../../define-run';

export type RegionScene = {
  tracts: TractsInput;
  columnCount: number;
  /** Row-major: tract i, column c at i * columnCount + c. */
  attributes: Float32Array;
  /** An extensive attribute (population) per tract. */
  population: Float32Array;
};

export function createRegionScene(
  random: () => number,
  world: {width: number; height: number},
  rows = 7,
  columnCount = 2
): RegionScene {
  const tracts = createTracts(random, world, rows, 0);
  const fields = Array.from({length: 2}, () => createSmoothField(random, 3, 3));
  const attributes = new Float32Array(tracts.count * columnCount);
  const population = new Float32Array(tracts.count);
  for (let tract = 0; tract < tracts.count; tract++) {
    const x = tracts.centers[tract * 2] / world.width;
    const y = tracts.centers[tract * 2 + 1] / world.height;
    for (let column = 0; column < columnCount; column++) {
      attributes[tract * columnCount + column] = fields[column](x, y) + (random() - 0.5) * 0.06;
    }
    population[tract] = 50 + 200 * fields[1](y, x) + random() * 30;
  }
  return {tracts, columnCount, attributes, population};
}

/** `[row, neighbor]` pair per adjacency, for drawing the contiguity graph. */
export function createAdjacencyEdges(tracts: TractsInput): Uint32Array {
  const pairs: number[] = [];
  const {offsets, neighbors} = tracts.contiguity;
  for (let row = 0; row < tracts.count; row++) {
    for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
      if (neighbors[slot] > row) pairs.push(row, neighbors[slot]);
    }
  }
  return Uint32Array.from(pairs);
}

/** Views a uint32 buffer of `[a, b]` pairs as the uint32x2 edge buffer the `segments` layer reads. */
export function asPairs(buffer: RunBuffer<'uint32'>): RunBuffer {
  return {...buffer, format: 'uint32x2', length: buffer.length / 2};
}

/** Label-colored tracts plus outline, for every regionalization result. */
export function drawLabels(
  scene: RegionScene,
  mesh: {
    vertices: RunBuffer<'float32x2'>;
    indices: RunBuffer<'uint32'>;
    owners: RunBuffer<'uint32'>;
    outline: RunBuffer<'float32x4'>;
  },
  labels: RunBuffer<'uint32'>
) {
  return [
    {
      layer: 'polygons' as const,
      positions: mesh.vertices,
      indices: mesh.indices,
      owners: mesh.owners,
      values: labels,
      triangleCount: scene.tracts.triangleCount
    },
    {
      layer: 'lines' as const,
      context: true,
      segments: mesh.outline,
      count: scene.tracts.outlineCount,
      alpha: 0.6
    }
  ];
}
