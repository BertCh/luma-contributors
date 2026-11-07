// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {
  getGPUMapColoringParameterValues,
  GPUMapColoring
} from '../../../src/gpu-spatial-analysis/map-coloring';
import {createWeights, createSeededRandom} from '../permutation-inference/permutation-oracle';
import {expectWeights7ChunkInvariant} from '../spatial-weights/weights7-chunk-harness';

/**
 * Map coloring over a symmetric queen-contiguity lattice with a few long-range edges. The greedy
 * result is deterministic in priority order, so chunked dispatches must reproduce the single-CSR
 * colors exactly, for both priority orders.
 */

const WIDTH = 9;
const HEIGHT = 8;
const ROW_COUNT = WIDTH * HEIGHT;

function createAdjacency() {
  const neighborSets = Array.from({length: ROW_COUNT}, () => new Set<number>());
  const connect = (a: number, b: number) => {
    if (a !== b) {
      neighborSets[a].add(b);
      neighborSets[b].add(a);
    }
  };
  for (let row = 0; row < HEIGHT; row++) {
    for (let column = 0; column < WIDTH; column++) {
      for (const [deltaRow, deltaColumn] of [
        [0, 1],
        [1, 0],
        [1, 1],
        [1, -1]
      ]) {
        const otherRow = row + deltaRow;
        const otherColumn = column + deltaColumn;
        if (otherRow < HEIGHT && otherColumn >= 0 && otherColumn < WIDTH) {
          connect(row * WIDTH + column, otherRow * WIDTH + otherColumn);
        }
      }
    }
  }
  const random = createSeededRandom(5);
  for (let edge = 0; edge < 12; edge++) {
    connect(Math.floor(random() * ROW_COUNT), Math.floor(random() * ROW_COUNT));
  }
  return createWeights(
    neighborSets.map(set => [...set].map((neighbor): [number, number] => [neighbor, 1]))
  );
}

const csr = createAdjacency();

for (const priority of ['largest-degree-first', 'random'] as const) {
  it(`chunked map coloring (${priority}) matches the single CSR for every chunking`, async () => {
    await expectWeights7ChunkInvariant(
      csr,
      ({graph, weights, output, input}) => {
        graph.add(
          new GPUMapColoring({
            id: `coloring-${priority}`,
            weights,
            priority,
            parameters: input('parameters', getGPUMapColoringParameterValues({seed: 3})),
            output: {
              colors: output('colors', 'uint32', ROW_COUNT),
              colorCount: output('colorCount', 'uint32', 1),
              conflictCount: output('conflictCount', 'uint32', 1),
              // The round count is not compared: later chunks see earlier chunks' colors within a
              // round, so a chunked run may converge in fewer rounds with the same colors.
              status: {converged: output('converged', 'uint32', 1)}
            }
          })
        );
      },
      0
    );
  });
}
