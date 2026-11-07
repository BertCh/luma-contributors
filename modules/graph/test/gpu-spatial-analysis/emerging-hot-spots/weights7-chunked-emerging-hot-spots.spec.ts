// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {
  getGPUEmergingHotSpotsParameterValues,
  GPUEmergingHotSpots
} from '../../../src/gpu-spatial-analysis/emerging-hot-spots';
import {createKnnWeights, createSeededRandom} from '../permutation-inference/permutation-oracle';
import {expectWeights7ChunkInvariant} from '../spatial-weights/weights7-chunk-harness';

/**
 * Space-time Gi* in weights mode reads the CSR through one kernel per weights chunk. Every
 * chunking, including one-row chunks, must reproduce the single-CSR results.
 */

const CELL_COUNT = 70;
const SLICE_COUNT = 14;

const random = createSeededRandom(7);
const positions = Float64Array.from({length: CELL_COUNT * 2}, () => random() * 100);
const csr = createKnnWeights(positions, 5, false);
const values = Float32Array.from({length: CELL_COUNT * SLICE_COUNT}, (_, bin) => {
  const cell = Math.floor(bin / SLICE_COUNT);
  const slice = bin % SLICE_COUNT;
  return Math.sin(positions[cell * 2] / 11) * 4 + (cell % 9 === 0 ? slice * 0.6 : 0) + random() * 2;
});
// Some missing bins and a masked cell.
values[3 * SLICE_COUNT + 2] = Number.NaN;
values[40 * SLICE_COUNT + 9] = Number.NaN;
const mask = Uint32Array.from({length: CELL_COUNT}, (_, cell) => (cell === 17 ? 0 : 1));

it('chunked weights-mode emerging hot spots match the single CSR for every chunking', async () => {
  await expectWeights7ChunkInvariant(
    csr,
    ({graph, weights, output, input}) => {
      const binCount = CELL_COUNT * SLICE_COUNT;
      graph.add(
        new GPUEmergingHotSpots({
          values: input('values', values),
          mask: input('mask', mask),
          weights,
          selfWeight: 1,
          sliceCount: SLICE_COUNT,
          multipleTesting: 'benjamini-hochberg',
          parameters: input(
            'parameters',
            getGPUEmergingHotSpotsParameterValues({
              radius: 1,
              temporalWindow: 2,
              multipleTesting: 'benjamini-hochberg',
              confidenceLevel: 0.95,
              trendSignificanceLevel: 0.1,
              minimumTrendSliceCount: 8
            })
          ),
          output: {
            giZScores: output('giZScores', 'float32', binCount),
            pValues: output('pValues', 'float32', binCount),
            adjustedPValues: output('adjustedPValues', 'float32', binCount),
            globalStatistics: output('globalStatistics', 'float32', 4),
            trendZ: output('trendZ', 'float32', CELL_COUNT),
            trendP: output('trendP', 'float32', CELL_COUNT),
            trendS: output('trendS', 'sint32', CELL_COUNT),
            category: output('category', 'uint32', CELL_COUNT),
            hotSliceCount: output('hotSliceCount', 'uint32', CELL_COUNT),
            coldSliceCount: output('coldSliceCount', 'uint32', CELL_COUNT),
            lastValidSlice: output('lastValidSlice', 'uint32', CELL_COUNT)
          }
        })
      );
    },
    // The Gi* neighbor sums walk the same slots in the same order: results are bit-exact.
    0
  );
});
