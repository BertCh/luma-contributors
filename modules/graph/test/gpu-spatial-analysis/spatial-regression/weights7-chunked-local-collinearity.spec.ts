// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {GPULocalCollinearity} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-local-collinearity';
import {createKnnWeights, createSeededRandom} from '../permutation-inference/permutation-oracle';
import {expectWeights7ChunkInvariant} from '../spatial-weights/weights7-chunk-harness';

/**
 * Local collinearity diagnostics are per-row neighbor loops, one kernel per weights chunk; every
 * chunking reproduces the single-CSR surfaces exactly.
 */

const ROW_COUNT = 70;
const COLUMN_COUNT = 3;

const random = createSeededRandom(53);
const positions = Float64Array.from({length: ROW_COUNT * 2}, () => random() * 100);
const csr = createKnnWeights(positions, 10, false);
const predictors = Float32Array.from({length: ROW_COUNT * COLUMN_COUNT}, (_, index) => {
  const row = Math.floor(index / COLUMN_COUNT);
  const column = index % COLUMN_COUNT;
  const base = random();
  return column === 1 ? base * 0.3 + positions[row * 2] / 100 : base + column;
});
const mask = Uint32Array.from({length: ROW_COUNT}, (_, row) => (row % 13 === 5 ? 0 : 1));

it('chunked local collinearity matches the single CSR for every chunking', async () => {
  await expectWeights7ChunkInvariant(
    csr,
    ({graph, weights, output, input}) => {
      graph.add(
        new GPULocalCollinearity({
          weights,
          predictors: input('predictors', predictors),
          predictorCount: COLUMN_COUNT,
          mask: input('mask', mask),
          output: {
            varianceInflationFactors: output('vif', 'float32', ROW_COUNT * COLUMN_COUNT),
            conditionNumbers: output('condition', 'float32', ROW_COUNT),
            designConditionNumbers: output('designCondition', 'float32', ROW_COUNT),
            varianceDecomposition: output(
              'decomposition',
              'float32',
              ROW_COUNT * (COLUMN_COUNT + 1)
            ),
            status: output('status', 'uint32', ROW_COUNT)
          }
        })
      );
    },
    0
  );
});
