// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {GPUSpatialErrorGM} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm';
import {GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-error-gm-parameters';
import {createKnnWeights, createSeededRandom} from '../permutation-inference/permutation-oracle';
import {expectWeights7ChunkInvariant} from '../spatial-weights/weights7-chunk-harness';

/**
 * The Kelejian-Prucha spatial error GM estimator reads the weights through per-chunk tile and row
 * kernels; every chunking must reproduce the single-CSR fit to f32 association. The homoskedastic
 * estimator needs the Gram traces (W'), so it stays whole-only.
 */

const ROW_COUNT = 80;
const PREDICTOR_COUNT = 2;

const random = createSeededRandom(41);
const positions = Float64Array.from({length: ROW_COUNT * 2}, () => random() * 100);
const csr = createKnnWeights(positions, 4, true);
const predictors = Float32Array.from({length: ROW_COUNT * PREDICTOR_COUNT}, () => random() * 2 - 1);
const response = Float32Array.from(
  {length: ROW_COUNT},
  (_, row) =>
    2 +
    predictors[row * 2] -
    0.5 * predictors[row * 2 + 1] +
    Math.cos(positions[row * 2 + 1] / 17) * 1.5 +
    random() * 0.7
);

it('chunked Kelejian-Prucha spatial error GM matches the single CSR', async () => {
  await expectWeights7ChunkInvariant(
    csr,
    ({graph, weights, output, input}) => {
      graph.add(
        new GPUSpatialErrorGM({
          id: 'error-gm',
          weights,
          predictors: input('predictors', predictors),
          response: input('response', response),
          predictorCount: PREDICTOR_COUNT,
          estimator: 'kelejian-prucha',
          output: {
            table: output('table', 'float32', (PREDICTOR_COUNT + 2) * 4),
            summary: output('summary', 'float32', GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH),
            status: {code: output('status', 'uint32', 1)},
            residuals: output('residuals', 'float32', ROW_COUNT)
          }
        })
      );
    },
    2e-3
  );
});
