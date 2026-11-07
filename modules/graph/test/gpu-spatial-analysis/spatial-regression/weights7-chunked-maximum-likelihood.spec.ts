// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {GPUSpatialErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood';
import {GPUSpatialLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood';
import {GPUSpatialWeightsTraces} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces';
import {GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-parameters';
import {createKnnWeights, createSeededRandom} from '../permutation-inference/permutation-oracle';
import {expectWeights7ChunkInvariant} from '../spatial-weights/weights7-chunk-harness';

/**
 * Chunk-aware spatial regression consumers: weights traces (exact and stochastic), spatial lag ML
 * and spatial error ML run with one CSR and with several chunkings and must agree. Tile layouts
 * differ per chunking (tiles never span chunks), so sums agree to f32 association.
 */

const ROW_COUNT = 80;
const PREDICTOR_COUNT = 2;
const MAXIMUM_POWER = 12;

const random = createSeededRandom(23);
const positions = Float64Array.from({length: ROW_COUNT * 2}, () => random() * 100);
const csr = createKnnWeights(positions, 4, true);
const predictors = Float32Array.from({length: ROW_COUNT * PREDICTOR_COUNT}, () => random() * 2 - 1);
const response = Float32Array.from({length: ROW_COUNT}, (_, row) => {
  const x0 = predictors[row * 2];
  const x1 = predictors[row * 2 + 1];
  return 1 + 1.5 * x0 - 0.75 * x1 + Math.sin(positions[row * 2] / 13) * 2 + random() * 0.8;
});

it('chunked weights traces match the single CSR (exact and stochastic)', async () => {
  for (const mode of ['exact', 'stochastic'] as const) {
    await expectWeights7ChunkInvariant(
      csr,
      ({graph, weights, output}) => {
        graph.add(
          new GPUSpatialWeightsTraces({
            id: `traces-${mode}`,
            weights,
            maximumPower: MAXIMUM_POWER,
            mode,
            probeCount: 24,
            output: {
              powerTraces: output('powerTraces', 'float32', MAXIMUM_POWER),
              powerSums: output('powerSums', 'float32', MAXIMUM_POWER),
              infinityNorm: output('infinityNorm', 'float32', 1)
            }
          })
        );
      },
      5e-4
    );
  }
});

it('chunked spatial lag ML matches the single CSR', async () => {
  await expectWeights7ChunkInvariant(
    csr,
    ({graph, weights, output, input}) => {
      graph.add(
        new GPUSpatialLagMaximumLikelihood({
          id: 'lag-ml',
          weights,
          predictors: input('predictors', predictors),
          response: input('response', response),
          predictorCount: PREDICTOR_COUNT,
          maximumPower: MAXIMUM_POWER,
          traceMode: 'exact',
          output: {
            table: output('table', 'float32', (PREDICTOR_COUNT + 2) * 4),
            summary: output('summary', 'float32', GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH),
            status: {code: output('status', 'uint32', 1)},
            residuals: output('residuals', 'float32', ROW_COUNT)
          }
        })
      );
    },
    2e-3
  );
});

it('chunked spatial error ML matches the single CSR', async () => {
  await expectWeights7ChunkInvariant(
    csr,
    ({graph, weights, output, input}) => {
      graph.add(
        new GPUSpatialErrorMaximumLikelihood({
          id: 'error-ml',
          weights,
          predictors: input('predictors', predictors),
          response: input('response', response),
          predictorCount: PREDICTOR_COUNT,
          maximumPower: MAXIMUM_POWER,
          traceMode: 'exact',
          output: {
            table: output('table', 'float32', (PREDICTOR_COUNT + 2) * 4),
            summary: output('summary', 'float32', GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH),
            status: {code: output('status', 'uint32', 1)},
            residuals: output('residuals', 'float32', ROW_COUNT)
          }
        })
      );
    },
    2e-3
  );
});
