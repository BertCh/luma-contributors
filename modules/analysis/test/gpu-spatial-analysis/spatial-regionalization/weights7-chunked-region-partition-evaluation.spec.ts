// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {
  GPURegionPartitionEvaluation,
  GPU_REGION_PARTITION_EVALUATION_LAYOUT
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/index';
import {createKnnWeights, createSeededRandom} from '../permutation-inference/permutation-oracle';
import {expectWeights7ChunkInvariant} from '../spatial-weights/weights7-chunk-harness';

/**
 * Partition evaluation reads the weights only for the cross-link fraction: per-row link counts
 * are integer load-balanced sums, so every chunking reproduces the single-CSR summary exactly.
 */

const ROW_COUNT = 90;
const LABEL_CAPACITY = 8;

const random = createSeededRandom(31);
const positions = Float64Array.from({length: ROW_COUNT * 2}, () => random() * 100);
const csr = createKnnWeights(positions, 5, false);
const values = Float32Array.from({length: ROW_COUNT}, () => random() * 10);
// Spatially coherent labels, a few unlabelled rows (>= capacity).
const labels = Uint32Array.from({length: ROW_COUNT}, (_, row) =>
  row % 17 === 3
    ? 0xffffffff
    : Math.min(LABEL_CAPACITY - 1, Math.floor(positions[row * 2] / (100 / LABEL_CAPACITY)))
);

it('chunked partition evaluation matches the single CSR for every chunking', async () => {
  await expectWeights7ChunkInvariant(
    csr,
    ({graph, weights, output, input}) => {
      graph.add(
        new GPURegionPartitionEvaluation({
          values: input('values', values),
          labels: input('labels', labels),
          labelCapacity: LABEL_CAPACITY,
          weights,
          output: {
            summary: output('summary', 'float32', GPU_REGION_PARTITION_EVALUATION_LAYOUT.length),
            regionSizes: output('regionSizes', 'uint32', LABEL_CAPACITY),
            regionWithinSumOfSquares: output('within', 'float32', LABEL_CAPACITY)
          }
        })
      );
    },
    0
  );
});
