// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPURasterRankFilter} from '../../src/raster-analysis/gpu-raster-rank-filter';
import {GPURasterSkeleton} from '../../src/raster-analysis/gpu-raster-skeleton';
import {GPURasterEmbeddingKMeans} from '../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-kmeans';
import {makeGPURasterMetadata} from '../../src/raster-analysis/raster-grid/index';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {createCensusRasterHarness, expectCensusValues} from './census-raster-oracle-utils';

for (const withChanged of [false, true]) {
  it(`census rank filter writes explicit changed flags (present=${withChanged})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createCensusRasterHarness(device);
    const result = await harness.run(
      new GPURasterRankFilter({
        grid: makeGPURasterMetadata({width: 3, height: 3}),
        radius: 1,
        input: {
          id: 'outlier',
          format: 'float32',
          storage: {
            kind: 'buffer',
            values: harness.input('float32', [1, 1, 1, 1, 9, 1, 1, 1, 1])
          }
        },
        output: {
          values: harness.output('values', 'float32', 9),
          validity: harness.output('validity', 'uint32', 9),
          ...(withChanged ? {changed: harness.output('changed', 'uint32', 9)} : {})
        }
      })
    );
    // Every clamped 3x3 window contains at most one 9: its fifth sorted sample is 1.
    // Only the input center differs from that median, so an all-zero changed view fails.
    expectCensusValues(result['values'], [1, 1, 1, 1, 1, 1, 1, 1, 1]);
    expectCensusValues(result['validity'], [1, 1, 1, 1, 1, 1, 1, 1, 1]);
    if (withChanged) expectCensusValues(result['changed'], [0, 0, 0, 0, 1, 0, 0, 0, 0]);
  });
}

for (const withStatus of [false, true]) {
  for (const maximumIterations of [1, 4]) {
    it(`census k-means checks each count and every status word (present=${withStatus}, budget=${maximumIterations})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = createCensusRasterHarness(device);
      const result = await harness.run(
        new GPURasterEmbeddingKMeans({
          pixelCount: 5,
          dimensions: 1,
          clusterCount: 2,
          maximumIterations,
          vectors: harness.input('float32', [0, 0, 10, 10, 10]),
          output: {
            labels: harness.output('labels', 'uint32', 5),
            centroids: harness.output('centroids', 'float32', 2),
            counts: harness.output('counts', 'uint32', 2),
            ...(withStatus ? {status: harness.output('status', 'uint32', 3)} : {})
          }
        })
      );
      // Evenly spaced seed rows floor(cluster * 5 / 2) are 0 and 2: centers 0 and 10.
      // All five previously unassigned labels change in round one; none change in round two.
      // Counts are deliberately unequal, so [3,2] cannot pass by preserving the total.
      expectCensusValues(result['labels'], [0, 0, 1, 1, 1]);
      expectCensusValues(result['centroids'], [0, 10]);
      expectCensusValues(result['counts'], [2, 3]);
      if (withStatus)
        expectCensusValues(result['status'], maximumIterations === 1 ? [1, 5, 0] : [2, 0, 1]);
    });
  }
}

for (const solid of [false, true]) {
  for (const withIterationCount of [false, true]) {
    it(`census skeleton counts the fixed-point verification (solid=${solid}, present=${withIterationCount})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = createCensusRasterHarness(device);
      const center = [0, 0, 0, 0, 1, 0, 0, 0, 0];
      const result = await harness.run(
        new GPURasterSkeleton({
          grid: makeGPURasterMetadata({width: 3, height: 3}),
          input: harness.input('uint32', solid ? [1, 1, 1, 1, 1, 1, 1, 1, 1] : center),
          maximumIterations: 4,
          output: {
            skeleton: harness.output('skeleton', 'uint32', 9),
            status: {
              converged: harness.output('converged', 'uint32', 1),
              iterationLimitReached: harness.output('limit', 'uint32', 1),
              ...(withIterationCount
                ? {iterationCount: harness.output('iterations', 'uint32', 1)}
                : {})
            }
          }
        })
      );
      // An isolated center has no deletable neighbors: one verifying round.
      // A solid 3x3 block loses its boundary in one Zhang-Suen round, then verifies the center.
      expectCensusValues(result['skeleton'], center);
      expectCensusValues(result['converged'], [1]);
      expectCensusValues(result['limit'], [0]);
      if (withIterationCount) expectCensusValues(result['iterations'], [solid ? 2 : 1]);
    });
  }
}
