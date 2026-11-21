// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPURasterEmbeddingKMeans} from '../../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-kmeans';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createRasterTimeSeriesHarness} from '../raster-time-series/raster-time-series-test-utils';

it('retains the nonzero deterministic seed of an empty embedding k-means cluster', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRasterTimeSeriesHarness(device);
  // Both evenly spaced seeds are [3,4]. Strict distance comparison chooses cluster 0
  // for all rows. Cluster 1 has no members and must retain [3,4], not become [0,0].
  const result = await harness.run(
    new GPURasterEmbeddingKMeans({
      pixelCount: 3,
      dimensions: 2,
      vectors: harness.floats([3, 4, 3, 4, 3, 4]),
      clusterCount: 2,
      maximumIterations: 4,
      output: {
        labels: harness.output('labels', 'uint32', 3),
        counts: harness.output('counts', 'uint32', 2),
        centroids: harness.output('centroids', 'float32', 4),
        status: harness.output('status', 'uint32', 3)
      }
    })
  );
  expect(result['labels']).toEqual([0, 0, 0]);
  expect(result['counts']).toEqual([3, 0]);
  expect(result['centroids']).toEqual([3, 4, 3, 4]);
  expect(result['status'][1]).toBe(0);
  expect(result['status'][2]).toBe(1);
  expect(result['status'][0]).toBeGreaterThan(0);
  expect(result['status'][0]).toBeLessThanOrEqual(4);
});
