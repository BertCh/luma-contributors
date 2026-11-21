// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPURasterEmbeddingKNearestClassifier} from '../../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-knn-classifier';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createRasterTimeSeriesHarness} from '../raster-time-series/raster-time-series-test-utils';

for (const searchStructure of ['brute-force', 'tiled'] as const) {
  for (const metric of ['euclidean', 'cosine'] as const) {
    for (const neighborCount of [1, 2, 4]) {
      it(`RT-141 ${searchStructure} ${metric} preserves cross-tile ties with k=${neighborCount}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const harness = createRasterTimeSeriesHarness(device);
        const training = Array(258 * 2).fill(NaN) as number[];
        const labels = Array(258).fill(0) as number[];
        // Only rows 0,255,256 are finite. Query [1,1] is equally distant from [2,0]
        // and [0,2] under both metrics. Order by (distance,row): 0,255,256.
        for (const [row, first, second, label] of [
          [0, 2, 0, 2],
          [255, 0, 2, 1],
          [256, 2, 0, 0]
        ] as const) {
          training[row * 2] = first;
          training[row * 2 + 1] = second;
          labels[row] = label;
        }
        training[257 * 2] = Infinity;
        const result = await harness.run(
          new GPURasterEmbeddingKNearestClassifier({
            pixelCount: 2,
            dimensions: 2,
            vectors: harness.floats([1, 1, NaN, 1]),
            trainingVectors: harness.floats(training),
            trainingLabels: harness.integers(labels),
            trainingCount: 258,
            classCount: 3,
            neighborCount,
            metric,
            searchStructure,
            output: {
              labels: harness.output('labels', 'uint32', 2),
              confidence: harness.output('confidence', 'float32', 2),
              probabilities: harness.output('probabilities', 'float32', 6)
            }
          })
        );
        const expected =
          neighborCount === 1
            ? [0, 0, 1]
            : neighborCount === 2
              ? [0, 0.5, 0.5]
              : [1 / 3, 1 / 3, 1 / 3];
        expect(result['labels']).toEqual([
          neighborCount === 1 ? 2 : neighborCount === 2 ? 1 : 0,
          0xffffffff
        ]);
        expect(Number.isFinite(result['confidence'][0])).toBe(true);
        expect(result['confidence'][0]).toBeCloseTo(
          neighborCount === 1 ? 1 : neighborCount === 2 ? 0.5 : 1 / 3,
          6
        );
        for (const [index, value] of expected.entries()) {
          expect(Number.isFinite(result['probabilities'][index])).toBe(true);
          expect(result['probabilities'][index]).toBeCloseTo(value, 6);
          expect(result['probabilities'][index + 3]).toBeNaN();
        }
        expect(result['confidence'][1]).toBeNaN();
      });
    }
  }
}
