// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPURasterEmbeddingKNearestClassifier} from '@luma-contributors/analysis/raster-analysis';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRasterTimeSeriesHarness} from '../raster-time-series/raster-time-series-test-utils';

const NO_LABEL = 0xffffffff;

async function getHarness() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return createRasterTimeSeriesHarness(device);
}

for (const metric of ['euclidean', 'cosine'] as const) {
  for (const neighborCount of [4, 16]) {
    for (const filledNeighborCount of [0, 1, 2, 3]) {
      it(`kNN ${metric} votes only with ${filledNeighborCount} filled slots of ${neighborCount}`, async () => {
        const harness = await getHarness();
        // Invalid rows, including sample 0 (class 0), surround the finite rows. Neither NaN nor
        // either infinity may vote, even for the zero pixel under cosine's zero-vector convention.
        const trainingVectors = Array.from({length: neighborCount * 2}, (_, index) =>
          index % 2
            ? 0
            : [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY][(index / 2) % 3]
        );
        const trainingLabels = new Array<number>(neighborCount).fill(0);
        const finiteVectors = [3, 0, 1, 0, 0, 2];
        const finiteLabels = [2, 1, 2];
        for (let sample = 0; sample < filledNeighborCount; sample++) {
          const trainingRow = sample + 1;
          trainingVectors[trainingRow * 2] = finiteVectors[sample * 2];
          trainingVectors[trainingRow * 2 + 1] = finiteVectors[sample * 2 + 1];
          trainingLabels[trainingRow] = finiteLabels[sample];
        }
        const result = await harness.run(
          new GPURasterEmbeddingKNearestClassifier({
            pixelCount: 5,
            dimensions: 2,
            metric,
            vectors: harness.floats([1, 0, 0, 0, Number.NaN, 0, Number.POSITIVE_INFINITY, 0, 1, 0]),
            mask: harness.integers([1, 1, 1, 1, 0]),
            trainingVectors: harness.floats(trainingVectors),
            trainingLabels: harness.integers(trainingLabels),
            trainingCount: neighborCount,
            classCount: 3,
            neighborCount,
            output: {
              labels: harness.output('labels', 'uint32', 5),
              confidence: harness.output('confidence', 'float32', 5),
              probabilities: harness.output('probabilities', 'float32', 15)
            }
          })
        );

        // All finite samples fit. Hand votes: none; [2]; [2, 1] (class tie); [2, 1, 2].
        const expectedLabel = [NO_LABEL, 2, 1, 2][filledNeighborCount];
        const expectedConfidence = [Number.NaN, 1, 1 / 2, 2 / 3][filledNeighborCount];
        const expectedProbabilities = [
          [Number.NaN, Number.NaN, Number.NaN],
          [0, 0, 1],
          [0, 1 / 2, 1 / 2],
          [0, 1 / 3, 2 / 3]
        ][filledNeighborCount];
        expect(result['labels']).toEqual([
          expectedLabel,
          expectedLabel,
          NO_LABEL,
          NO_LABEL,
          NO_LABEL
        ]);
        for (let pixel = 0; pixel < 5; pixel++) {
          if (pixel >= 2 || filledNeighborCount === 0) {
            expect(result['confidence'][pixel]).toBeNaN();
            for (let classIndex = 0; classIndex < 3; classIndex++) {
              expect(result['probabilities'][pixel * 3 + classIndex]).toBeNaN();
            }
          } else {
            expect(result['confidence'][pixel]).toBeCloseTo(expectedConfidence, 6);
            for (let classIndex = 0; classIndex < 3; classIndex++) {
              expect(result['probabilities'][pixel * 3 + classIndex]).toBeCloseTo(
                expectedProbabilities[classIndex],
                6
              );
            }
          }
        }
      });
    }
  }
}

it('kNN returns the missing label with no finite distances when optional outputs are absent', async () => {
  const harness = await getHarness();
  // Finite components can still overflow the squared Euclidean distance; no slot is filled.
  const result = await harness.run(
    new GPURasterEmbeddingKNearestClassifier({
      pixelCount: 1,
      dimensions: 1,
      vectors: harness.floats([3e38]),
      trainingVectors: harness.floats([-3e38]),
      trainingLabels: harness.integers([0]),
      trainingCount: 1,
      classCount: 1,
      neighborCount: 1,
      output: {labels: harness.output('labels', 'uint32', 1)}
    })
  );
  expect(result['labels']).toEqual([NO_LABEL]);
});

it('kNN fills a slot with a finite distance above the former empty-distance sentinel', async () => {
  const harness = await getHarness();
  // (1.8e19)^2 is about 3.24e38: finite float32, but above the old 3.0e38 cutoff.
  const result = await harness.run(
    new GPURasterEmbeddingKNearestClassifier({
      pixelCount: 1,
      dimensions: 1,
      vectors: harness.floats([0]),
      trainingVectors: harness.floats([Number.NaN, 1.8e19]),
      trainingLabels: harness.integers([0, 1]),
      trainingCount: 2,
      classCount: 2,
      neighborCount: 2,
      output: {
        labels: harness.output('labels', 'uint32', 1),
        confidence: harness.output('confidence', 'float32', 1),
        probabilities: harness.output('probabilities', 'float32', 2)
      }
    })
  );
  expect(result['labels']).toEqual([1]);
  expect(result['confidence']).toEqual([1]);
  expect(result['probabilities']).toEqual([0, 1]);
});
