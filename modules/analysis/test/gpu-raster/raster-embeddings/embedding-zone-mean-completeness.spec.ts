// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPURasterEmbeddingZoneMean} from '@luma-contributors/analysis/raster-analysis';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRasterTimeSeriesHarness} from '../raster-time-series/raster-time-series-test-utils';

async function getHarness() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return createRasterTimeSeriesHarness(device);
}

function expectFiniteClose(actual: number[], expected: number[], tolerance: number) {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((value, index) => {
    expect(Number.isFinite(actual[index]), `element ${index} is finite`).toBe(true);
    expect(Math.abs(actual[index] - value), `element ${index}`).toBeLessThanOrEqual(tolerance);
  });
}

type TupleCase = {
  name: string;
  capacity: number;
  count: number;
  requiredCount: number;
  overflow: number;
  zoneValidity: number;
  // Closed form: the unit vectors [1, 0] and [0, 1] with weight one each have mean direction
  // (1, 1) / sqrt(2) and weight 2; a suppressed zone is NaN.
  means: number[];
  weights: number[];
};

const tupleCases: TupleCase[] = [
  {
    name: 'complete list',
    capacity: 2,
    count: 2,
    requiredCount: 2,
    overflow: 0,
    zoneValidity: 1,
    means: [Math.SQRT1_2, Math.SQRT1_2],
    weights: [2]
  },
  {
    name: 'truncated prefix',
    capacity: 1,
    count: 1,
    requiredCount: 2,
    overflow: 1,
    zoneValidity: 1,
    means: [Number.NaN, Number.NaN],
    weights: [Number.NaN]
  },
  {
    name: 'invalid zone',
    capacity: 2,
    count: 2,
    requiredCount: 2,
    overflow: 0,
    zoneValidity: 0,
    means: [Number.NaN, Number.NaN],
    weights: [Number.NaN]
  }
];

for (const tupleCase of tupleCases) {
  it(`RT-126 zone mean over fractional tuples honours producer completeness: ${tupleCase.name}`, async () => {
    const harness = await getHarness();
    const tupleIndices = Array.from({length: tupleCase.capacity}, (_, index) => index);
    const result = await harness.run(
      new GPURasterEmbeddingZoneMean({
        pixelCount: 2,
        dimensions: 2,
        zoneCount: 1,
        vectors: harness.floats([1, 0, 0, 1]),
        tuples: {
          zoneIds: harness.integers(tupleIndices.map(() => 0)),
          cellIds: harness.integers(tupleIndices),
          fractions: harness.floats(tupleIndices.map(() => 1)),
          status: {
            count: harness.integers([tupleCase.count]),
            requiredCount: harness.integers([tupleCase.requiredCount]),
            overflow: harness.integers([tupleCase.overflow]),
            candidateOverflow: harness.integers([0])
          },
          zoneValidity: harness.integers([tupleCase.zoneValidity])
        },
        output: {
          means: harness.output('means', 'float32', 2),
          weights: harness.output('weights', 'float32', 1)
        }
      })
    );
    if (Number.isNaN(tupleCase.weights[0])) {
      expect(result['means'][0]).toBeNaN();
      expect(result['means'][1]).toBeNaN();
      expect(result['weights'][0]).toBeNaN();
    } else {
      expectFiniteClose(result['means'], tupleCase.means, 1e-6);
      expectFiniteClose(result['weights'], tupleCase.weights, 1e-6);
    }
  });
}

it('RT-127 zone mean normalizes a tiny nonzero mean whose square underflows float32', async () => {
  const harness = await getHarness();
  // [1, 0] and [-1, 2^-80] average to [0, 2^-81]; 2^-162 underflows float32, so squaring first
  // gives length 0 and no normalization. Expected unit direction [0, 1], length 2^-81, weight 2.
  const result = await harness.run(
    new GPURasterEmbeddingZoneMean({
      pixelCount: 2,
      dimensions: 2,
      zoneCount: 1,
      vectors: harness.floats([1, 0, -1, 2 ** -80]),
      zoneRaster: harness.integers([0, 0]),
      output: {
        means: harness.output('means', 'float32', 2),
        weights: harness.output('weights', 'float32', 1),
        meanLengths: harness.output('meanLengths', 'float32', 1)
      }
    })
  );
  expectFiniteClose(result['means'], [0, 1], 1e-6);
  expectFiniteClose(result['weights'], [2], 1e-6);
  expect(Number.isFinite(result['meanLengths'][0])).toBe(true);
  expect(result['meanLengths'][0] / 2 ** -81).toBeCloseTo(1, 5);
});
