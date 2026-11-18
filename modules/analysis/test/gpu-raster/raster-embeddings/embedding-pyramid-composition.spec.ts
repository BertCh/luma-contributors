// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPURasterEmbeddingPyramidLevel} from '../../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-pyramid-level';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRasterTimeSeriesHarness} from '../raster-time-series/raster-time-series-test-utils';

it('R4 a tiny representable north-pointing sum is usable rather than exact cancellation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRasterTimeSeriesHarness(device);
  const result = await harness.run(
    new GPURasterEmbeddingPyramidLevel({
      sourceWidth: 2,
      sourceHeight: 1,
      dimensions: 2,
      scale: 2,
      aggregation: {
        vectorSums: harness.floats([1, 0, -1, 2 ** -80]),
        validCount: harness.integers([1, 1])
      },
      output: {
        vectors: harness.output('vectors', 'float32', 2),
        vectorSums: harness.output('sums', 'float32', 2),
        validCount: harness.output('count', 'uint32', 1),
        validity: harness.output('validity', 'uint32', 1),
        meanLength: harness.output('meanLength', 'float32', 1)
      }
    })
  );
  expect(result['count']).toEqual([2]);
  expect(result['sums']).toEqual([0, 2 ** -80]);
  expect(result['validity']).toEqual([1]);
  expect(result['vectors'].every(Number.isFinite)).toBe(true);
  expect(result['vectors'][0]).toBeCloseTo(0, 6);
  expect(result['vectors'][1]).toBeCloseTo(1, 6);
  expect(Number.isFinite(result['meanLength'][0])).toBe(true);
  expect(result['meanLength'][0] / 2 ** -81).toBeCloseTo(1, 5);
});

const CASES = [
  {name: 'odd edge', vectors: [1, 0, 1, 0, 0, 1], mask: [1, 1, 1], sum: [2, 1], count: 3},
  {
    name: 'masked child',
    vectors: [1, 0, 0, 1, 0, 1, 0, 1],
    mask: [1, 0, 1, 1],
    sum: [1, 2],
    count: 3
  },
  {
    name: 'different child concentrations',
    vectors: [1, 0, 0, 1, 0, 1, 0, 1],
    mask: [1, 1, 1, 1],
    sum: [1, 3],
    count: 4
  },
  {
    name: 'cancelled child with support',
    vectors: [1, 0, -1, 0, 0, 1],
    mask: [1, 1, 1],
    sum: [0, 1],
    count: 3
  },
  {
    name: 'cancelled parent',
    vectors: [1, 0, -1, 0, 0, 1, 0, -1],
    mask: [1, 1, 1, 1],
    sum: [0, 0],
    count: 4
  },
  {name: 'empty support', vectors: [1, 0, 1, 0, 0, 1], mask: [0, 0, 0], sum: [0, 0], count: 0}
];

for (const fixture of CASES) {
  it(`sums/counts preserve original pixel weights across levels: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRasterTimeSeriesHarness(device);
    const sourceWidth = fixture.mask.length;
    const childWidth = Math.ceil(sourceWidth / 2);
    const vectors = harness.floats(fixture.vectors);
    const mask = harness.integers(fixture.mask);
    const child = new GPURasterEmbeddingPyramidLevel({
      id: 'child',
      sourceWidth,
      sourceHeight: 1,
      dimensions: 2,
      scale: 2,
      vectors,
      mask,
      output: {
        vectors: harness.output('childVectors', 'float32', childWidth * 2),
        validity: harness.output('childValidity', 'uint32', childWidth),
        vectorSums: harness.output('childSums', 'float32', childWidth * 2),
        validCount: harness.output('childCount', 'uint32', childWidth)
      }
    });
    const createOutput = (prefix: string) => ({
      vectors: harness.output(`${prefix}Vectors`, 'float32', 2),
      vectorSums: harness.output(`${prefix}Sums`, 'float32', 2),
      validity: harness.output(`${prefix}Validity`, 'uint32', 1),
      validCount: harness.output(`${prefix}Count`, 'uint32', 1),
      meanLength: harness.output(`${prefix}Length`, 'float32', 1)
    });
    const parent = new GPURasterEmbeddingPyramidLevel({
      id: 'parent',
      sourceWidth: childWidth,
      sourceHeight: 1,
      dimensions: 2,
      scale: 2,
      // Vector validity is deliberately absent here: a cancelled child still carries support.
      aggregation: {
        vectorSums: child.props.output.vectorSums!,
        validCount: child.props.output.validCount!
      },
      output: createOutput('parent')
    });
    const direct = new GPURasterEmbeddingPyramidLevel({
      id: 'direct',
      sourceWidth,
      sourceHeight: 1,
      dimensions: 2,
      scale: 4,
      vectors,
      mask,
      output: createOutput('direct')
    });
    const result = await harness.run([child, parent, direct]);
    const length = Math.hypot(...fixture.sum);
    for (const prefix of ['direct', 'parent']) {
      expect(result[`${prefix}Count`]).toEqual([fixture.count]);
      expect(result[`${prefix}Sums`]).toEqual(fixture.sum);
      expect(result[`${prefix}Validity`]).toEqual([Number(length > 0)]);
      if (fixture.count > 0) {
        expect(Number.isFinite(result[`${prefix}Length`][0])).toBe(true);
        expect(result[`${prefix}Length`][0]).toBeCloseTo(length / fixture.count, 6);
      } else {
        expect(result[`${prefix}Length`][0]).toBeNaN();
      }
      for (let band = 0; band < 2; band++) {
        if (length > 0) {
          expect(Number.isFinite(result[`${prefix}Vectors`][band])).toBe(true);
          expect(result[`${prefix}Vectors`][band]).toBeCloseTo(fixture.sum[band] / length, 6);
        } else {
          expect(result[`${prefix}Vectors`][band]).toBeNaN();
        }
      }
    }
    if (fixture.name === 'cancelled child with support') {
      expect(result['childCount']).toEqual([2, 1]);
      expect(result['childValidity']).toEqual([0, 1]);
      expect(result['childVectors'][0]).toBeNaN();
      expect(result['childVectors'][1]).toBeNaN();
      expect(result['childSums']).toEqual([0, 0, 0, 1]);
    }
  });
}
