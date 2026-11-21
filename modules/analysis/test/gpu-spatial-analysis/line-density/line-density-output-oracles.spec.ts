// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPULineDensity} from '../../../src/gpu-spatial-analysis/line-density/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {createLineOracleHarness} from '../line-clip/line-output-oracle-harness';

it('density reserves exactly three cell records for a horizontal three-cell segment', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const recordCapacity of [2, 4]) {
    for (const includeRequired of [true, false]) {
      const harness = createLineOracleHarness(device);
      const {graph, view, read} = harness;
      const lengths = view('float32', 3);
      const overflow = view('uint32', 1);
      const requiredCount = includeRequired ? view('uint32', 1) : undefined;
      graph.add(
        new GPULineDensity({
          positions: view('float32x2', 2, [0, 0.5, 3, 0.5]),
          lineOffsets: view('uint32', 2, [0, 2]),
          parameters: view('float32', 4, [0, 0, 1, 1]),
          columns: 3,
          rows: 1,
          recordCapacity,
          output: {lengths, status: {overflow, requiredCount}}
        })
      );
      const destroy = await harness.run();
      try {
        // Three unit cells intersect the segment in three unit lengths. Capacity drops a suffix.
        expect(await read(overflow)).toEqual([recordCapacity === 2 ? 1 : 0]);
        if (requiredCount) expect(await read(requiredCount)).toEqual([3]);
        const actual = await read(lengths);
        const expected = recordCapacity === 2 ? [1, 1, 0] : [1, 1, 1];
        actual.forEach((value, index) => {
          expect(Number.isFinite(value)).toBe(true);
          // f32 1/3 traversal parameters multiply by length 3: < 1e-6 absolute error.
          expect(Math.abs(value - expected[index])).toBeLessThan(1e-6);
        });
      } finally {
        destroy();
        harness.destroy();
      }
    }
  }
});

it('density counts the zero-length corner reservation at capacities two and three', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const recordCapacity of [2, 3]) {
    for (const includeRequired of [true, false]) {
      const harness = createLineOracleHarness(device);
      const {graph, view, read} = harness;
      const lengths = view('float32', 4);
      const overflow = view('uint32', 1);
      const requiredCount = includeRequired ? view('uint32', 1) : undefined;
      graph.add(
        new GPULineDensity({
          positions: view('float32x2', 2, [0.5, 0.5, 1.5, 1.5]),
          lineOffsets: view('uint32', 2, [0, 2]),
          parameters: view('float32', 4, [0, 0, 1, 1]),
          columns: 2,
          rows: 2,
          recordCapacity,
          output: {lengths, status: {overflow, requiredCount}}
        })
      );
      const destroy = await harness.run();
      try {
        // The corner advances x then y at the same parameter: record lengths are
        // sqrt(1/2), 0, sqrt(1/2), not just the two positive Shapely cell spans.
        expect(await read(overflow)).toEqual([recordCapacity === 2 ? 1 : 0]);
        if (requiredCount) expect(await read(requiredCount)).toEqual([3]);
        const actual = await read(lengths);
        const expected =
          recordCapacity === 2
            ? [0.7071067811865476, 0, 0, 0]
            : [0.7071067811865476, 0, 0, 0.7071067811865476];
        actual.forEach((value, index) => {
          expect(Number.isFinite(value)).toBe(true);
          // One f32 square root and multiplication; 1e-6 is >8 ulps at this scale.
          expect(Math.abs(value - expected[index])).toBeLessThan(1e-6);
        });
      } finally {
        destroy();
        harness.destroy();
      }
    }
  }
});
