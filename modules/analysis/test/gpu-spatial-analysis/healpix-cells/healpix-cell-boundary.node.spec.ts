// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getGPUAnalysisOperator} from '../../../src/gpu-spatial-analysis/ontology';
import {
  getHealpixCellBoundary,
  computeHealpixPointToCellOnCPU
} from '../../../src/gpu-spatial-analysis';

test('HEALPix display boundaries round-trip interior positions on every face and across seams', () => {
  for (const order of [0, 1, 4, 10, 14, 29]) {
    for (let face = 0; face < 12; face++) {
      const key = BigInt(face) * 4n ** BigInt(order) + (4n ** BigInt(order) - 1n) / 3n;
      const ring = getHealpixCellBoundary(key, order);
      expect(ring.length).toBe(33);
      expect(ring.at(-1)).toEqual(ring[0]);
      expect(ring.every(position => position.every(Number.isFinite))).toBe(true);
      for (let index = 1; index < ring.length; index++)
        expect(Math.abs(ring[index][0] - ring[index - 1][0])).toBeLessThanOrEqual(180);
      // At high orders a Float32 point cannot identify every cell; verify invertibility below that limit.
      if (order <= 10) {
        const center = [0, 1].map(
          component =>
            ring.slice(0, -1).reduce((sum, position) => sum + position[component], 0) / 32
        );
        const output = computeHealpixPointToCellOnCPU(Float32Array.from(center), {order});
        const recovered = BigInt(output.cells[0]) | (BigInt(output.cells[1]) << 32n);
        expect(recovered, `order ${order}, face ${face}`).toBe(key);
      }
    }
  }
});

test('HEALPix equatorial face has the expected equator and transition-latitude corners', () => {
  const ring = getHealpixCellBoundary(4n, 0, 1);
  expect(ring[0]).toEqual([0, (-Math.asin(2 / 3) * 180) / Math.PI]);
  expect(ring[1]).toEqual([45, 0]);
  expect(ring[2]).toEqual([0, (Math.asin(2 / 3) * 180) / Math.PI]);
  expect(ring[3]).toEqual([-45, 0]);
});

test('HEALPix boundary validates nested key, order and display segment limits', () => {
  expect(() => getHealpixCellBoundary(-1n, 1)).toThrow('cell');
  expect(() => getHealpixCellBoundary(48n, 1)).toThrow('cell');
  expect(() => getHealpixCellBoundary(0n, 30)).toThrow('order');
  expect(() => getHealpixCellBoundary(0n, 1, 0)).toThrow('edgeSegments');
});

test('HEALPix display helper has a catalog entry with its limits and helper role', () => {
  const entry = getGPUAnalysisOperator('getHealpixCellBoundary')!;
  expect(entry.role).toBe('helper');
  expect(entry.limits?.map(limit => limit.maximum)).toEqual([29, 64]);
});

test('order-29 polar cells retain latitude extent instead of collapsing to the pole', () => {
  const north = getHealpixCellBoundary(4n ** 29n - 1n, 29, 1);
  expect(Math.max(...north.map(position => position[1]))).toBe(90);
  expect(Math.min(...north.map(position => position[1]))).toBeLessThan(90 - 1e-8);
  const south = getHealpixCellBoundary(8n * 4n ** 29n, 29, 1);
  expect(Math.min(...south.map(position => position[1]))).toBe(-90);
  expect(Math.max(...south.map(position => position[1]))).toBeGreaterThan(-90 + 1e-8);
});
