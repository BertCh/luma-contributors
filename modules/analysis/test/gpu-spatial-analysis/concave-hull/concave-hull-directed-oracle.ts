// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** Complete one-triangle readback, for a counterclockwise three-site fixture. */
export function expectRetainedCounterClockwiseTriangle(result: {
  converged: number;
  capacity: number;
  triangles: number[];
  triangleValidity: number[];
  edges: number[];
  edgeValidity: number[];
}): void {
  expect(result.converged).toBe(1);
  expect(result.capacity).toBe(1);
  expect(result.triangles.length).toBe(3);
  expect(result.triangleValidity).toEqual([1]);
  expect([
    [0, 1, 2],
    [1, 2, 0],
    [2, 0, 1]
  ]).toContainEqual(result.triangles);
  expect(result.edges.length).toBe(6);
  expect(result.edgeValidity).toEqual([1, 1, 1]);
  const pairs = [0, 1, 2].map(row => result.edges.slice(2 * row, 2 * row + 2).join(','));
  expect(pairs.sort()).toEqual(['0,1', '1,2', '2,0']);
}
