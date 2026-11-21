// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** Review counterexamples: nearby endpoints are distinct; stored shared endpoints stay shared. */
export const OFFSET_CURVE_ENDPOINT_CASES = [
  {
    name: 'accepted tiny cut interval retains the raw start',
    originalPositions: [2 ** -24, 0, 2, 0],
    rawPositions: [0, 1, 2, 1],
    distance: 1,
    positions: [0, 1, 2 ** -24, 1, 2, 1],
    components: [0, 3, 0, 0],
    sourceEdges: [0, 0, 0],
    status: [1, 3, 1, 3, 0, 0, 0, 0]
  },
  {
    name: 'R1 narrow hairpin retains both long legs',
    originalPositions: [0, 0, 100000, 0, 100000, 1, 0, 1],
    rawPositions: [0, 0.25, 99999.75, 0.25, 99999.75, 0.75, 0, 0.75],
    distance: 0.25,
    positions: [0, 0.25, 99999.75, 0.25, 99999.75, 0.75, 0, 0.75],
    components: [0, 4, 0, 0],
    sourceEdges: [0, 0, 1, 2],
    status: [1, 4, 1, 4, 0, 0, 0, 0]
  },
  {
    name: 'R2 clipped gap remains two components',
    originalPositions: [0, -1024, 0, 1024],
    rawPositions: [-262144, 0, 262144, 0],
    distance: 1,
    positions: [-262144, 0, -1, 0, 1, 0, 262144, 0],
    components: [0, 2, 0, 0, 2, 2, 0, 0],
    sourceEdges: [0, 0, 0, 0],
    status: [2, 4, 2, 4, 0, 0, 0, 0]
  },
  {
    name: 'R3 stored shared endpoint survives clipping',
    originalPositions: [16384, 0, 2 ** -40, 0, 0, 0],
    rawPositions: [16384, 0, 2 ** -40, 0, 0, 0],
    distance: 0,
    positions: [16384, 0, 2 ** -40, 0, 0, 0],
    components: [0, 3, 0, 0],
    sourceEdges: [0, 0, 1],
    status: [1, 3, 1, 3, 0, 0, 0, 0]
  }
];

/** Assert a horizontal offset and every source vertex, allowing extra collinear clipping cuts. */
export function expectOffsetCurveLineWithSubdivisions(
  result: Readonly<Record<string, ArrayLike<number>>>,
  expectedPositions: Float32Array,
  expectedSourceEdges: readonly number[]
): void {
  const positions = Array.from(result['positions']);
  const sourceEdges = Array.from(result['sourceEdges']);
  const status = Array.from(result['status']);
  const vertexCount = status[1];
  const expectedVertexCount = expectedPositions.length / 2;
  const diagnostic = JSON.stringify({
    positions: positions.slice(0, vertexCount * 2),
    sourceEdges: sourceEdges.slice(0, vertexCount),
    status
  });
  expect(Number.isInteger(vertexCount)).toBe(true);
  expect(vertexCount).toBeGreaterThanOrEqual(expectedVertexCount);
  expect(vertexCount).toBeLessThanOrEqual(positions.length / 2);
  expect(status).toEqual([1, vertexCount, 1, vertexCount, 0, 0, 0, 0]);
  expect(Array.from(result['components'])).toEqual([
    0,
    vertexCount,
    0,
    0,
    ...new Array(result['components'].length - 4).fill(0)
  ]);
  expect(positions[0], diagnostic).toBe(expectedPositions[0]);
  expect(positions[(vertexCount - 1) * 2], diagnostic).toBe(
    expectedPositions[expectedPositions.length - 2]
  );
  let expectedVertex = 0;
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    const horizontal = positions[vertex * 2];
    const vertical = positions[vertex * 2 + 1];
    expect(Number.isFinite(horizontal)).toBe(true);
    expect(Number.isFinite(vertical)).toBe(true);
    expect(vertical, diagnostic).toBe(expectedPositions[1]);
    if (vertex > 0) expect(horizontal, diagnostic).toBeGreaterThan(positions[(vertex - 1) * 2]);
    // No cut may skip a required original vertex. Extra cuts inherit the source of the
    // fragment ending at the next original vertex, including the long edges after the prefix.
    expect(horizontal, diagnostic).toBeLessThanOrEqual(expectedPositions[expectedVertex * 2]);
    expect(sourceEdges[vertex], diagnostic).toBe(expectedSourceEdges[expectedVertex]);
    if (horizontal === expectedPositions[expectedVertex * 2]) expectedVertex++;
  }
  expect(expectedVertex).toBe(expectedVertexCount);
}

/** Exact oblique oracle: (-3,4) is the length-five left normal of (400,300). */
export const PROJECTED_OBLIQUE_OFFSET_CASES = [
  [0, 0],
  [100000, 100000],
  [1000000, 5000000],
  [10000000, 10000000]
].flatMap(([originX, originY]) =>
  [0, 0.05].map(tolerance => ({
    name: `origin (${originX},${originY}), tolerance ${tolerance}`,
    originX,
    originY,
    tolerance,
    originalPositions: Float32Array.of(originX, originY, originX + 400, originY + 300),
    expectedPositions: Float32Array.of(originX - 3, originY + 4, originX + 397, originY + 304)
  }))
);

/** No extra cuts are expected on this single straight segment, including rounded duplicates. */
export function expectCompleteObliqueOffset(
  result: Readonly<Record<string, ArrayLike<number>>>,
  expectedPositions: Float32Array
): void {
  const diagnostic = JSON.stringify(
    Object.fromEntries(Object.entries(result).map(([name, values]) => [name, Array.from(values)]))
  );
  expect(Array.from(result['status']), diagnostic).toEqual([1, 2, 1, 2, 0, 0, 0, 0]);
  expect(Array.from(result['components']).slice(0, 4), diagnostic).toEqual([0, 2, 0, 0]);
  expect(Array.from(result['positions']).slice(0, 4), diagnostic).toEqual(
    Array.from(expectedPositions)
  );
  expect(Array.from(result['sourceEdges']).slice(0, 2), diagnostic).toEqual([0, 0]);
}
