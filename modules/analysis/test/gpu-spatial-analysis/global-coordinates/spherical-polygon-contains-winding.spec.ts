// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// FX-17: `GPUSphericalPolygonContains` sums `atan2(numerator, denominator)` per ring edge, the
// same winding as `GPUS2PolygonCovering`. A ring whose vertices carry signed zeros, seen from a
// query on a coordinate plane, makes the denominator exactly -0.0 with a positive numerator, and
// the test GPU's `atan2(y, -0.0)` returns the opposite half-turn. Each query is compared with a
// binary64 evaluation of the same winding formula. On the test GPU (2026-11-10) the kernel's
// `dot` of these buffer-read vertices rounds to +0.0 where binary64 gives -0.0, so the defect is
// not reached and the code is unchanged; this spec pins that.

import {expect, it} from 'vitest';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUSphericalPolygonContains} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-contains';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

type Vector = [number, number, number];

const INVERSE_SQUARE_ROOT_TWO = Math.fround(Math.SQRT1_2);

/** Counterclockwise wedge X, Y, (-r, 0, r) with signed zeros: dot(X, Y) is -0. */
const RING: Vector[] = [
  [1, -0, 0],
  [-0, 1, -0],
  [-INVERSE_SQUARE_ROOT_TWO, -0, INVERSE_SQUARE_ROOT_TWO]
];

/** Interior queries on the plane x = 0 (both signs of zero) and one ordinary interior query. */
const BASE_QUERIES: Vector[] = [
  [-0, 0.6, 0.8],
  [-0, 0.8, 0.6],
  [0, 0.6, 0.8],
  [0, 0.8, 0.6],
  [0.28, 0.6, 0.7494]
];

/** Signed axis permutations with determinant one: exact in float32, rings stay counterclockwise. */
const SYMMETRIES: ((vector: Vector) => Vector)[] = [
  vector => vector,
  ([x, y, z]) => [-y, x, z],
  ([x, y, z]) => [z, x, y],
  ([x, y, z]) => [y, -z, -x]
];

function crossVectors(left: Vector, right: Vector): Vector {
  return [
    left[1] * right[2] - left[2] * right[1],
    left[2] * right[0] - left[0] * right[2],
    left[0] * right[1] - left[1] * right[0]
  ];
}

function dotVectors(left: Vector, right: Vector): number {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

function roundVector(vector: Vector): Vector {
  return vector.map(value => Math.fround(value)) as Vector;
}

/** Binary64 winding with `Math.atan2`, which keeps the sign of zero. */
function computeReferenceContains(
  ring: Vector[],
  point: Vector
): {inside: boolean; negativeZero: boolean} {
  let winding = 0;
  let negativeZero = false;
  ring.forEach((first, vertex) => {
    const second = ring[(vertex + 1) % ring.length]!;
    const numerator = dotVectors(point, crossVectors(first, second));
    const denominator =
      dotVectors(first, second) - dotVectors(point, first) * dotVectors(point, second);
    negativeZero ||= Object.is(denominator, -0) && numerator !== 0;
    if (Math.abs(numerator) + Math.abs(denominator) > 1e-12) {
      winding += Math.atan2(numerator, denominator);
    }
  });
  return {inside: winding > Math.PI, negativeZero};
}

it('spherical polygon contains winding agrees with binary64 where the denominator is -0.0', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  let negativeZeroCount = 0;
  const failures: string[] = [];
  for (const [symmetryIndex, symmetry] of SYMMETRIES.entries()) {
    const ring = RING.map(vertex => roundVector(symmetry(vertex)));
    const queries = BASE_QUERIES.map(query => roundVector(symmetry(query)));
    const expected = queries.map(query => computeReferenceContains(ring, query));
    negativeZeroCount += expected.filter(reference => reference.negativeZero).length;
    const graph = new GPUCommandGraph(device);
    const buffers = [
      createInputBuffer(device, new Float32Array(ring.flat())),
      createInputBuffer(device, new Uint32Array([0, ring.length])),
      createInputBuffer(device, new Uint32Array([0, 1])),
      createInputBuffer(device, new Float32Array(queries.flat())),
      createInputBuffer(device, new Uint32Array(queries.length)),
      createOutputBuffer(device, queries.length),
      createOutputBuffer(device, queries.length)
    ];
    const view = <Format extends 'float32x3' | 'uint32'>(
      index: number,
      format: Format,
      length: number
    ) => importGraphBuffer(graph, `buffer-${index}`, buffers[index], format, length);
    graph.add(
      new GPUSphericalPolygonContains({
        positions: view(0, 'float32x3', ring.length),
        ringOffsets: view(1, 'uint32', 2),
        polygonOffsets: view(2, 'uint32', 2),
        queries: view(3, 'float32x3', queries.length),
        polygonIndices: view(4, 'uint32', queries.length),
        output: {
          contains: view(5, 'uint32', queries.length),
          status: view(6, 'uint32', queries.length)
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const contains = await readUint32(buffers[5], queries.length);
      const status = await readUint32(buffers[6], queries.length);
      queries.forEach((query, queryIndex) => {
        const reference = expected[queryIndex]!;
        if (status[queryIndex] !== 0 || contains[queryIndex] !== (reference.inside ? 1 : 0)) {
          failures.push(
            `symmetry ${symmetryIndex} query ${queryIndex} (${query.map(value => (Object.is(value, -0) ? '-0' : value))}): GPU contains ${contains[queryIndex]} status ${status[queryIndex]}, CPU ${reference.inside}`
          );
        }
      });
    } finally {
      compiled.destroy();
      buffers.forEach(buffer => buffer.destroy());
    }
  }
  // The configuration really reaches a -0.0 denominator with a nonzero numerator.
  expect(negativeZeroCount).toBeGreaterThan(0);
  expect(failures).toEqual([]);
});
