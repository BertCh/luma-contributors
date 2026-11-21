// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {EXACT_ORIENTATION_WGSL} from '../../../src/analysis-core/kernels/exact-orientation-wgsl';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {TRIANGLE_CIRCUMCIRCLE_WGSL} from '../../../src/gpu-spatial-analysis/shared/triangle-circumcircle-wgsl';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

// CORE-103 device probe: the unchanged composed predicate (orientSign) and construction
// (getTriangleCircumcircle) on thin integer triangles whose scaled edge cross products cancel
// when the products are rounded separately. Expected values are exact rationals (Python
// fractions; see triangle-circumcircle-cancellation.node.spec.ts). Each row reads
// [predicate sign, circle sign, radius, centerX, centerY].
const TRIANGLES: {
  name: string;
  positions: number[];
  center: [number, number];
  orientation: number;
  radius: number;
}[] = [
  {
    name: 'thin 8192',
    positions: [0, 0, 8192, 8191, 8191, 8190],
    center: [-67084289.5, 67100671.5],
    orientation: -1,
    radius: 94883096.5634592
  },
  {
    name: 'thin 2^23',
    positions: [0, 0, 8388608, 8388607, 8388607, 8388606],
    center: [-70368719011841.5, 70368735789055.5],
    orientation: -1,
    radius: 99516408656650.2
  }
];

// Exact dyadic algebra, distinct from product cancellation in the integer triangles:
// det = u^2 - 2u, X = Y = (2-u^2)/(4-2u), R = sqrt(2)*(1-X), u = 2^-26.
const dyadicPositions = [1, 1, 2 ** -26, 0, 0, 2 ** -26];
const DYADIC_TRIANGLES = [0, 2, 4].map(offset => ({
  name: `dyadic cyclic rotation ${offset / 2}`,
  positions: [...dyadicPositions.slice(offset), ...dyadicPositions.slice(0, offset)],
  center: [0.5000000037252903, 0.5000000037252903] as [number, number],
  orientation: -1,
  radius: 0.7071067759181915
}));

// B2 exact rational linear-system reference (see Node spec); the original nonzero
// rounded determinant produced center (0,1), R=1 rather than this finite circle.
const asymmetricPositions = [1, 1, 0, 2 ** -26, 2 ** -24, 0];
const ASYMMETRIC_TRIANGLES = [0, 2, 4].map(offset => ({
  name: `asymmetric cyclic rotation ${offset / 2}`,
  positions: [...asymmetricPositions.slice(offset), ...asymmetricPositions.slice(0, offset)],
  center: [0.2000000247359275, 0.7999999871850011] as [number, number],
  orientation: 1,
  radius: 0.8246211042342516
}));

it.each([
  {name: 'thin integer', triangles: TRIANGLES, relativeTolerance: 2 ** -9},
  {name: 'dyadic edge construction', triangles: DYADIC_TRIANGLES, relativeTolerance: 2 ** -21},
  {name: 'asymmetric dyadic', triangles: ASYMMETRIC_TRIANGLES, relativeTolerance: 2 ** -21}
])('getTriangleCircumcircle keeps the circle of $name triangles on the device (CORE-103)', async ({
  triangles,
  relativeTolerance
}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Last row: a well-conditioned right triangle at the same scale, centre (4096, 4095.5).
  const positions = Float32Array.from([
    ...triangles.flatMap(triangle => triangle.positions),
    0,
    0,
    8192,
    0,
    0,
    8191
  ]);
  const rows = triangles.length + 1;
  const input = createInputBuffer(device, positions);
  const output = createOutputBuffer(device, rows * 5);
  const graph = new GPUCommandGraph(device, {id: 'circumcircle-cancellation-test'});
  const inputView = importGraphBuffer(graph, 'positions', input, 'float32', rows * 6);
  const outputView = importGraphBuffer(graph, 'circles', output, 'float32', rows * 5);
  graph.add({
    getCommandNodes: target => [
      createWGSLKernelNode(target, {
        id: 'circumcircle-cancellation-kernel',
        operation: 'CircumcircleCancellationTest',
        bindings: [
          {name: 'positions', view: inputView, type: 'f32', access: 'read'},
          {name: 'circles', view: outputView, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `${EXACT_ORIENTATION_WGSL}\n${TRIANGLE_CIRCUMCIRCLE_WGSL}`,
        body: `let base = positionsOffset + index * 6u;
  let first = vec2f(positions[base], positions[base + 1u]);
  let second = vec2f(positions[base + 2u], positions[base + 3u]);
  let third = vec2f(positions[base + 4u], positions[base + 5u]);
  let circle = getTriangleCircumcircle(first, second, third);
  let row = circlesOffset + index * 5u;
  circles[row] = f32(orientSign(first, second, third));
  circles[row + 1u] = f32(circle.orientation);
  circles[row + 2u] = circle.radius;
  circles[row + 3u] = circle.center.x;
  circles[row + 4u] = circle.center.y;`
      })
    ]
  });
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const values = await readFloat32(output);
    expect(values).toHaveLength(rows * 5);
    expect(values.every(Number.isFinite)).toBe(true);
    triangles.forEach((triangle, row) => {
      const [predicate, sign, radius, centerX, centerY] = values.slice(row * 5, row * 5 + 5);
      expect(predicate, triangle.name).toBe(triangle.orientation);
      // Zero here with predicate -1 is the construction loss the finding describes.
      expect(sign, triangle.name).toBe(triangle.orientation);
      // Integer fixtures allow ~5e-4 of R from cancellation of large centre terms;
      // dyadic fixtures use a tighter tolerance because shifting the origin retains their short edge.
      expect(
        Math.hypot(centerX - triangle.center[0], centerY - triangle.center[1]),
        triangle.name
      ).toBeLessThan(triangle.radius * relativeTolerance);
      expect(Math.abs(radius - triangle.radius), triangle.name).toBeLessThan(
        triangle.radius * relativeTolerance
      );
    });
    const well = values.slice(triangles.length * 5);
    expect(well[0]).toBe(1);
    expect(well[1]).toBe(1);
    expect(well[3]).toBe(4096);
    expect(well[4]).toBe(4095.5);
  } finally {
    compiled.destroy();
    input.destroy();
    output.destroy();
  }
}, 60000);
