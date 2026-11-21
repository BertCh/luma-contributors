// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {GPUGeometryPredicates} from '../../../src/gpu-spatial-analysis/geometry-predicates/gpu-geometry-predicates';
import {
  createEqualityMetricFixture,
  type EqualityView
} from '../geometry-refinement/canonical-equality-fixture';
import {createWaveHarness} from '../geometry-validity/w5-test-harness';
it.each([
  'exact',
  'tree',
  'polygon'
] as const)('%s equality certifies the 13,84,85 boundary against stored operands', async kind => {
  for (const [tolerance, expected] of [
    [85, 1],
    [84.99999237060547, 0]
  ] as const) {
    const harness = createWaveHarness();
    const view: EqualityView = (format, values) =>
      harness.allocate(
        'equality',
        format,
        values.length / (format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1),
        values
      );
    const fixture =
      kind === 'exact'
        ? (() => {
            const output = harness.allocate('result', 'uint32', 1);
            return {
              output,
              contributor: new GPUGeometryPredicates({
                geometry: {kind: 'points', positions: view('float32x2', [0, 0])},
                other: {kind: 'points', positions: view('float32x2', [13, 84])},
                parameters: harness.allocate('parameters', 'float32', 4, [tolerance, 0, 0, 0]),
                output: {equalsExact: output}
              })
            };
          })()
        : createEqualityMetricFixture(kind, 'euclidean', view, [13, 84], tolerance);
    try {
      await harness.run(fixture.contributor.getCommandNodes(harness.graph));
      // 13²+84²=7225=85² exactly; the prior float32 is 85-2^-17.
      expect(await harness.read(fixture.output)).toEqual([expected]);
    } finally {
      harness.destroy();
    }
  }
});

import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GEOMETRY_EQUALITY_DISTANCE_WGSL} from '../../../src/gpu-spatial-analysis/geometry-predicates/geometry-equality-distance-wgsl';
it('Euclidean certification handles translated, tiny, signed-zero and overflowing differences', async () => {
  const harness = createWaveHarness();
  // Hand fixtures: 3²+4²=5², independent of translation and power-of-two scaling.
  // Two smallest subnormal components have squared length 2*tolerance², so fail.
  const tiny = 2 ** -80;
  const subnormal = 2 ** -149;
  const maximum = 3.4028234663852886e38;
  const coordinates = harness.allocate('operands', 'float32x4', 7, [
    2 ** 30,
    2 ** 30,
    2 ** 30 + 384,
    2 ** 30 + 512,
    0,
    0,
    3 * tiny,
    4 * tiny,
    0,
    0,
    subnormal,
    0,
    0,
    0,
    subnormal,
    subnormal,
    1,
    1,
    1,
    1,
    maximum,
    0,
    -maximum,
    0,
    -13,
    -84,
    0,
    0
  ]);
  const tolerances = harness.allocate('tolerances', 'float32', 7, [
    640,
    5 * tiny,
    subnormal,
    subnormal,
    -0,
    maximum,
    85
  ]);
  const result = harness.allocate('result', 'uint32', 7);
  const node = createWGSLKernelNode(harness.graph, {
    id: 'certificate',
    operation: 'EqualityCertificateTest',
    invocationCount: 7,
    bindings: [
      {name: 'coordinates', view: coordinates, type: 'f32', access: 'read'},
      {name: 'tolerances', view: tolerances, type: 'f32', access: 'read'},
      {name: 'result', view: result, type: 'u32', access: 'read_write'}
    ],
    declarations: GEOMETRY_EQUALITY_DISTANCE_WGSL,
    body: 'let base=coordinatesOffset+4u*index;result[resultOffset+index]=select(0u,1u,geometryWithinTolerance(vec2f(coordinates[base],coordinates[base+1u]),vec2f(coordinates[base+2u],coordinates[base+3u]),tolerances[tolerancesOffset+index]));'
  });
  try {
    await harness.run([node]);
    expect(await harness.read(result)).toEqual([1, 1, 1, 0, 1, 0, 1]);
  } finally {
    harness.destroy();
  }
});

it.each(['tree', 'polygon'] as const)('%s retains the per-coordinate default', async kind => {
  const harness = createWaveHarness();
  const view: EqualityView = (format, values) =>
    harness.allocate(
      'default',
      format,
      values.length / (format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1),
      values
    );
  const fixture = createEqualityMetricFixture(kind, undefined, view, [0.75, 0.75], 1);
  try {
    await harness.run(fixture.contributor.getCommandNodes(harness.graph));
    // Both coordinate differences are <= 1, although their Euclidean length exceeds 1.
    expect(await harness.read(fixture.output)).toEqual([1]);
  } finally {
    harness.destroy();
  }
});

import {vi} from 'vitest';
import {ScalarData, VectorData, WgslExec} from 'wgsl_reflect';
it('review B6 certifies a smaller delta flushed to zero from normal operands', async () => {
  const harness = createWaveHarness();
  const normal = 2 ** -126;
  const operands = harness.allocate('normal-operands', 'float32x4', 1, [
    1.5 * normal,
    2 * normal,
    normal,
    normal
  ]);
  const tolerance = harness.allocate('normal-tolerance', 'float32', 1, [1.0625 * normal]);
  const result = harness.allocate('answer', 'uint32', 1);
  const node = createWGSLKernelNode(harness.graph, {
    id: 'flush-control',
    operation: 'EqualityFlushTest',
    invocationCount: 1,
    bindings: [
      {name: 'operands', view: operands, type: 'f32', access: 'read'},
      {name: 'tolerance', view: tolerance, type: 'f32', access: 'read'},
      {name: 'result', view: result, type: 'u32', access: 'read_write'}
    ],
    declarations: GEOMETRY_EQUALITY_DISTANCE_WGSL,
    body: 'let base=operandsOffset;result[resultOffset]=select(0u,1u,geometryWithinTolerance(vec2f(operands[base],operands[base+1u]),vec2f(operands[base+2u],operands[base+3u]),tolerance[toleranceOffset]));'
  });
  const original = WgslExec.prototype._evalBinaryOp;
  const patch = vi.spyOn(WgslExec.prototype, '_evalBinaryOp').mockImplementation(function (
    this: WgslExec,
    expression,
    context
  ) {
    const value = original.call(this, expression, context);
    if (
      ['+', '-', '*', '/'].includes(expression.operator) &&
      (value instanceof ScalarData || value instanceof VectorData) &&
      value.data instanceof Float32Array
    ) {
      for (let index = 0; index < value.data.length; index++) {
        if (Math.abs(value.data[index]) < normal) value.data[index] = 0;
      }
    }
    return value;
  });
  try {
    await harness.run([node]);
    // Exact separation squared is (1/4+1)*N² = 5N²/4 > (17N/16)² = 289N²/256.
    expect(await harness.read(result)).toEqual([0]);
  } finally {
    patch.mockRestore();
    harness.destroy();
  }
});
