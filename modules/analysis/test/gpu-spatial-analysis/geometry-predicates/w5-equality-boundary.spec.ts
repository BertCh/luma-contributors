// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GEOMETRY_EQUALITY_DISTANCE_WGSL} from '../../../src/gpu-spatial-analysis/geometry-predicates/geometry-equality-distance-wgsl';
import {GPUGeometryPredicates} from '../../../src/gpu-spatial-analysis/geometry-predicates/gpu-geometry-predicates';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createEqualityMetricFixture,
  type EqualityView
} from '../geometry-refinement/canonical-equality-fixture';
import {createWaveHarness} from '../geometry-validity/w5-test-harness';
it.each([
  'exact',
  'tree',
  'polygon'
] as const)('GPU %s equality certifies 13,84 at 85 and the preceding float', async kind => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [tolerance, expected] of [
    [85, 1],
    [84.99999237060547, 0]
  ] as const) {
    const harness = createWaveHarness(device);
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
      expect(await harness.read(fixture.output)).toEqual([expected]);
    } finally {
      harness.destroy();
    }
  }
});
it('GPU separate predicate passes publish adjacent equals and touches words', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createWaveHarness(device);
  const geometry = {
    kind: 'points' as const,
    positions: harness.allocate('point', 'float32x2', 1, [1, 2])
  };
  const packed = harness.allocate('packed', 'uint32', 2);
  const equals = harness.graph.createDataView(packed.buffer, {format: 'uint32', length: 1});
  const touches = harness.graph.createDataView(packed.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 4
  });
  const nodes = new GPUGeometryPredicates({
    geometry,
    other: geometry,
    output: {topological: {equals, touches}}
  }).getCommandNodes(harness.graph);
  try {
    await harness.run(nodes);
    expect(await harness.read(packed)).toEqual([1, 0]);
  } finally {
    harness.destroy();
  }
});

// Review B6: all operands are normal (N = 2^-126) but the x difference 0.5N is subnormal and may
// flush to zero on the device. Exact squared separation (1/4+1)N² = 5N²/4 exceeds the squared
// tolerance (17N/16)² = 289N²/256, so the pair must not compare equal.
it('GPU Euclidean certificate rejects a pair whose smaller delta is subnormal', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createWaveHarness(device);
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
  try {
    await harness.run([node]);
    expect(await harness.read(result)).toEqual([0]);
  } finally {
    harness.destroy();
  }
});

// Review B4: these output pairs are written by separate dispatches, so adjacent words in one
// buffer are valid WebGPU bindings. Point simplicity writes 1 and point/empty status writes 0.
it.each([
  'point-simple-status',
  'point-simple-ring',
  'empty-line-status'
] as const)('GPU separately dispatched adjacent outputs publish on %s', async route => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createWaveHarness(device);
  const packed = harness.allocate('packed', 'uint32', 2);
  const first = harness.graph.createDataView(packed.buffer, {format: 'uint32', length: 1});
  const second = harness.graph.createDataView(packed.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 4
  });
  const geometry =
    route === 'empty-line-status'
      ? {
          kind: 'lines' as const,
          // A zero-length view over a one-row buffer keeps a valid aligned binding.
          positions: harness.graph.createDataView(
            harness.allocate('positions', 'float32x2', 1, [0, 0]).buffer,
            {format: 'float32x2', length: 0}
          ),
          lineOffsets: harness.allocate('parts', 'uint32', 2, [0, 0])
        }
      : {kind: 'points' as const, positions: harness.allocate('positions', 'float32x2', 1, [1, 2])};
  const output =
    route === 'point-simple-ring'
      ? {
          isSimple: first,
          isRing: second,
          status: {overflow: harness.allocate('overflow', 'uint32', 1)}
        }
      : route === 'point-simple-status'
        ? {isSimple: first, status: {overflow: second}}
        : {status: {overflow: first, intersectionCount: second}};
  try {
    const nodes = new GPUGeometryPredicates({
      geometry,
      intersectionCapacity: 1,
      output
    }).getCommandNodes(harness.graph);
    await harness.run(nodes);
    expect(await harness.read(packed)).toEqual(route === 'empty-line-status' ? [0, 0] : [1, 0]);
  } finally {
    harness.destroy();
  }
});
