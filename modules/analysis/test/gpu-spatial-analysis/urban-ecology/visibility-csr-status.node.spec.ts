// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, expectTypeOf, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  addVisibilityGraphRecipe,
  decodeGPUVisibilityGraphAdjacencyCSRStatus,
  getGPUSpatialWeightsCapacityPlan,
  VISIBILITY_GRAPH_CSR_MAXIMUM_EDGE_CAPACITY
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('STA-09 decodes unknown candidate coverage before sizing directed slots', () => {
  const decode = decodeGPUVisibilityGraphAdjacencyCSRStatus;
  expect(decode(Uint32Array.of(0, 0xffffffff, 0, 1, 0)).requirement).toEqual({kind: 'unknown'});
  expect(decode(Uint32Array.of(0, 4, 1, 0, 0)).requirement).toEqual({kind: 'exact', value: 4});
  expect(decode(Uint32Array.of(4, 4, 0, 0, 0)).requirement).toEqual({kind: 'exact', value: 4});
  expect(decode(Uint32Array.of(0, 0xffffffff, 1, 0, 0)).requirement).toEqual({
    kind: 'saturated',
    value: 0xffffffff
  });
  const bytes = new Uint8Array(24);
  bytes.set(new Uint8Array(Uint32Array.of(4, 4, 0, 0, 0).buffer), 4);
  expect(decode(bytes.subarray(4)).count).toBe(4);
  expect(decode(Uint32Array.of(4, 4, 0, 0, 0).buffer).requiredCount).toBe(4);
  expect(() => decode(new Uint8Array(19))).toThrow();
  expect(() => decode(new Uint32Array(4))).toThrow();
});

it.each([
  [4097, 16781312, 134217728, /edgeCapacity/],
  [33, 1056, 2048, /CSR neighbors and weights/]
] as const)('STA-09 exact requirement for %s points still obeys route/device caps', (pointCount, requiredCount, bindingLimit, message) => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    // A complete symmetric graph has N*(N-1) directed slots, always an even number.
    // Possible producer evidence: points (0,0)..(N-1,0), radius N, obstacles outside the hull.
    // These host declarations check capacity rejection, without running the visibility query.
    expect(requiredCount).toBe(pointCount * (pointCount - 1));
    const positions = createTransientView(graph, 'cap-positions', 'float32x2', pointCount);
    const segments = createTransientView(graph, 'cap-segments', 'float32x4', 1);
    const plan = getGPUSpatialWeightsCapacityPlan({
      rowCount: pointCount,
      capacity: 2,
      headroom: 0,
      requiredCount: decodeGPUVisibilityGraphAdjacencyCSRStatus(
        Uint32Array.of(0, requiredCount, 1, 0, 0)
      ).requirement
    });
    expect(plan.retryStage).toBe('final');
    expect(plan.capacity).toBe(requiredCount);
    Object.assign(device.limits, {maxStorageBufferBindingSize: bindingLimit});
    expect(() =>
      addVisibilityGraphRecipe(graph, {
        mode: 'csr',
        positions,
        segments,
        maximumDistance: pointCount,
        edgeCapacity: plan.capacity
      })
    ).toThrow(message);
    expect((graph as unknown as {nodes: unknown[]}).nodes).toHaveLength(0);
  } finally {
    device.destroy();
  }
});

it('B3 preserves the decoder exact three-kind public requirement type', () => {
  expectTypeOf<
    ReturnType<typeof decodeGPUVisibilityGraphAdjacencyCSRStatus>['requirement']['kind']
  >().toEqualTypeOf<'unknown' | 'exact' | 'saturated'>();
});

it('STA-09 sends the public decoded requirement to the shared capacity planner', () => {
  // Three collinear points at x=0,1,2 and radius 1.1 have two undirected pairs,
  // hence four directed slots. Candidate loss cannot tell us this requirement yet.
  const plan = (values: Uint32Array) =>
    getGPUSpatialWeightsCapacityPlan({
      rowCount: 3,
      capacity: 2,
      headroom: 0,
      requiredCount: decodeGPUVisibilityGraphAdjacencyCSRStatus(values).requirement
    });
  const unknown = plan(Uint32Array.of(0, 0xffffffff, 0, 1, 0));
  expect(unknown.requirement).toEqual({kind: 'unknown'});
  expect(unknown.retryStage).toBe('candidates');
  expect(unknown.capacity).toBe(2);
  const saturated = plan(Uint32Array.of(0, 0xffffffff, 1, 0, 0));
  expect(saturated.requirement).toEqual({kind: 'saturated', value: 0xffffffff});
  expect(saturated.retryStage).toBe('candidates');
  expect(saturated.capacity).toBe(2);
  const exact = plan(Uint32Array.of(0, 4, 1, 0, 0));
  expect(exact.requirement).toEqual({kind: 'exact', value: 4});
  expect(exact.retryStage).toBe('final');
  expect(exact.capacity).toBe(4);
  // Two uint32/float32 slot arrays plus four uint32 offsets: 4*8 + 4*4 = 48.
  expect(exact.estimatedOutputBytes).toBe(48);
  const complete = getGPUSpatialWeightsCapacityPlan({
    rowCount: 3,
    capacity: 4,
    requiredCount: decodeGPUVisibilityGraphAdjacencyCSRStatus(Uint32Array.of(4, 4, 0, 0, 0))
      .requirement
  });
  expect(complete.retryStage).toBe('none');
  expect(complete.capacity).toBe(4);
  // An exact full graph of 4097 points exceeds this route even with no headroom.
  const tooLarge = getGPUSpatialWeightsCapacityPlan({
    rowCount: 4097,
    capacity: 2,
    headroom: 0,
    requiredCount: decodeGPUVisibilityGraphAdjacencyCSRStatus(Uint32Array.of(0, 16781312, 1, 0, 0))
      .requirement
  });
  expect(tooLarge.capacity).toBe(16781312); // 4097 * 4096, an independent pair count.
  expect(tooLarge.capacity).toBeGreaterThan(VISIBILITY_GRAPH_CSR_MAXIMUM_EDGE_CAPACITY);
});
