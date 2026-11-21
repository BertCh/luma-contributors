// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUSpatialJoinPrepared,
  getGPUSpatialJoinPreparedMemoryPlan
} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared';
import {GPUSpatialPredicateJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import {decodeGPUSpatialJoinGeometry} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-join-geometry';
import {
  getGPUSpatialPredicateJoinExecutionPlan,
  GPU_SPATIAL_PREDICATE_JOIN_ROUTES
} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-predicate-route-plan';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {Buffer} from '@luma.gl/core';
import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window/gpu-study-window-preparation';
import {createSpatialPreparationRetryViews} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-preparation-retry';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {addSpatialPredicateUnmatchedNodes} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-predicate-unmatched';

it('B5 plans allocated polygon offset levels and fixed generation storage before declaration', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const shape = {
    kind: 'polygons' as const,
    featureCount: 1,
    vertexCount: 3,
    polygonCount: 1,
    ringCount: 40000000
  };
  const plan = getGPUSpatialJoinPreparedMemoryPlan(shape, device.limits);
  const prepared = new GPUSpatialJoinPrepared({
    geometry: {
      kind: 'polygons',
      positions: createTransientView(graph, 'positions', 'float32x2', 3),
      featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
      polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
      ringOffsets: createTransientView(graph, 'rings', 'uint32', 40000001)
    }
  });
  try {
    expect(() => prepared.getCommandNodes(graph)).toThrow(/160000004/);
    expect(plan.feasible).toBe(false);
    expect(plan.largestBindingBytes).toBe(160000004); // (40,000,000 allocated rings + terminal) * 4.
    const small = getGPUSpatialJoinPreparedMemoryPlan(
      {kind: 'points', featureCount: 1, vertexCount: 1},
      {...device.limits, maxStorageBufferBindingSize: 512}
    );
    expect(small.largestBindingBytes).toBe(1024); // Immutable table of all 256 uint32 byte values.
    expect(small.feasible).toBe(false);
  } finally {
    prepared.destroy();
    device.destroy();
  }
});

it('B6 planner and constructor both reject metric relate output', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const scalar = () => createTransientView(graph, `scalar-${serial++}`, 'uint32', 1);
  const shape = {
    predicate: 'dwithin' as const,
    outputRelate: true,
    leftKind: 'points' as const,
    rightKind: 'points' as const,
    leftCount: 1,
    rightCount: 1,
    leftVertexCount: 1,
    rightVertexCount: 1
  };
  try {
    expect(
      () =>
        new GPUSpatialPredicateJoin({
          left: {kind: 'points', positions: createTransientView(graph, 'left', 'float32x2', 1)},
          right: {kind: 'points', positions: createTransientView(graph, 'right', 'float32x2', 1)},
          predicate: 'dwithin',
          distance: 1,
          output: {
            relate: scalar(),
            pairs: {
              leftRows: scalar(),
              rightRows: scalar(),
              status: {count: scalar(), overflow: scalar()}
            }
          }
        })
    ).toThrow(/relate.*dwithin/);
    expect(() => getGPUSpatialPredicateJoinExecutionPlan(shape)).toThrow(/relate.*dwithin/);
  } finally {
    device.destroy();
  }
});

it('B5 includes allocated multipart line parts in planner/declaration agreement', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const prepared = new GPUSpatialJoinPrepared({
    geometry: {
      kind: 'lines',
      positions: createTransientView(graph, 'multipart-positions', 'float32x2', 3),
      featureOffsets: createTransientView(graph, 'multipart-features', 'uint32', 2),
      lineOffsets: createTransientView(graph, 'multipart-lines', 'uint32', 40000001)
    }
  });
  try {
    const plan = getGPUSpatialJoinPreparedMemoryPlan(
      {kind: 'lines', featureCount: 1, vertexCount: 3, multipart: true, partCount: 40000000},
      device.limits
    );
    expect(plan.feasible).toBe(false);
    expect(plan.largestBindingBytes).toBe(160000004);
    expect(() => prepared.getCommandNodes(graph)).toThrow(/160000004/);
  } finally {
    prepared.destroy();
    device.destroy();
  }
});

it('S3 refuses unknown decoded geometry discriminants', () => {
  expect(() =>
    decodeGPUSpatialJoinGeometry({kind: 'bad' as 'points', positions: Float32Array.of(1, 2)}, 1)
  ).toThrow(/points.*lines.*polygons/);
});

it('S4 route metadata describes cooperative uncertainty tracking', () => {
  expect(GPU_SPATIAL_PREDICATE_JOIN_ROUTES[0].accuracyImplications).not.toMatch(
    /does not track uncertainCount/
  );
});

it('B7 retry reuses the fresh graph producer handle and observes its writes', () => {
  const device = createNullWebGPUDevice();
  const buffer = device.createBuffer({byteLength: 8, usage: Buffer.STORAGE | Buffer.COPY_DST});
  const original = new GPUCommandGraph(device),
    fresh = new GPUCommandGraph(device);
  const oldPositions = importGraphBuffer(original, 'old-positions', buffer, 'float32x2', 1);
  const canonical = importGraphBuffer(fresh, 'produced-positions', buffer, 'float32x2', 1);
  const retried = createSpatialPreparationRetryViews(fresh, 'retry', {positions: oldPositions});
  const result = createTransientView(fresh, 'observed', 'float32', 1);
  const writer = createWGSLKernelNode(fresh, {
    id: 'producer',
    operation: 'RetryProbe',
    invocationCount: 1,
    bindings: [{name: 'positions', view: canonical, type: 'f32', access: 'read_write'}],
    body: 'positions[positionsOffset] = 5.0;'
  });
  const reader = createWGSLKernelNode(fresh, {
    id: 'consumer',
    operation: 'RetryProbe',
    invocationCount: 1,
    bindings: [
      {name: 'positions', view: retried.positions, type: 'f32', access: 'read'},
      {name: 'result', view: result, type: 'f32', access: 'read_write'}
    ],
    body: 'result[resultOffset] = positions[positionsOffset];'
  });
  try {
    const memory = createCPUGraphMemory();
    executeCommandNodesOnCPU([writer, reader], memory);
    expect(Array.from(memory.readFloat32(result))).toEqual([5]);
    expect(retried.positions.buffer).toBe(canonical.buffer);
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

for (const kind of ['join', 'window'])
  it(`B8 ${kind} exposes recovery after graph node registration fails`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const buffers = [32, 8, 8, 8].map(byteLength =>
      device.createBuffer({byteLength, usage: Buffer.STORAGE | Buffer.COPY_DST})
    );
    const polygon = {
      positions: importGraphBuffer(graph, 'positions', buffers[0], 'float32x2', 4),
      featureOffsets: importGraphBuffer(graph, 'features', buffers[1], 'uint32', 2),
      polygonOffsets: importGraphBuffer(graph, 'polygons', buffers[2], 'uint32', 2),
      ringOffsets: importGraphBuffer(graph, 'rings', buffers[3], 'uint32', 2)
    };
    const prepared =
      kind === 'join'
        ? new GPUSpatialJoinPrepared({id: 'collision', geometry: {kind: 'polygons', ...polygon}})
        : new GPUStudyWindowPreparation({id: 'collision', ...polygon, samplingGrid: [1, 1]});
    const marker = createTransientView(graph, 'marker', 'uint32', 1);
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'collision-snapshot-request',
        operation: 'CollisionProbe',
        invocationCount: 1,
        bindings: [{name: 'marker', view: marker, type: 'u32', access: 'read_write'}],
        body: 'marker[markerOffset] = 0u;'
      })
    );
    try {
      expect(() => graph.add(prepared)).toThrow(/collision-snapshot-request/);
      expect(typeof prepared.recoverFailedDeclaration).toBe('function');
      prepared.recoverFailedDeclaration();
      expect(prepared.isDeclaredIn(graph)).toBe(false);
      const fresh = new GPUCommandGraph(device);
      expect(() => fresh.add(prepared)).not.toThrow();
      expect(prepared.isDeclaredIn(fresh)).toBe(true);
    } finally {
      prepared.destroy();
      for (const buffer of buffers) buffer.destroy();
      device.destroy();
    }
  });

it('S2 does not publish definite absence from an uncertified false classification', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const state = createTransientView(graph, 'state', 'uint32', 6);
  const counts = createTransientView(graph, 'matches', 'uint32', 1);
  const nodes: GPUCommandNode[] = [];
  addSpatialPredicateUnmatchedNodes(graph, nodes, {
    id: 'uncertain',
    leftCount: 1,
    candidateCapacity: 1,
    finishOnly: true,
    candidatePairs: createTransientView(graph, 'candidates', 'uint32x2', 1),
    flags: createTransientView(graph, 'flags', 'uint32', 1),
    leftMatchCounts: counts,
    state,
    unmatched: {
      ids: createTransientView(graph, 'ids', 'uint32', 1),
      status: {
        count: createTransientView(graph, 'count', 'uint32', 1),
        overflow: createTransientView(graph, 'overflow', 'uint32', 1)
      }
    }
  });
  const classify = nodes.find(node => node.id === 'uncertain-unmatched-flags')!;
  const output = classify.resources!.find(
    resource => 'buffer' in resource && resource.usage !== 'storage-read'
  )!;
  if (!('buffer' in output)) throw new Error('missing unmatched flags');
  const memory = createCPUGraphMemory();
  try {
    memory.write(state, Uint32Array.of(0, 1, 0, 0, 0, 0));
    executeCommandNodesOnCPU([classify], memory);
    expect(Array.from(memory.readUint32(output.buffer as GraphDataView))).toEqual([0]);
    memory.write(state, new Uint32Array(6));
    executeCommandNodesOnCPU([classify], memory);
    expect(Array.from(memory.readUint32(output.buffer as GraphDataView))).toEqual([1]);
  } finally {
    device.destroy();
  }
});

function createLiveDiagnosticFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const scalar = () => createTransientView(graph, `live-scalar-${serial++}`, 'uint32', 1);
  const left = {
    kind: 'polygons' as const,
    positions: createTransientView(graph, 'left-positions', 'float32x2', 12),
    featureOffsets: createTransientView(graph, 'features', 'uint32', 4),
    polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 4),
    ringOffsets: createTransientView(graph, 'rings', 'uint32', 4)
  };
  const live = scalar(),
    truncated = createTransientView(graph, 'truncated', 'uint32', 3),
    geographic = scalar();
  const join = new GPUSpatialPredicateJoin({
    id: 'live-diagnostics',
    left,
    leftFeatureCount: live,
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'},
    right: {
      kind: 'points',
      positions: createTransientView(graph, 'right-positions', 'float32x2', 1)
    },
    predicate: 'intersects',
    output: {
      candidateTruncated: truncated,
      pairs: {
        leftRows: scalar(),
        rightRows: scalar(),
        status: {count: scalar(), overflow: scalar()}
      },
      status: {candidateTruncatedCount: scalar(), antimeridianUncertainCount: geographic}
    }
  });
  const nodes = join.getCommandNodes(graph);
  const memory = createCPUGraphMemory();
  memory.write(
    left.positions,
    Float32Array.of(
      0,
      0,
      1,
      0,
      1,
      1,
      0,
      1,
      179,
      0,
      -179,
      0,
      -179,
      1,
      179,
      1,
      10,
      0,
      11,
      0,
      11,
      1,
      10,
      1
    )
  );
  memory.write(left.featureOffsets, Uint32Array.of(0, 1, 2, 3));
  memory.write(left.polygonOffsets, Uint32Array.of(0, 1, 2, 3));
  memory.write(left.ringOffsets, Uint32Array.of(0, 4, 8, 12));
  return {device, nodes, memory, live, truncated, geographic};
}

function resourceView(node: GPUCommandNode, index: number): GraphDataView {
  const resource = node.resources![index];
  if (!('buffer' in resource)) throw new Error('expected buffer resource');
  return resource.buffer as GraphDataView;
}

it('B4 shrinks per-left loss flags and their reduction to live rows', () => {
  const {device, nodes, memory, live, truncated} = createLiveDiagnosticFixture();
  const classify = nodes.find(node => node.id === 'live-diagnostics-candidate-truncated')!;
  const reduce = nodes.find(node => node.id === 'live-diagnostics-count-truncated')!;
  const selected = nodes.filter(node =>
    [classify.id, 'live-diagnostics-live-truncation', reduce.id].includes(node.id)
  );
  const state = resourceView(reduce, 1);
  try {
    memory.write(resourceView(classify, 2), Uint32Array.of(1)); // Global BVH/input loss.
    memory.write(live, Uint32Array.of(3));
    executeCommandNodesOnCPU(selected, memory);
    expect(Array.from(memory.readUint32(truncated))).toEqual([1, 1, 1]);
    memory.write(state, new Uint32Array(6));
    memory.write(live, Uint32Array.of(1));
    executeCommandNodesOnCPU(selected, memory);
    expect(Array.from(memory.readUint32(truncated))).toEqual([1, 0, 0]);
    expect(memory.readUint32(state)[4]).toBe(1);
  } finally {
    device.destroy();
  }
});

it('B4 ignores a stale crossing ring beyond the terminal live polygon offset', () => {
  const {device, nodes, memory, live, geographic} = createLiveDiagnosticFixture();
  const selected = nodes.filter(
    node =>
      node.id === 'live-diagnostics-clear-antimeridianUncertainCount' ||
      node.id === 'live-diagnostics-live-ring-offsets' ||
      node.id === 'live-diagnostics-lon-lat-polygon-check-0'
  );
  try {
    memory.write(live, Uint32Array.of(3));
    executeCommandNodesOnCPU(selected, memory);
    expect(Array.from(memory.readUint32(geographic))).toEqual([1]);
    memory.write(live, Uint32Array.of(1));
    executeCommandNodesOnCPU(selected, memory);
    expect(Array.from(memory.readUint32(geographic))).toEqual([0]);
  } finally {
    device.destroy();
  }
});
