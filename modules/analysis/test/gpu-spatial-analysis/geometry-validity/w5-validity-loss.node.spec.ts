// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUGeometryValidity} from '../../../src/gpu-spatial-analysis/geometry-validity/gpu-geometry-validity';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
useCPUExecutorTestHooks();
it.each([
  4, 8
])('validity aggregates candidate loss for six vertices and %s leaves on replay', leafCapacity => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const positions = createTransientView(graph, 'positions', 'float32x2', 6);
  const words = (name: string, values: number[]) => {
    const view = createTransientView(graph, name, 'uint32', values.length);
    memory.write(view, Uint32Array.from(values));
    return view;
  };
  const overflow = words('overflow', [99]);
  const validity = words('validity', [99]);
  const issueCode = words('issues', [0]);
  const nodes = new GPUGeometryValidity({
    polygons: {
      kind: 'polygons',
      positions,
      featureOffsets: words('features', [0, 1]),
      polygonOffsets: words('polygons', [0, 1]),
      ringOffsets: words('rings', [0, 6])
    },
    leafCapacity,
    intersectionCapacity: 32,
    output: {issueCode, validity, status: {overflow}}
  }).getCommandNodes(graph);
  // Convex hexagon: no nonadjacent intersections. Only the restricted tree loses candidates.
  memory.write(positions, Float32Array.of(0, 0, 2, 0, 3, 1, 2, 2, 0, 2, -1, 1));
  try {
    for (let replay = 0; replay < 2; replay++) {
      memory.write(overflow, Uint32Array.of(99));
      executeCommandNodesOnCPU(
        nodes.filter(
          node =>
            node.id.startsWith('geometry-validity-intersection') || node.id.endsWith('-validity')
        ),
        memory
      );
      expect(Array.from(memory.readUint32(overflow))).toEqual([leafCapacity === 4 ? 1 : 0]);
      expect(Array.from(memory.readUint32(validity))).toEqual([leafCapacity === 4 ? 0 : 1]);
    }
  } finally {
    device.destroy();
  }
});

it.each([
  'polygons',
  'lines',
  'points'
] as const)('review B7 validates only co-bound scalar ranges for %s', kind => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const words = (name: string, length: number) =>
    createTransientView(graph, name, 'uint32', length);
  const packed = words('packed', 2);
  const overflow = graph.createDataView(packed.buffer, {format: 'uint32', length: 1});
  const intersectionCount = graph.createDataView(packed.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 4
  });
  const positions = createTransientView(
    graph,
    'positions',
    'float32x2',
    kind === 'polygons' ? 5 : 2
  );
  const geometry =
    kind === 'polygons'
      ? {
          polygons: {
            kind,
            positions,
            featureOffsets: words('features', 2),
            polygonOffsets: words('polygons', 2),
            ringOffsets: words('rings', 2)
          }
        }
      : kind === 'lines'
        ? {lines: {kind, positions, lineOffsets: words('parts', 2)}}
        : {points: {kind, positions}};
  try {
    const construct = () =>
      new GPUGeometryValidity({
        ...geometry,
        leafCapacity: 4,
        intersectionCapacity: 16,
        output: {
          issueCode: words('issues', kind === 'points' ? 2 : 1),
          status: {overflow, intersectionCount}
        }
      });
    if (kind === 'polygons')
      expect(construct).toThrow(/overflow.*intersectionCount.*aligned binding/);
    else expect(construct).not.toThrow();
  } finally {
    device.destroy();
  }
});
