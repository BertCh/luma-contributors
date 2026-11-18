// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUBVH,
  GPUCommandGraph,
  GraphDataView,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialJoinCandidates} from '../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-candidates';
import {GPUSegmentIntersection} from '../../src/gpu-spatial-analysis/segment-intersection/gpu-segment-intersection';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../gpu-spatial-analysis/segment-intersection/cpu-executor-test-hooks';

// Keep the published shader path before the hooks replace cooperative primitives for other cases.
const getPublishedBVHNodes = GPUBVH.prototype.getCommandNodes<void>;
useCPUExecutorTestHooks();

it.each([
  2, 4
])('published BVH counts all %s source slots after invalidation and clears overflow after resizing', sourceCount => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const minima = createTransientView(graph, 'probe-minima', 'float32x2', sourceCount);
  const maxima = createTransientView(graph, 'probe-maxima', 'float32x2', sourceCount);
  const count = createTransientView(graph, 'probe-count', 'uint32', 1);
  const overflow = createTransientView(graph, 'probe-overflow', 'uint32', 1);
  const build = (leafCapacity: number) => {
    const nodeCount = 2 * leafCapacity - 1;
    const bvh = new GPUBVH({
      id: `probe-${leafCapacity}`,
      minima,
      maxima,
      leafCapacity,
      // Separate level dispatches have no cooperative barriers, so the interpreter executes
      // the actual published WGSL instead of the BVH stand-in used by the other tests.
      strategy: 'level',
      nodeMinima: createTransientView(
        graph,
        `probe-${leafCapacity}-minima`,
        'float32x2',
        nodeCount
      ),
      nodeMaxima: createTransientView(
        graph,
        `probe-${leafCapacity}-maxima`,
        'float32x2',
        nodeCount
      ),
      nodeChildren: createTransientView(
        graph,
        `probe-${leafCapacity}-children`,
        'uint32x2',
        nodeCount
      ),
      leafIds: createTransientView(graph, `probe-${leafCapacity}-leaves`, 'uint32', leafCapacity),
      count,
      overflow
    });
    return getPublishedBVHNodes.call(bvh, graph);
  };
  const nodes = build(1);
  const memory = createCPUGraphMemory();
  try {
    memory.write(minima, new Float32Array(sourceCount * 2));
    memory.write(maxima, new Float32Array(sourceCount * 2).fill(1));
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(overflow))).toEqual([1]);
    const invalidMinima = new Float32Array(sourceCount * 2).fill(1);
    const invalidMaxima = new Float32Array(sourceCount * 2);
    invalidMinima.set([0, 0]);
    invalidMaxima.set([1, 1]);
    memory.write(minima, invalidMinima);
    memory.write(maxima, invalidMaxima);
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(count))).toEqual([sourceCount]);
    expect(Array.from(memory.readUint32(overflow))).toEqual([1]);
    executeCommandNodesOnCPU(build(4), memory);
    expect(Array.from(memory.readUint32(count))).toEqual([sourceCount]);
    expect(Array.from(memory.readUint32(overflow))).toEqual([0]);
  } finally {
    device.destroy();
  }
});

function getBinding(nodes: readonly GPUCommandNode[], name: string): GraphDataView {
  for (const node of nodes) {
    const source = getGPUCommandNodeKernelSources([node])[0]?.source;
    const match = source?.match(
      new RegExp('@binding\\((\\d+)\\) var<storage, [^>]+> ' + name + ':')
    );
    const resource = match && node.resources?.[Number(match[1])];
    if (resource && 'buffer' in resource && resource.buffer instanceof GraphDataView)
      return resource.buffer;
  }
  throw new Error('Missing binding ' + name);
}

for (const kind of ['candidates', 'segments'] as const) {
  for (const mutate of [false, true]) {
    it(`final1 ${kind} rewrites independent pair and BVH loss causes on reuse (mutated map: ${mutate})`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
      const pairs = {
        leftRows: createTransientView(graph, 'left-ids', 'uint32', 4),
        rightRows: createTransientView(graph, 'right-ids', 'uint32', 4),
        status: {
          count: scalar('count'),
          overflow: scalar('overflow'),
          candidateOverflow: scalar('candidate-overflow'),
          requiredCount: scalar('required-count')
        }
      };
      const positions = createTransientView(graph, 'positions', 'float32x2', 2);
      const lineOffsets = createTransientView(graph, 'line-offsets', 'uint32', 2);
      const withoutCandidateStatus = {
        ...pairs,
        status: {count: pairs.status.count, overflow: pairs.status.overflow}
      };
      expect(() =>
        kind === 'candidates'
          ? new GPUSpatialJoinCandidates({
              left: {kind: 'points', positions},
              right: {kind: 'points', positions},
              leafCapacity: 1,
              output: withoutCandidateStatus
            })
          : new GPUSegmentIntersection({
              left: {kind: 'lines', positions, lineOffsets},
              leafCapacity: 1,
              output: {pairs: withoutCandidateStatus}
            })
      ).toThrow(/candidateOverflow.*leafCapacity/);
      const producer =
        kind === 'candidates'
          ? new GPUSpatialJoinCandidates({
              left: {kind: 'points', positions},
              right: {kind: 'points', positions},
              leafCapacity: 1,
              output: pairs
            })
          : new GPUSegmentIntersection({
              left: {kind: 'lines', positions, lineOffsets},
              leafCapacity: 1,
              output: {pairs}
            });
      const candidateDestination = pairs.status.candidateOverflow;
      if (mutate) Reflect.deleteProperty(pairs.status, 'candidateOverflow');
      try {
        const allNodes = producer.getCommandNodes(graph);
        const nodes = allNodes.filter(node =>
          /-(write-pairs|candidate-status|finalize)$/.test(node.id)
        );
        const state = getBinding(nodes, 'state');
        const bvhResource = allNodes
          .flatMap(node => node.resources ?? [])
          .find(
            resource =>
              'buffer' in resource &&
              resource.buffer instanceof GraphDataView &&
              resource.buffer.buffer.id.endsWith('-bvh-overflow')
          );
        if (
          !bvhResource ||
          !('buffer' in bvhResource) ||
          !(bvhResource.buffer instanceof GraphDataView)
        )
          throw new Error('Missing BVH overflow');
        const bvhOverflow = bvhResource.buffer;
        const memory = createCPUGraphMemory();
        for (const [total, lostLeaves, expected] of [
          [1, 1, [0, 1]],
          [5, 0, [1, 0]],
          [1, 0, [0, 0]]
        ] as const) {
          memory.write(state, Uint32Array.of(total, 0, 0, 0));
          memory.write(bvhOverflow, Uint32Array.of(lostLeaves));
          memory.write(pairs.status.overflow, Uint32Array.of(29));
          memory.write(candidateDestination, Uint32Array.of(37));
          executeCommandNodesOnCPU(nodes, memory);
          expect([
            memory.readUint32(pairs.status.overflow)[0],
            memory.readUint32(candidateDestination)[0]
          ]).toEqual(expected);
          expect(memory.readUint32(pairs.status.count)[0]).toBe(Math.min(total, 4));
        }
      } finally {
        device.destroy();
      }
    });
  }
}
