// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {createCPUGraphMemory, executeCommandNodesOnCPU} from './wgsl-graph-executor';
import {useCPUExecutorTestHooks} from './cpu-executor-test-hooks';
import {
  createCapacityBoundaryFixture,
  createStatusOnlyFixture,
  createWrapperLossFixture,
  LOSS_WRAPPERS,
  type WrapperView
} from './wrapper-loss-fixture';

useCPUExecutorTestHooks();
function assertParallelFinalizerSources(kernels: readonly {source: string}[]) {
  expect(kernels.length).toBeGreaterThan(0);
  for (const kernel of kernels)
    expect(kernel.source).not.toMatch(/\b(?:while|for)\s*\(|\bloop\s*\{/);
}
it.each([
  '',
  'for(var item=0u;item<2u;item++) {}',
  'loop {break;}'
])('finalizer oracle rejects missing or serial source: %s', source => {
  expect(() => assertParallelFinalizerSources(source ? [{source}] : [])).toThrow();
});

it.each([
  'split',
  'clip',
  'shared'
] as const)('%s capacity publication retains only whole prefixes, including equal endpoints', kind => {
  const {device, graph, memory, view} = createHarness();
  const views: GraphDataView[] = [];
  const createDataView = graph.createDataView.bind(graph);
  const observe = vi.spyOn(graph, 'createDataView').mockImplementation((...parameters) => {
    const result = createDataView(...parameters);
    views.push(result);
    return result;
  });
  const fixture = createWrapperLossFixture(kind, view);
  const nodes = fixture.contributor.getCommandNodes(graph);
  observe.mockRestore();
  const prefix =
    kind === 'split' ? 'line-split' : kind === 'clip' ? 'line-clip-by-polygon' : 'shared-paths';
  const scratch = (suffix: string) => {
    const result = views.find(candidate => candidate.buffer.id === `${prefix}-${suffix}`);
    if (!result) throw new Error(`missing ${prefix}-${suffix}`);
    return result;
  };
  const write = (suffix: string, values: number[]) =>
    memory.write(scratch(suffix), Uint32Array.from(values));
  try {
    for (const ends of [
      [2, 5, 257, 300],
      [2, 5, 5, 300]
    ]) {
      const starts = [0, ...ends.slice(0, -1)];
      if (kind === 'split') {
        write('piece-starts', [0]);
        write('piece-counts', [4]);
        write('piece-vertex-starts', starts);
        write(
          'piece-vertex-counts',
          ends.map((end, index) => end - starts[index])
        );
      } else {
        const flags = new Uint32Array(scratch('run-flags').length);
        const ranks = new Uint32Array(scratch('run-ranks').length);
        flags[flags.length - 1] = 1;
        ranks[ranks.length - 1] = 3;
        memory.write(scratch('run-flags'), flags);
        memory.write(scratch('run-ranks'), ranks);
        const table = new Uint32Array(scratch('run-table').length);
        starts.forEach((start, index) => {
          table[index * (kind === 'clip' ? 2 : 5)] = start;
        });
        memory.write(scratch('run-table'), table);
        if (kind === 'shared') write('vertex-total', [300]);
        else {
          const vertexStarts = new Uint32Array(scratch('vertex-starts').length);
          vertexStarts[vertexStarts.length - 1] = 300;
          memory.write(scratch('vertex-starts'), vertexStarts);
          memory.write(scratch('vertex-counts'), new Uint32Array(scratch('vertex-counts').length));
        }
      }
      executeCommandNodesOnCPU(
        nodes.filter(
          node =>
            node.id.startsWith(`${prefix}-capacity-prefix`) || node.id === `${prefix}-finalize`
        ),
        memory
      );
      // Capacity is 256 vertices. Ends 2,5 fit; 257 does not. An equal third end 5 fits too.
      expect(Array.from(memory.readUint32(scratch('state')).slice(0, 5))).toEqual([
        ends[2] === 5 ? 3 : 2,
        5,
        4,
        300,
        1
      ]);
    }
  } finally {
    device.destroy();
  }
});

it.each([
  'split',
  'clip',
  'shared'
] as const)('%s finalizer has no serial capacity search', kind => {
  const {device, graph, view} = createHarness();
  try {
    const fixture = createWrapperLossFixture(kind, view);
    const finalizer = fixture.contributor
      .getCommandNodes(graph)
      .filter(
        node =>
          node.id ===
          `${kind === 'split' ? 'line-split' : kind === 'clip' ? 'line-clip-by-polygon' : 'shared-paths'}-finalize`
      );
    expect(finalizer).toHaveLength(1);
    assertParallelFinalizerSources(getGPUCommandNodeKernelSources(finalizer));
  } finally {
    device.destroy();
  }
});

function createHarness() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  let serial = 0;
  const view: WrapperView = (format, length, values) => {
    const result = createTransientView(graph, `wrapper-${serial++}`, format, length);
    const data = values ?? new Array(length * (format === 'float32x2' ? 2 : 1)).fill(99);
    memory.write(result, format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data));
    return result;
  };
  return {device, graph, memory, view};
}

it.each(LOSS_WRAPPERS)('%s forwards nested BVH loss with spare final capacity', kind => {
  const {device, graph, memory, view} = createHarness();
  try {
    const fixture = createWrapperLossFixture(kind, view);

    // The interpreter cannot bind the containment probe's chunkSettings uniform. This check
    // exercises pair loss and publication; the paired GPU spec executes containment as well.
    const nodes = fixture.contributor
      .getCommandNodes(graph)
      .filter(node => kind !== 'clip' || !node.id.includes('-containment-'));
    for (let submission = 0; submission < 2; submission++) {
      memory.write(fixture.overflow, Uint32Array.of(99));
      memory.write(fixture.candidateOverflow, Uint32Array.of(99));
      executeCommandNodesOnCPU(nodes, memory);
      if (kind === 'predicates')
        expect(Array.from(memory.readUint32(fixture.isSimple))).toEqual([0]);
      expect(Array.from(memory.readUint32(fixture.overflow))).toEqual([kind === 'split' ? 0 : 1]);
      if (kind !== 'predicates') {
        expect(Array.from(memory.readUint32(fixture.candidateOverflow))).toEqual([1]);
        expect(memory.readUint32(fixture.requiredCount)[0]).toBeLessThan(64);
      }
    }
  } finally {
    device.destroy();
  }
});

it.each([
  'count',
  'overflow',
  'both'
] as const)('status-only %s is written and reset on reuse', request => {
  const {device, graph, memory, view} = createHarness();
  try {
    const fixture = createStatusOnlyFixture(request, view);
    const nodes = fixture.contributor.getCommandNodes(graph);
    expect(nodes.length).toBeGreaterThan(0);
    for (const clean of [false, true]) {
      if (clean) memory.write(fixture.positions, fixture.cleanPositions);
      memory.write(fixture.intersectionCount, Uint32Array.of(99));
      memory.write(fixture.overflow, Uint32Array.of(99));
      executeCommandNodesOnCPU(nodes, memory);
      if (request !== 'overflow')
        expect(Array.from(memory.readUint32(fixture.intersectionCount))).toEqual([clean ? 0 : 1]);
      if (request !== 'count')
        expect(Array.from(memory.readUint32(fixture.overflow))).toEqual([
          request === 'overflow' && !clean ? 1 : 0
        ]);
    }
  } finally {
    device.destroy();
  }
});

it.each([0, -1, 1.5, NaN])('status-only validates intersectionCapacity %s', capacity => {
  const {device, view} = createHarness();
  try {
    expect(() => createStatusOnlyFixture('count', view, capacity)).toThrow(
      /status-only.*intersectionCapacity/
    );
  } finally {
    device.destroy();
  }
});

it.each([
  'split',
  'shared'
] as const)('%s retains one complete line at the vertex capacity boundary', kind => {
  const {device, graph, memory, view} = createHarness();
  try {
    const fixture = createCapacityBoundaryFixture(kind, view);
    executeCommandNodesOnCPU(fixture.contributor.getCommandNodes(graph), memory);
    expect(Array.from(memory.readUint32(fixture.count))).toEqual([1]);
    expect(Array.from(memory.readUint32(fixture.overflow))).toEqual([1]);
    expect(Array.from(memory.readUint32(fixture.candidateOverflow))).toEqual([0]);
    expect(Array.from(memory.readUint32(fixture.offsets))).toEqual([0, 2, 2]);
    expect(Array.from(memory.readFloat32(fixture.positions))).toEqual([0, 0, 1, 0]);
  } finally {
    device.destroy();
  }
});

it('complete predicate tree preserves a positive simplicity result', () => {
  const {device, graph, memory, view} = createHarness();
  const fixture = createWrapperLossFixture('predicates', view, 8);
  try {
    executeCommandNodesOnCPU(fixture.contributor.getCommandNodes(graph), memory);
    expect(Array.from(memory.readUint32(fixture.isSimple))).toEqual([1]);
    expect(Array.from(memory.readUint32(fixture.overflow))).toEqual([0]);
  } finally {
    device.destroy();
  }
});
