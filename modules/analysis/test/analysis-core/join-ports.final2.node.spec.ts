// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getViewBindingRange,
  GPUCommandGraph,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {getStorageTopKNodeRecord} from '../../src/analysis-core/kernels/storage-backed-top-k';
import {GPUSpatialJoinCandidates} from '../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-candidates';
import {GPUSpatialPredicateJoin} from '../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import {GPUNearestFeatureWeights} from '../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-weights';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../gpu-spatial-analysis/segment-intersection/cpu-executor-test-hooks';

useCPUExecutorTestHooks();

it('final2 nearest weights capture getter-backed output metadata and status once', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let rowCount = 1;
  let rowReads = 0;
  let outputReads = 0;
  const information = {
    get rowCount() {
      rowReads++;
      return rowCount;
    },
    transform: 'raw' as const,
    symmetric: 'unknown' as const,
    provenance: {source: 'original'}
  };
  const candidate = createTransientView(graph, 'metadata-candidate', 'uint32', 1);
  class Output {
    offsets = createTransientView(graph, 'metadata-offsets', 'uint32', 2);
    neighbors = createTransientView(graph, 'metadata-neighbors', 'uint32', 1);
    weights = createTransientView(graph, 'metadata-weights', 'float32', 1);
    status = {
      overflow: createTransientView(graph, 'metadata-overflow', 'uint32', 1),
      candidateOverflow: candidate
    };
    get information() {
      return information;
    }
  }
  const output = new Output();
  try {
    const producer = new GPUNearestFeatureWeights({
      neighborRows: createTransientView(graph, 'metadata-input', 'uint32', 1),
      neighborCounts: createTransientView(graph, 'metadata-counts', 'uint32', 1),
      slotCapacity: 1,
      get output() {
        outputReads++;
        return output;
      }
    });
    rowCount = 5;
    information.provenance.source = 'changed';
    Reflect.deleteProperty(output.status, 'candidateOverflow');
    expect(producer.props.output.information?.rowCount).toBe(1);
    expect(producer.props.output.information?.provenance?.source).toBe('original');
    expect([rowReads, outputReads]).toEqual([1, 1]);
    expect(producer.props.output.status.candidateOverflow).toBe(candidate);
  } finally {
    device.destroy();
  }
});

it('final2 anti join retains unmatched row 1 after caller redirects ids over candidate status', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const left = createTransientView(graph, 'left', 'float32x2', 2);
  const right = createTransientView(graph, 'right', 'float32x2', 1);
  const ids = createTransientView(graph, 'unmatched-ids', 'uint32', 1);
  const candidate = createTransientView(graph, 'candidate', 'uint32', 1);
  const unmatched = {
    ids,
    status: {
      count: createTransientView(graph, 'count', 'uint32', 1),
      overflow: createTransientView(graph, 'unmatched-overflow', 'uint32', 1)
    }
  };
  let outputReads = 0;
  const producer = new GPUSpatialPredicateJoin({
    left: {kind: 'points', positions: left},
    right: {kind: 'points', positions: right},
    predicate: 'intersects',
    engine: 'fast',
    how: 'anti',
    candidateCapacity: 2,
    get output() {
      outputReads++;
      return {unmatched, status: {candidateOverflow: candidate}};
    }
  });
  expect(outputReads).toBe(1);
  unmatched.ids = candidate;
  const memory = createCPUGraphMemory();
  memory.write(left, Float32Array.of(0, 0, 2, 0));
  memory.write(right, Float32Array.of(0, 0));
  try {
    const nodes = producer
      .getCommandNodes(graph)
      .filter(node => /-(scatter-unmatched|finalize(?:-\d+)?)$/.test(node.id));
    const scatter = nodes.find(node => node.id.endsWith('-scatter-unmatched'))!;
    const source = getGPUCommandNodeKernelSources([scatter])[0].source;
    const location = source.match(/@binding\((\d+)\) var<storage, read> unmatchedFlags:/)!;
    const resource = scatter.resources![Number(location[1])];
    if (!('buffer' in resource) || !(resource.buffer instanceof GraphDataView))
      throw new Error('Missing unmatched flags');
    memory.write(resource.buffer, Uint32Array.of(0, 1));
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(ids))).toEqual([1]);
    expect(Array.from(memory.readUint32(candidate))).toEqual([0]);
  } finally {
    device.destroy();
  }
});

for (const kind of ['candidates', 'weights'] as const) {
  it(`final2 ${kind} accepts adjacent status words in separate passes and rejects the same word`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const flags = createTransientView(graph, 'flags', 'uint32', 2);
    const overflow = graph.createDataView(flags.buffer, {
      format: 'uint32',
      byteOffset: 0,
      length: 1
    });
    const candidate = graph.createDataView(flags.buffer, {
      format: 'uint32',
      byteOffset: 4,
      length: 1
    });
    const input = createTransientView(graph, 'input', 'float32x2', 2);
    const upstream = createTransientView(graph, 'upstream', 'uint32', 1);
    const neighbors = createTransientView(graph, 'neighbor-rows', 'uint32', 1);
    const neighborCounts = createTransientView(graph, 'neighbor-counts', 'uint32', 1);
    const output = {
      offsets: createTransientView(graph, 'offsets', 'uint32', 2),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 1),
      weights: createTransientView(graph, 'weights', 'float32', 1),
      status: {overflow, candidateOverflow: candidate}
    };
    const pairs = {
      leftRows: createTransientView(graph, 'pair-left', 'uint32', 4),
      rightRows: createTransientView(graph, 'pair-right', 'uint32', 4),
      status: {
        count: createTransientView(graph, 'pair-count', 'uint32', 1),
        overflow,
        candidateOverflow: candidate
      }
    };
    const create = (sameWord: boolean) =>
      kind === 'candidates'
        ? new GPUSpatialJoinCandidates({
            left: {kind: 'points', positions: input},
            right: {kind: 'points', positions: input},
            leafCapacity: 1,
            output: {
              ...pairs,
              status: {...pairs.status, candidateOverflow: sameWord ? overflow : candidate}
            }
          })
        : new GPUNearestFeatureWeights({
            slotCapacity: 1,
            neighborRows: neighbors,
            neighborCounts,
            neighborStatus: {overflow: upstream},
            output: {
              ...output,
              status: {overflow, candidateOverflow: sameWord ? overflow : candidate}
            }
          });
    try {
      expect(() => create(true)).toThrow(/overlap/);
      const producer = create(false);
      const allNodes = producer.getCommandNodes(graph);
      const nodes = allNodes.filter(node =>
        kind === 'weights'
          ? /-(offsets|candidate-status)$/.test(node.id)
          : /-(write-pairs|candidate-status)$/.test(node.id)
      );
      const memory = createCPUGraphMemory();
      memory.write(neighbors, Uint32Array.of(3));
      memory.write(neighborCounts, Uint32Array.of(1));
      if (kind === 'weights') {
        const counts = allNodes
          .flatMap(node => node.resources ?? [])
          .find(
            resource =>
              'buffer' in resource &&
              resource.buffer instanceof GraphDataView &&
              resource.buffer.buffer.id.endsWith('-counts')
          );
        if (!counts || !('buffer' in counts) || !(counts.buffer instanceof GraphDataView))
          throw new Error('Missing counts');
        memory.write(counts.buffer, Uint32Array.of(1));
      }
      if (kind === 'candidates') {
        const resources = allNodes.flatMap(node => node.resources ?? []);
        const state = resources.find(
          resource =>
            'buffer' in resource &&
            resource.buffer instanceof GraphDataView &&
            resource.buffer.buffer.id.endsWith('-state')
        );
        if (!state || !('buffer' in state) || !(state.buffer instanceof GraphDataView))
          throw new Error('Missing state');
        memory.write(state.buffer, Uint32Array.of(1, 0, 0, 0));
      }
      const lossView =
        kind === 'weights'
          ? upstream
          : allNodes
              .flatMap(node => node.resources ?? [])
              .flatMap(resource =>
                'buffer' in resource &&
                resource.buffer instanceof GraphDataView &&
                resource.buffer.buffer.id.endsWith('-bvh-overflow')
                  ? [resource.buffer]
                  : []
              )[0];
      if (!lossView) throw new Error('Missing loss input');
      for (const loss of [1, 0]) {
        memory.write(flags, Uint32Array.of(29, 37));
        memory.write(lossView, Uint32Array.of(loss));
        if (kind === 'weights') {
          for (const node of nodes) {
            const record = getStorageTopKNodeRecord(node)!;
            const source = record.source
              .replace(/^\s*_\s*=[^;]*;/gm, '')
              .replace(/@workgroup_size\((\d+)\)/g, '@workgroup_size($1, 1, 1)');
            const copies = record.bindings.map(binding => {
              const range = getViewBindingRange(binding.view);
              const original = new Uint32Array(
                memory.getBuffer(binding.view.buffer).backing,
                range.offset,
                range.size / 4
              );
              return {original, words: original.slice()};
            });
            const bindings: Record<number, Uint32Array> = Object.fromEntries(
              copies.map(({words}, index) => [index, words])
            );
            bindings[record.bindings.length] = record.settings;
            new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
              'main',
              [record.layout.x, record.layout.y, record.layout.z],
              {0: bindings}
            );
            copies.forEach(({original, words}) => original.set(words));
          }
        } else executeCommandNodesOnCPU(nodes, memory);
        expect(Array.from(memory.readUint32(flags))).toEqual([0, loss]);
      }
    } finally {
      device.destroy();
    }
  });
}
