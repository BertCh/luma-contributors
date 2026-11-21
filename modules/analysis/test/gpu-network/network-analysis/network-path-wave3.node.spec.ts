// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {
  createPublishNode,
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createPathResolveEdgesNode,
  createPathCheckTotalsNode,
  createPathSuppressUnrepresentableNode,
  createPathTotalsNode
} from '../../../src/gpu-network/network-analysis/network-path-extraction-passes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createReachabilityPredecessorEdgesNode} from '../../../src/gpu-network/network-reachability/network-reachability-passes';
import {GPUNetworkPathExtraction} from '../../../src/gpu-network/network-analysis/gpu-network-path-extraction';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

it('checks the actual WGSL final scan interval before publishing 2^32 rows', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const starts = createTransientView(graph, 'starts', 'uint32', 1),
      lengths = createTransientView(graph, 'lengths', 'uint32', 1),
      status = createTransientView(graph, 'status', 'uint32', 3),
      total = createTransientView(graph, 'total', 'uint32', 1),
      pathOffsets = createTransientView(graph, 'offsets', 'uint32', 2);
    createPathCheckTotalsNode(graph, {id: 'check', starts, lengths, status});
    createPathSuppressUnrepresentableNode(graph, {id: 'suppress', starts, lengths, status});
    createPathTotalsNode(graph, {
      id: 'publish',
      targetCapacity: 1,
      nodes: {starts, lengths, total, pathOffsets}
    });
    const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
    createPublishNode(graph, {
      id: 'publish-status',
      operation: 'test',
      requiredCount: total,
      overflowSources: [
        graph.createDataView(status.buffer, {format: 'uint32', length: 1, byteOffset: 8})
      ],
      output: {
        ids: createTransientView(graph, 'ids', 'uint32', 0),
        status: {
          count: scalar('count'),
          overflow: scalar('overflow'),
          requiredCount: scalar('required')
        }
      }
    });
    const data: Record<string, Uint32Array> = {
      starts: Uint32Array.of(4294901760),
      lengths: Uint32Array.of(65536),
      status: Uint32Array.of(0, 1, 0),
      nodeStarts: new Uint32Array(),
      nodeLengths: new Uint32Array(),
      nodeTotal: new Uint32Array(1),
      nodePathOffsets: new Uint32Array(2)
    };
    data['nodeStarts'] = data['starts'];
    data['nodeLengths'] = data['lengths'];
    data['totalIn'] = data['nodeTotal'];
    data['overflowIn0'] = data['status'];
    data['countOut'] = new Uint32Array(1);
    data['overflowOut'] = new Uint32Array(1);
    data['totalOut'] = new Uint32Array(1);
    for (const id of ['check', 'suppress', 'publish', 'publish-status']) {
      const record = getWGSLKernelNodeRecord(graph, id)!;
      expect(auditWGSLModule(record.source)).toEqual([]);
      new WgslExec(
        new WgslParser().parse(record.source.replace(/^\s*_\s*=[^;]*;/gm, ''))
      ).dispatchWorkgroups(
        'main',
        [1, 1, 1],
        {
          0: Object.fromEntries(
            record.bindings.map((binding, index) => [index, data[binding.name]])
          )
        },
        {constants: record.constants}
      );
    }
    expect(Array.from(data['status'])).toEqual([0, 1, 1]);
    expect(Array.from(data['nodeTotal'])).toEqual([0]);
    expect(Array.from(data['nodePathOffsets'])).toEqual([0, 0]);
    expect(Array.from(data['countOut'])).toEqual([0]);
    expect(Array.from(data['overflowOut'])).toEqual([1]);
  } finally {
    device.destroy();
  }
});

it('selects and resolves minimum-weight parallel CSR provenance in the actual WGSL', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const parents = createTransientView(graph, 'parents', 'uint32', 3),
      edges = createTransientView(graph, 'edges', 'uint32', 3);
    const offsets = createTransientView(graph, 'offsets', 'uint32', 4),
      neighbors = createTransientView(graph, 'neighbors', 'uint32', 3),
      weights = createTransientView(graph, 'weights', 'float32', 3);
    createReachabilityPredecessorEdgesNode(graph, {
      id: 'provenance',
      predecessors: parents,
      predecessorEdges: edges,
      topology: {offsets, neighbors, weights}
    });
    createPathResolveEdgesNode(graph, {
      id: 'resolve',
      nodeCount: 3,
      edgeCount: 3,
      linkTails: createTransientView(graph, 'tails', 'uint32', 2),
      linkHeads: createTransientView(graph, 'heads', 'uint32', 2),
      costs: createTransientView(graph, 'costs', 'float32', 3),
      offsets,
      neighbors,
      weights,
      predecessorEdges: edges,
      ids: createTransientView(graph, 'ids', 'uint32', 2)
    });
    const data: Record<string, Uint32Array | Float32Array> = {
      parents: Uint32Array.of(0xffffffff, 0, 1),
      edges: new Uint32Array(3).fill(0xffffffff),
      offsets: Uint32Array.of(0, 1, 3, 3),
      neighbors: Uint32Array.of(1, 2, 2),
      weights: Float32Array.of(0.1, 9, 0.2),
      linkTails: Uint32Array.of(0, 1),
      linkHeads: Uint32Array.of(1, 2),
      ids: new Uint32Array(2)
    };
    const execute = (id: string) => {
      const record = getWGSLKernelNodeRecord(graph, id)!;
      expect(auditWGSLModule(record.source)).toEqual([]);
      new WgslExec(
        new WgslParser().parse(record.source.replace(/^\s*_\s*=[^;]*;/gm, ''))
      ).dispatchWorkgroups(
        'main',
        [1, 1, 1],
        {
          0: Object.fromEntries(
            record.bindings.map((binding, index) => [index, data[binding.name]])
          )
        },
        {constants: record.constants}
      );
    };
    execute('provenance');
    data['predecessorEdges'] = data['edges'];
    execute('resolve');
    expect(Array.from(data['edges'])).toEqual([0xffffffff, 0, 2]);
    expect(Array.from(data['ids'])).toEqual([0, 2]);
    data['edges'][2] = 0;
    execute('resolve');
    expect(Array.from(data['ids'])).toEqual([0, 0xffffffff]);
  } finally {
    device.destroy();
  }
});

it('wires checked totals only when the active walk bound can exceed uint32', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const maximumPathLength of [1, 65536]) {
      const graph = new GPUCommandGraph(device);
      recordWGSLKernelNodes(graph);
      const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
      const contributor = new GPUNetworkPathExtraction({
        predecessors: createTransientView(graph, 'parents', 'uint32', 65536),
        costs: createTransientView(graph, 'costs', 'float32', 65536),
        targets: createTransientView(graph, 'targets', 'uint32', 65536),
        maximumPathLength,
        output: {
          ids: createTransientView(graph, 'ids', 'uint32', 0),
          status: {
            count: scalar('count'),
            overflow: scalar('overflow'),
            requiredCount: scalar('required'),
            unrepresentable: scalar('unrepresentable')
          }
        }
      });
      contributor.getCommandNodes(graph);
      expect(Boolean(getWGSLKernelNodeRecord(graph, 'network-path-extraction-check-totals'))).toBe(
        maximumPathLength > 1
      );
    }
  } finally {
    device.destroy();
  }
});

it('S6 rejects negative subnormal provenance even when float comparisons flush to signed zero', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const parents = createTransientView(graph, 'parents', 'uint32', 2),
      edges = createTransientView(graph, 'edges', 'uint32', 2);
    const offsets = createTransientView(graph, 'offsets', 'uint32', 3),
      neighbors = createTransientView(graph, 'neighbors', 'uint32', 3),
      weights = createTransientView(graph, 'weights', 'float32', 3);
    createReachabilityPredecessorEdgesNode(graph, {
      id: 'choose',
      predecessors: parents,
      predecessorEdges: edges,
      topology: {offsets, neighbors, weights}
    });
    createPathResolveEdgesNode(graph, {
      id: 'resolve',
      nodeCount: 2,
      edgeCount: 3,
      linkTails: createTransientView(graph, 'tails', 'uint32', 1),
      linkHeads: createTransientView(graph, 'heads', 'uint32', 1),
      costs: createTransientView(graph, 'costs', 'float32', 2),
      offsets,
      neighbors,
      weights,
      predecessorEdges: edges,
      ids: createTransientView(graph, 'ids', 'uint32', 1)
    });
    const data: Record<string, Uint32Array | Float32Array> = {
      parents: Uint32Array.of(0xffffffff, 0),
      edges: new Uint32Array(2).fill(0xffffffff),
      offsets: Uint32Array.of(0, 3, 3),
      neighbors: Uint32Array.of(1, 1, 1),
      weights: new Float32Array(Uint32Array.of(0x80000001, 0, 0x80000000).buffer),
      linkTails: Uint32Array.of(0),
      linkHeads: Uint32Array.of(1),
      ids: new Uint32Array(1)
    };
    const execute = (id: string) => {
      const record = getWGSLKernelNodeRecord(graph, id)!;
      // Model the permitted FTZ comparison: a negative subnormal compares as signed zero.
      const source = record.source
        .replaceAll(
          'weight >= 0.0',
          '((bitcast<u32>(weight) & 0x7fffffffu) < 0x00800000u || weight >= 0.0)'
        )
        .replace(/^\s*_\s*=[^;]*;/gm, '');
      new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
        'main',
        [1, 1, 1],
        {
          0: Object.fromEntries(
            record.bindings.map((binding, index) => [index, data[binding.name]])
          )
        },
        {constants: record.constants}
      );
    };
    execute('choose');
    expect(Array.from(data['edges'])).toEqual([0xffffffff, 1]);
    data['weights'][1] = Infinity;
    data['weights'][2] = Infinity;
    data['edges'].fill(0xffffffff);
    execute('choose');
    expect(Array.from(data['edges'])).toEqual([0xffffffff, 0xffffffff]);
    data['weights'][1] = 0;
    data['weights'][2] = -0;
    data['predecessorEdges'] = Uint32Array.of(0xffffffff, 0);
    execute('resolve');
    expect(Array.from(data['ids'])).toEqual([0xffffffff]);
    for (const slot of [1, 2]) {
      data['predecessorEdges'][1] = slot;
      execute('resolve');
      expect(Array.from(data['ids'])).toEqual([slot]);
    }
  } finally {
    device.destroy();
  }
});
