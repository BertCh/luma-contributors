// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {
  computeNetworkReachabilityOnCPU,
  CPUNetworkReachabilityOperation
} from '../../../src/gpu-network/network-reachability/cpu-network-reachability-operation';
import {computeNetworkPathExtractionOnCPU} from '../../../src/gpu-network/network-analysis/cpu-network-path-extraction-operation';

it('composes authoritative predecessor slots across rounded cost labels', () => {
  const topology = {
    offsets: Uint32Array.of(0, 1, 3, 3),
    neighbors: Uint32Array.of(1, 2, 2),
    weights: Float32Array.of(0.1, 9, 0.2)
  };
  const search = computeNetworkReachabilityOnCPU({topology, startNodes: Uint32Array.of(0)});
  const result = computeNetworkPathExtractionOnCPU({
    predecessors: search.predecessors,
    predecessorEdges: search.predecessorEdges,
    predecessorsComplete: search.converged,
    costs: search.costs,
    targets: Uint32Array.of(2),
    capacity: 3,
    edges: {...topology, capacity: 2}
  });
  // The sole cheap route has weights .1 + .2; parallel slot 1 costs 9.
  expect(Array.from(result.edges!.edgeRows)).toEqual([0, 2]);
});

it('composes the guide certificate for a truncated zero-weight tie chain', () => {
  const topology = {
    offsets: Uint32Array.of(0, 1, 2, 3, 3),
    neighbors: Uint32Array.of(1, 2, 3),
    weights: Float32Array.of(0, 0, 0)
  };
  const search = computeNetworkReachabilityOnCPU(
    {topology, startNodes: Uint32Array.of(0)},
    {localIterations: 1, maximumTieIterations: 1}
  );
  const result = computeNetworkPathExtractionOnCPU({
    predecessors: search.predecessors,
    costs: search.costs,
    predecessorsComplete: search.converged,
    targets: Uint32Array.of(3),
    capacity: 4
  });
  expect(search.converged[0]).toBe(0);
  expect(result.pathFound[0]).toBe(0);
  expect(Array.from(result.ids.slice(0, result.count[0]))).toEqual([]);
});

it('admits 65536 targets and vertices when the walk bound is one', () => {
  const result = computeNetworkPathExtractionOnCPU({
    predecessors: new Uint32Array(65536).fill(0xffffffff),
    costs: new Float32Array(65536),
    targets: new Uint32Array(65536),
    maximumPathLength: 1,
    capacity: 0
  });
  expect(result.requiredCount[0]).toBe(65536);
  expect(result.overflow[0]).toBe(1);
});

it('rejects provenance from the wrong parent and preserves minimum-weight then slot ties', () => {
  const topology = {
    offsets: Uint32Array.of(0, 3, 3),
    neighbors: Uint32Array.of(1, 1, 1),
    weights: Float32Array.of(4, 1, 1)
  };
  const search = computeNetworkReachabilityOnCPU({topology, startNodes: Uint32Array.of(0)});
  expect(Array.from(search.predecessorEdges)).toEqual([0xffffffff, 1]);
  expect(() =>
    computeNetworkPathExtractionOnCPU({
      predecessors: search.predecessors,
      predecessorEdges: Uint32Array.of(0xffffffff, 3),
      costs: search.costs,
      targets: Uint32Array.of(1),
      capacity: 2,
      edges: {...topology, capacity: 1}
    })
  ).toThrow(/predecessorEdges.*parent/);
});

it('publishes predecessor slots from the CPU named search form', async () => {
  const operation = new CPUNetworkReachabilityOperation({
    nodeCount: 3,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    startNodes: 'starts',
    output: {
      costs: 'costs',
      predecessors: 'parents',
      predecessorEdges: 'slots',
      status: {converged: 'complete'}
    }
  });
  const values = {
    offsets: Uint32Array.of(0, 1, 3, 3),
    neighbors: Uint32Array.of(1, 2, 2),
    weights: Float32Array.of(0.1, 9, 0.2),
    startNodes: Uint32Array.of(0)
  };
  const inputs = Object.fromEntries(
    Object.entries(values).map(([port, rows]) => [
      port,
      {
        format: port === 'weights' ? ('float32' as const) : ('uint32' as const),
        length: rows.length,
        chunks: [{values: rows, length: rows.length}],
        pack: () => rows
      }
    ])
  );
  const result = await operation.execute({inputs});
  expect(result['slots']).toEqual(Uint32Array.of(0xffffffff, 0, 2));
  expect(result['complete']).toEqual(Uint32Array.of(1));
});

it('S4 skips CSR provenance reads for an empty multi-lane search', () => {
  let offsetReads = 0;
  const offsets = new Proxy(Uint32Array.of(0, 1, 1), {
    get(target, property) {
      if (typeof property === 'string' && /^\d+$/.test(property)) offsetReads++;
      return Reflect.get(target, property, target);
    }
  });
  const result = computeNetworkReachabilityOnCPU(
    {
      topology: {offsets, neighbors: Uint32Array.of(1), weights: Float32Array.of(1)},
      startNodes: new Uint32Array()
    },
    {laneCount: 2}
  );
  expect(Array.from(result.predecessorEdges)).toEqual(new Array(4).fill(0xffffffff));
  expect(offsetReads).toBe(0);
});
