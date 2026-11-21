// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {computeAlternativeRoutesOnCPU} from '../../../src/gpu-network/alternative-routes/cpu-alternative-routes-operation';
import {computeKShortestPathsOnCPU} from '../../../src/gpu-network/k-shortest-paths/cpu-k-shortest-paths-operation';
import {computeNetworkStructureOnCPU} from '../../../src/gpu-network/network-structure/cpu-network-structure-operation';
import {computeNetworkCentralityOnCPU} from '../../../src/gpu-network/network-centrality/cpu-network-centrality-operation';
import {computeNetworkLouvainOnCPU} from '../../../src/gpu-network/network-optimization/cpu-network-louvain-operation';
const topology = {
  offsets: Uint32Array.of(0, 0),
  neighbors: new Uint32Array(),
  weights: new Float32Array()
};
it('accepts zero queries over a one-node graph', () => {
  const input = {topology, sources: new Uint32Array(), targets: new Uint32Array()};
  for (const result of [
    computeAlternativeRoutesOnCPU(input, {routeCount: 2}),
    computeKShortestPathsOnCPU(input, {routeCount: 2})
  ]) {
    expect(result.counts.length).toBe(0);
    expect(result.costs.length).toBe(0);
  }
  const result = computeKShortestPathsOnCPU(input, {routeCount: 2});
  expect(Array.from(result.converged)).toEqual([1]);
  expect(Array.from(result.iterationLimitReached)).toEqual([0]);
});
it('accepts nested structure topology with the same exact labels', () => {
  expect(computeNetworkStructureOnCPU({topology}).strongComponents).toEqual(Uint32Array.of(0));
  expect(computeNetworkStructureOnCPU(topology).strongComponents).toEqual(Uint32Array.of(0));
});
it('computes outward weighted distances without a transpose and requires one for betweenness', () => {
  const input = {
    topology: {
      offsets: Uint32Array.of(0, 1, 2, 2),
      neighbors: Uint32Array.of(1, 2),
      weights: Float32Array.of(2, 3)
    },
    pivotNodes: Uint32Array.of(0)
  };
  const result = computeNetworkCentralityOnCPU(input, {computeBetweenness: false});
  // Distances 2 and 5: closeness = 2/7; harmonic = 1/2 + 1/5.
  expect(Number.isFinite(result.closeness[0])).toBe(true);
  expect(result.closeness[0]).toBeCloseTo(2 / 7);
  expect(result.harmonicCloseness[0]).toBeCloseTo(0.7);
  expect(() => computeNetworkCentralityOnCPU(input)).toThrow(
    /reverseTopology is required for betweenness/
  );
});
it('names the empty-weight Louvain diagnostics', () => {
  const result = computeNetworkLouvainOnCPU(
    {weights: Float32Array.of(0), parameters: Float32Array.of(1, 0)},
    {nodeCount: 1}
  );
  expect(result.status).toEqual({
    iterationCount: Uint32Array.of(0),
    converged: Uint32Array.of(1),
    iterationLimitReached: Uint32Array.of(0),
    invalidCount: Uint32Array.of(0),
    approximation: Uint32Array.of(1)
  });
});

it('distinguishes missing travel costs from a complete rational-access decision and recovers', async () => {
  const {computeNetworkRationalAccessOnCPU} = await import(
    '../../../src/gpu-network/network-optimization/cpu-network-rational-access-operation'
  );
  const input = {
    costs: Float32Array.of(Infinity),
    demand: Float32Array.of(1),
    supply: Float32Array.of(1),
    parameters: Float32Array.of(1, 1, 0.5, 0, 10, 0.01),
    costsComplete: Uint32Array.of(0)
  };
  const partial = computeNetworkRationalAccessOnCPU(input);
  expect(Array.from(partial.unservedDemand)).toEqual([1]);
  expect(Array.from(partial.status.complete)).toEqual([0]);
  input.costs[0] = 2;
  input.costsComplete[0] = 1;
  const recovered = computeNetworkRationalAccessOnCPU(input);
  expect(Array.from(recovered.assignments)).toEqual([1]);
  expect(Array.from(recovered.status.complete)).toEqual([1]);
  expect(
    Array.from(
      computeNetworkRationalAccessOnCPU({...input, costLimit: Float32Array.of(1)}).status.complete
    )
  ).toEqual([0]);
});
it('reports the independent demand cap even when candidate count remains one', async () => {
  const {getNetworkFacilitySearchLimits} = await import(
    '../../../src/gpu-network/network-optimization/gpu-network-facility-optimization'
  );
  const options = {operation: 'capacitated-p-median' as const, facilityCount: 1};
  expect(getNetworkFacilitySearchLimits({...options, demandCount: 64})).toMatchObject({
    candidateCount: 1,
    supported: true
  });
  expect(getNetworkFacilitySearchLimits({...options, demandCount: 65})).toMatchObject({
    candidateCount: 1,
    supported: false,
    limitingDimension: 'demandCount'
  });
});
it('decodes empty route queries with distinct completion scopes and byte offsets', async () => {
  const {decodeGPUAlternativeRoutes} = await import(
    '../../../src/gpu-network/alternative-routes/gpu-alternative-routes'
  );
  const {decodeGPUKShortestPaths} = await import(
    '../../../src/gpu-network/k-shortest-paths/k-shortest-paths-output'
  );
  const byteWords = (words: Uint32Array | Float32Array) => {
    const storage = new Uint8Array(words.byteLength + 3);
    storage.set(new Uint8Array(words.buffer, words.byteOffset, words.byteLength), 3);
    return storage.subarray(3);
  };
  const columns = {
    vertices: byteWords(Uint32Array.of(0, 1, 0, 0, 0, 0, 0, 0)),
    edges: byteWords(Uint32Array.of(0, 0, 0, 0, 0, 0, 0, 0)),
    lengths: byteWords(Uint32Array.of(2, 0, 0, 0)),
    costs: byteWords(Float32Array.of(1, Infinity, Infinity, Infinity)),
    counts: byteWords(Uint32Array.of(1, 0))
  };
  const alternative = decodeGPUAlternativeRoutes(
    {...columns, complete: byteWords(Uint32Array.of(1, 1))},
    {queryCount: 2, routeCount: 2, nodeCount: 2}
  );
  expect(alternative.queries[1]).toEqual({count: 0, routes: [], complete: true});
  expect(alternative.queries[0].routes[0]).toEqual({
    length: 2,
    cost: 1,
    vertices: [0, 1],
    edges: [0]
  });
  const decoded = decodeGPUKShortestPaths(
    {
      ...columns,
      status: {
        converged: Uint32Array.of(0),
        iterationLimitReached: Uint32Array.of(1),
        candidateOverflow: Uint32Array.of(0),
        iterationCount: Uint32Array.of(1)
      }
    },
    {queryCount: 2, routeCount: 2, maximumPathLength: 2}
  );
  expect(decoded.queries[1]).toEqual({count: 0, routes: []});
  expect(decoded.status.iterationLimitReached).toBe(true);
  expect(() =>
    decodeGPUAlternativeRoutes(
      {...columns, complete: new Uint8Array(3)},
      {queryCount: 2, routeCount: 2, nodeCount: 2}
    )
  ).toThrow(/whole/);
});

it('constructs both GPU route forms for zero-query batches', async () => {
  const {GPUAlternativeRoutesOperation} = await import(
    '../../../src/gpu-network/alternative-routes/gpu-alternative-routes-operation'
  );
  const {GPUKShortestPathsOperation} = await import(
    '../../../src/gpu-network/k-shortest-paths/gpu-k-shortest-paths-operation'
  );
  const {constructNetworkOperationContributor} = await import(
    '../../utils/network-operation-test-utils'
  );
  const props = {
    nodeCount: 1,
    queryCount: 0,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    sources: 'sources',
    targets: 'targets',
    routeCount: 2,
    output: {
      vertices: 'vertices',
      edges: 'edges',
      lengths: 'lengths',
      costs: 'costs',
      counts: 'counts',
      complete: 'complete',
      status: {
        converged: 'converged',
        iterationLimitReached: 'limit',
        iterationCount: 'iterations',
        candidateOverflow: 'overflow'
      }
    }
  };
  const inputs = {
    offsets: {format: 'uint32' as const, length: 2},
    neighbors: {format: 'uint32' as const, length: 0},
    weights: {format: 'float32' as const, length: 0},
    sources: {format: 'uint32' as const, length: 0},
    targets: {format: 'uint32' as const, length: 0}
  };
  expect(() =>
    constructNetworkOperationContributor(
      new GPUAlternativeRoutesOperation({
        ...props,
        output: {
          vertices: 'vertices',
          edges: 'edges',
          lengths: 'lengths',
          costs: 'costs',
          counts: 'counts',
          complete: 'complete'
        }
      }),
      inputs
    )
  ).not.toThrow();
  expect(() =>
    constructNetworkOperationContributor(new GPUKShortestPathsOperation(props), inputs)
  ).not.toThrow();
});

it('publishes identical literal Louvain diagnostics through the array and graph forms', async () => {
  const {CPUNetworkLouvainOperation} = await import(
    '../../../src/gpu-network/network-optimization/cpu-network-louvain-operation'
  );
  const names = [
    'iterationCount',
    'converged',
    'iterationLimitReached',
    'invalidCount',
    'approximation'
  ];
  const operation = new CPUNetworkLouvainOperation({
    nodeCount: 1,
    weights: 'weights',
    parameters: 'parameters',
    output: {
      communities: 'communities',
      objectiveTrace: 'trace',
      summary: 'summary',
      status: Object.fromEntries(names.map(name => [name, name]))
    }
  });
  const vector = (values: Float32Array) => ({
    format: 'float32' as const,
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  const rows = await operation.execute({
    inputs: {weights: vector(Float32Array.of(0)), parameters: vector(Float32Array.of(1, 0))}
  });
  // No edge weight: no sweep needed; converged identity partition remains heuristic.
  for (const [name, expected] of Object.entries({
    iterationCount: 0,
    converged: 1,
    iterationLimitReached: 0,
    invalidCount: 0,
    approximation: 1
  }))
    expect(rows[name]).toEqual(Uint32Array.of(expected));
});

it('constructs GPU centrality distance metrics without transpose and refuses both betweenness ports', async () => {
  const {GPUNetworkCentralityOperation} = await import(
    '../../../src/gpu-network/network-centrality/gpu-network-centrality-operation'
  );
  const {constructNetworkOperationContributor} = await import(
    '../../utils/network-operation-test-utils'
  );
  const props = {
    nodeCount: 3,
    edgeCount: 2,
    pivotCount: 1,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    pivotNodes: 'pivots',
    output: {closeness: 'closeness', harmonicCloseness: 'harmonic', reachedCounts: 'reached'}
  };
  expect(() =>
    constructNetworkOperationContributor(new GPUNetworkCentralityOperation(props), {
      'topology.offsets': {format: 'uint32', length: 4},
      'topology.neighbors': {format: 'uint32', length: 2},
      'topology.weights': {format: 'float32', length: 2},
      pivotNodes: {format: 'uint32', length: 1}
    })
  ).not.toThrow();
  for (const output of [{betweenness: 'nodes'}, {edgeBetweenness: 'edges'}])
    expect(() => new GPUNetworkCentralityOperation({...props, output})).toThrow(
      /reverseTopology is required for betweenness/
    );
});
