// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {Buffer as NodeBuffer} from 'node:buffer';
import {runInNewContext} from 'node:vm';
import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {decodeNetworkLouvainStatus} from '../../../src/gpu-network/network-optimization/cpu-network-louvain-operation';
import {
  decodeGPUKShortestPaths,
  readRouteWords
} from '../../../src/gpu-network/k-shortest-paths/k-shortest-paths-output';
import {GPUNetworkRationalAccessOperation} from '../../../src/gpu-network/network-optimization/gpu-network-rational-access-operation';
import {CPUNetworkRationalAccessOperation} from '../../../src/gpu-network/network-optimization/cpu-network-rational-access-operation';
import {GPUNetworkCentralityOperation} from '../../../src/gpu-network/network-centrality/gpu-network-centrality-operation';
import {GPUNetworkCentrality} from '../../../src/gpu-network/network-centrality/gpu-network-centrality';
import {
  GPUNetworkFacilityOptimization,
  getNetworkFacilitySearchLimits
} from '../../../src/gpu-network/network-optimization/gpu-network-facility-optimization';
import {prepareGPUNetworkFlowDemands} from '../../../src/gpu-network/network-optimization/prepare-flow-demands';
import {GPUKShortestPaths} from '../../../src/gpu-network/k-shortest-paths/gpu-k-shortest-paths';
import {setupKShortestPaths} from '../k-shortest-paths/k-shortest-paths-test-utils';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('B1 decodes the exact Node Buffer range and foreign-realm summary', () => {
  const storage = NodeBuffer.alloc(32);
  [1, 1, 2, 3].forEach((word, index) => storage.writeUInt32LE(word, 16 + index * 4));
  const foreign = runInNewContext('Uint32Array.of(1,1,2,3).buffer') as ArrayBuffer;
  const expected = {
    iterationCount: Uint32Array.of(3),
    converged: Uint32Array.of(1),
    invalidCount: Uint32Array.of(0),
    iterationLimitReached: Uint32Array.of(0),
    approximation: Uint32Array.of(1)
  };
  for (const input of [storage.subarray(16), foreign])
    expect(decodeNetworkLouvainStatus(input)).toEqual(expected);
});
it('B8 reads foreign byte buffers and rejects wrong numeric element types', () => {
  for (const expression of [
    'new Uint8Array(Uint32Array.of(1,2,3,4).buffer)',
    'Uint32Array.of(1,2,3,4).buffer'
  ]) {
    const input = runInNewContext(expression) as Uint8Array | ArrayBuffer;
    expect(readRouteWords(input, 'uint32', 4, 'foreign')).toEqual(Uint32Array.of(1, 2, 3, 4));
    const result = decodeGPUKShortestPaths(
      {
        vertices: input,
        edges: input,
        lengths: Uint32Array.of(2),
        costs: Float32Array.of(1),
        counts: Uint32Array.of(1),
        status: {
          converged: Uint32Array.of(1),
          iterationLimitReached: Uint32Array.of(0),
          candidateOverflow: Uint32Array.of(0),
          iterationCount: Uint32Array.of(1)
        }
      },
      {queryCount: 1, routeCount: 1, maximumPathLength: 4}
    );
    expect(result.queries).toEqual([
      {count: 1, routes: [{length: 2, cost: 1, vertices: [1, 2], edges: [1]}]}
    ]);
  }
  expect(() => readRouteWords(Float32Array.of(1), 'uint32', 1, 'wrong')).toThrow(/Uint32Array/);
});
const rationalProps = {
  id: 'review-rational',
  originCount: 1,
  facilityCount: 1,
  costs: 'costs',
  demand: 'demand',
  supply: 'supply',
  parameters: 'parameters',
  output: {
    assignments: 'assignments',
    scores: 'scores',
    unservedDemand: 'unserved',
    summary: 'summary',
    status: {complete: 'complete'}
  }
};
for (const Constructor of [GPUNetworkRationalAccessOperation, CPUNetworkRationalAccessOperation]) {
  it(`B3 ${Constructor.name} refuses supplied empty optional graph names`, () => {
    for (const name of ['costsComplete', 'costLimit'] as const)
      expect(() => new Constructor({...rationalProps, [name]: ''})).toThrow(/non-empty/);
    expect(
      () =>
        new Constructor({
          ...rationalProps,
          output: {...rationalProps.output, status: {complete: ''}}
        })
    ).toThrow(/non-empty/);
  });
  it(`S2 ${Constructor.name} validates enums at construction`, () => {
    expect(
      () => new Constructor({...rationalProps, demandMode: 'continous' as 'integral'})
    ).toThrow(/review-rational.*demandMode.*continous/);
    expect(() => new Constructor({...rationalProps, stepMode: 'relative' as 'fraction'})).toThrow(
      /review-rational.*stepMode.*relative/
    );
  });
}
it('B2 preserves omission of the public transpose when reusing distance props', () => {
  const operation = new GPUNetworkCentralityOperation({
    nodeCount: 3,
    edgeCount: 2,
    pivotCount: 1,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    pivotNodes: 'pivots',
    output: {closeness: 'closeness'}
  });
  const contributor = constructNetworkOperationContributor(operation, {
    'topology.offsets': {format: 'uint32', length: 4},
    'topology.neighbors': {format: 'uint32', length: 2},
    'topology.weights': {format: 'float32', length: 2},
    pivotNodes: {format: 'uint32', length: 1}
  }) as GPUNetworkCentrality;
  expect(contributor.props.reverseTopology === undefined).toBe(true);
  const graph = contributor.props.topology.offsets.buffer.graph;
  if (!(graph instanceof GPUCommandGraph)) {
    throw new Error('Expected the operation fixture to own a GPUCommandGraph');
  }
  expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  expect(
    () =>
      new GPUNetworkCentrality({
        ...contributor.props,
        output: {betweenness: createTransientView(graph, 'betweenness', 'float32', 3)}
      })
  ).toThrow(/reverseTopology is required for betweenness/);
});
it('B4 applies the two-facility minimum for dispersion preflight', () => {
  expect(
    getNetworkFacilitySearchLimits({operation: 'p-dispersion', facilityCount: 1, demandCount: 1})
  ).toMatchObject({candidateCount: 2, supported: false, limitingDimension: 'facilityCount'});
  expect(
    getNetworkFacilitySearchLimits({operation: 'p-dispersion', facilityCount: 2, demandCount: 1})
  ).toMatchObject({candidateCount: 4, supported: true});
});
it('B6 writes only the four declared empty-batch status ports', () => {
  const {props, graph} = setupKShortestPaths({queryCount: 0});
  try {
    const myApplicationFlag = createTransientView(graph, 'application', 'uint32', 1);
    const status = {...props.output.status, myApplicationFlag};
    const contributor = new GPUKShortestPaths({...props, output: {...props.output, status}});
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes).toHaveLength(4);
    expect(
      getGPUCommandNodeKernelSources(nodes)
        .map(kernel => kernel.id)
        .join(' ')
    ).not.toContain('myApplicationFlag');
  } finally {
    graph.device.destroy();
  }
});
it('S1 identifies the missing target after a valid row and rejects null containers', () => {
  const prepare = (edges: unknown) =>
    prepareGPUNetworkFlowDemands({
      nodeCount: 3,
      edges: edges as [number, number][],
      capacities: [1, 1],
      demands: [0, 0, 0]
    });
  expect(() => prepare([[0, 1], [2]])).toThrow(/edges\[1\]\.target.*\[source, target\].*undefined/);
  expect(() => prepare([[0, 1], null])).toThrow(/edges\[1\].*\[source, target\]/);
});
it('S5 retains the caller validation identity for facility options', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let sequence = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `facility-${sequence++}`, format, length);
  try {
    expect(
      () =>
        new GPUNetworkFacilityOptimization({
          id: 'caller-facility',
          operation: 'typo' as 'backup-coverage',
          facilityCount: 1,
          costs: view('float32', 1),
          demand: view('float32', 1),
          capacities: view('float32', 1),
          threshold: view('float32', 1),
          output: {
            selectedFacilities: view('uint32', 1),
            assignments: view('uint32', 1),
            objective: view('float32', 1),
            summary: view('uint32', 4)
          }
        })
    ).toThrow(/caller-facility.*operation/);
  } finally {
    device.destroy();
  }
});
it('N1 validates an explicitly declared reverse edge count in GPU lowering', () => {
  const operation = new GPUNetworkCentralityOperation({
    id: 'reverse-count',
    nodeCount: 2,
    edgeCount: 1,
    reverseEdgeCount: 3,
    pivotCount: 1,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    reverseTopology: {
      offsets: 'reverse-offsets',
      neighbors: 'reverse-neighbors',
      weights: 'reverse-weights'
    },
    pivotNodes: 'pivots',
    output: {closeness: 'closeness'}
  });
  expect(() =>
    constructNetworkOperationContributor(operation, {
      'topology.offsets': {format: 'uint32', length: 3},
      'topology.neighbors': {format: 'uint32', length: 1},
      'topology.weights': {format: 'float32', length: 1},
      'reverseTopology.offsets': {format: 'uint32', length: 3},
      'reverseTopology.neighbors': {format: 'uint32', length: 1},
      'reverseTopology.weights': {format: 'float32', length: 1},
      pivotNodes: {format: 'uint32', length: 1}
    })
  ).toThrow(/reverse-count.*reverseEdgeCount.*3/);
});

it('S3 decodes later queries with unequal producer strides and mixed certificates', async () => {
  const {decodeGPUAlternativeRoutes} = await import(
    '../../../src/gpu-network/alternative-routes/gpu-alternative-routes'
  );
  const padding = 0xffffffff;
  const columns = {
    lengths: Uint32Array.of(0, 2, 1),
    costs: Float32Array.of(Infinity, 2, 0),
    counts: Uint32Array.of(0, 1, 1)
  };
  const queries = [
    {count: 0, routes: []},
    {count: 1, routes: [{length: 2, cost: 2, vertices: [5, 6], edges: [9]}]},
    {count: 1, routes: [{length: 1, cost: 0, vertices: [7], edges: []}]}
  ];
  expect(
    decodeGPUAlternativeRoutes(
      {
        ...columns,
        vertices: Uint32Array.of(padding, padding, padding, 5, 6, padding, 7, padding, padding),
        edges: Uint32Array.of(
          padding,
          padding,
          padding,
          9,
          padding,
          padding,
          padding,
          padding,
          padding
        ),
        complete: Uint32Array.of(1, 0, 1)
      },
      {queryCount: 3, routeCount: 1, nodeCount: 3}
    )
  ).toEqual({queries: queries.map((query, row) => ({...query, complete: row !== 1}))});
  expect(
    decodeGPUKShortestPaths(
      {
        ...columns,
        vertices: Uint32Array.of(
          padding,
          padding,
          padding,
          padding,
          5,
          6,
          padding,
          padding,
          7,
          padding,
          padding,
          padding
        ),
        edges: Uint32Array.of(
          padding,
          padding,
          padding,
          padding,
          9,
          padding,
          padding,
          padding,
          padding,
          padding,
          padding,
          padding
        ),
        status: {
          converged: Uint32Array.of(0),
          iterationLimitReached: Uint32Array.of(1),
          candidateOverflow: Uint32Array.of(0),
          iterationCount: Uint32Array.of(1)
        }
      },
      {queryCount: 3, routeCount: 1, maximumPathLength: 4}
    )
  ).toEqual({
    queries,
    status: {
      converged: false,
      iterationLimitReached: true,
      candidateOverflow: false,
      iterationCount: 1
    }
  });
});
