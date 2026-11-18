// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {
  computeNetworkFlowOnCPU,
  CPUNetworkFlowOperation
} from '../../src/gpu-network/network-optimization/cpu-network-flow-operation';
import {GPUNetworkFlowOperation} from '../../src/gpu-network/network-optimization/gpu-network-flow-operation';
import {computeKShortestPathsOnCPU} from '../../src/gpu-network/k-shortest-paths/cpu-k-shortest-paths-operation';
import {computeNetworkReachabilityOnCPU} from '../../src/gpu-network/network-reachability/cpu-network-reachability-operation';
import {
  GPUNetworkPathExtractionOperation,
  type NetworkPathExtractionOperationProps
} from '../../src/gpu-network/network-analysis/gpu-network-path-extraction-operation';
import {CPUNetworkPathExtractionOperation} from '../../src/gpu-network/network-analysis/cpu-network-path-extraction-operation';
import {constructNetworkOperationContributor} from '../utils/network-operation-test-utils';

function createChain(nodeCount: number) {
  return {
    offsets: Uint32Array.from({length: nodeCount + 1}, (_, row) => Math.min(row, nodeCount - 1)),
    neighbors: Uint32Array.from({length: nodeCount - 1}, (_, row) => row + 1),
    weights: new Float32Array(nodeCount - 1).fill(1)
  };
}

for (const costed of [false, true]) {
  it(`network:G4 binds zero ${costed ? 'cost phases' : 'relabels'} independently of augmentations`, () => {
    const input = {
      edges: Uint32Array.of(0, 1),
      capacities: Uint32Array.of(1),
      parameters: Uint32Array.of(0, 1, 1),
      costs: costed ? Uint32Array.of(2) : undefined
    };
    const options = {
      nodeCount: 2,
      solver: 'parallel' as const,
      mode: costed ? ('primal-dual' as const) : undefined,
      maximumRelabels: costed ? undefined : 0,
      maximumCostPhases: costed ? 0 : undefined
    };
    const result = computeNetworkFlowOnCPU(input, options);
    expect(result.flow).toEqual(Uint32Array.of(0));
    expect(result.status.converged[0]).toBe(0);
    expect(result.status.iterationLimitReached[0]).toBe(1);
  });
  it(`network:G4 selects the parallel ${costed ? 'primal-dual' : 'uncosted'} controlling budget`, () => {
    const input = {
      edges: Uint32Array.of(0, 1),
      capacities: Uint32Array.of(1),
      parameters: Uint32Array.of(0, 1, 1),
      costs: costed ? Uint32Array.of(2) : undefined
    };
    const options = {
      nodeCount: 2,
      solver: 'parallel' as const,
      mode: costed ? ('primal-dual' as const) : undefined,
      maximumAugmentations: 0
    };
    const result = computeNetworkFlowOnCPU(input, options);
    expect(result.flow).toEqual(Uint32Array.of(1));
    expect(result.status.converged[0]).toBe(1);
    const stopped = computeNetworkFlowOnCPU(input, {
      ...options,
      maximumRelabels: costed ? undefined : 0,
      maximumCostPhases: costed ? 0 : undefined
    });
    expect(stopped.flow).toEqual(Uint32Array.of(0));
    expect(stopped.status.converged[0]).toBe(0);
    expect(stopped.status.iterationLimitReached[0]).toBe(1);
  });
}

it('network:G4 captures inherited status getters once in both operation forms', () => {
  for (const Operation of [GPUNetworkFlowOperation, CPUNetworkFlowOperation]) {
    let reads = 0;
    const status = Object.create({
      get converged() {
        reads++;
        return 'complete';
      }
    }) as {converged: string};
    const operation = new Operation({
      nodeCount: 2,
      edgeCount: 1,
      solver: 'parallel',
      maximumRelabels: 1,
      edges: 'edges',
      capacities: 'capacities',
      parameters: 'parameters',
      output: {flow: 'flow', cut: 'cut', summary: 'summary', status}
    });
    expect(operation.contract.outputs).toHaveProperty('complete');
    expect(reads).toBe(1);
  }
});

it('network:G5 interrupts every active K-shortest query before committing a bounded round', () => {
  const topology = createChain(72);
  // Two disconnected components: chain 0..69 and edge 70->71.
  topology.offsets[70] = 69;
  topology.offsets[71] = 70;
  topology.offsets[72] = 70;
  topology.neighbors[69] = 71;
  const result = computeKShortestPathsOnCPU(
    {topology, sources: Uint32Array.of(0, 70), targets: Uint32Array.of(69, 71)},
    {routeCount: 1, maximumPathLength: 72, maximumIterations: 1}
  );
  expect(result.counts).toEqual(Uint32Array.of(0, 0));
  expect(result.converged[0]).toBe(0);
  expect(result.iterationLimitReached[0]).toBe(1);
});

it('network:G6 empty starts require a positive round before certifying reachability', () => {
  const result = computeNetworkReachabilityOnCPU({
    topology: createChain(2),
    startNodes: Uint32Array.of(0),
    startCount: Uint32Array.of(0),
    activeIterations: Uint32Array.of(0)
  });
  expect(result.costs).toEqual(Float32Array.of(Infinity, Infinity));
  expect(result.converged[0]).toBe(0);
  expect(result.iterationLimitReached[0]).toBe(1);
});

const PATH_PROPS: NetworkPathExtractionOperationProps = {
  nodeCount: 2,
  targetCapacity: 1,
  predecessors: 'predecessors',
  costs: 'costs',
  targets: 'targets',
  output: {
    capacity: 2,
    ids: 'ids',
    status: {count: 'count', overflow: 'overflow', requiredCount: 'required'}
  }
};

it('network:G9 rejects swapped row ports before lowering', () => {
  for (const Operation of [GPUNetworkPathExtractionOperation, CPUNetworkPathExtractionOperation]) {
    expect(
      () =>
        new Operation({
          ...PATH_PROPS,
          output: {
            ...PATH_PROPS.output,
            edgeRows: 'caller-edge-rows'
          } as NetworkPathExtractionOperationProps['output']
        })
    ).toThrow(/output.edgeRows/);
    expect(
      () =>
        new Operation({
          ...PATH_PROPS,
          edges: {
            edgeCount: 1,
            topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
            output: {
              capacity: 1,
              ids: 'edge-ids',
              status: {
                count: 'edge-count',
                overflow: 'edge-overflow',
                requiredCount: 'edge-required'
              },
              nodeRows: 'wrong'
            } as NonNullable<NetworkPathExtractionOperationProps['edges']>['output']
          }
        })
    ).toThrow(/edges.output.nodeRows/);
  }
});

it('network:G9 lowering keeps getter-backed output maps and validated capacities unchanged', () => {
  let reads = 0;
  const output = {...PATH_PROPS.output, status: {...PATH_PROPS.output.status}, nodeRows: 'rows'};
  Object.defineProperty(output, 'nodeRows', {
    get() {
      return ++reads === 1 ? 'rows' : 'changed';
    }
  });
  const operation = new GPUNetworkPathExtractionOperation({...PATH_PROPS, output});
  output.capacity = 1;
  output.status.count = 'targets';
  expect(operation.outputs['nodeRows']).toEqual({value: 'rows', format: 'uint32', length: 2});
  expect(() =>
    constructNetworkOperationContributor(operation, {
      predecessors: {format: 'uint32', length: 2},
      costs: {format: 'float32', length: 2},
      targets: {format: 'uint32', length: 1}
    })
  ).not.toThrow();
  expect(reads).toBe(1);
});

it('network:G12 copies only sparse frontier distances, preserving costs and round status', () => {
  let copiedElements = 0;
  const originalSlice = Float64Array.prototype.slice;
  const slice = vi.spyOn(Float64Array.prototype, 'slice').mockImplementation(function (
    this: Float64Array,
    start,
    end
  ) {
    const copy = originalSlice.call(this, start, end);
    copiedElements += copy.length;
    return copy;
  });
  try {
    const result = computeNetworkReachabilityOnCPU(
      {topology: createChain(4096), startNodes: Uint32Array.of(0)},
      {maximumIterations: 128, localIterations: 32}
    );
    expect(result.costs).toEqual(Float32Array.from({length: 4096}, (_, row) => row));
    expect(result.converged[0]).toBe(1);
    expect(result.iterationCount[0]).toBe(128);
    expect(copiedElements).toBeLessThanOrEqual(4096);
  } finally {
    slice.mockRestore();
  }
});

it('network:G12 snapshots all frontier sources before relaxing any of them', () => {
  const result = computeNetworkReachabilityOnCPU(
    {
      topology: createChain(3),
      startNodes: Uint32Array.of(0, 1),
      startCosts: Float32Array.of(0, 10)
    },
    {maximumIterations: 1, localIterations: 1}
  );
  expect(result.costs).toEqual(Float32Array.of(0, 1, 11));
  expect(result.converged[0]).toBe(0);
});

for (const costed of [false, true]) {
  it(`network:G4 one ${costed ? 'cost phase' : 'relabel phase'} can contain multiple paths`, () => {
    const result = computeNetworkFlowOnCPU(
      {
        edges: Uint32Array.of(0, 1, 0, 1),
        capacities: Uint32Array.of(1, 1),
        costs: costed ? Uint32Array.of(2, 2) : undefined,
        parameters: Uint32Array.of(0, 1, 2)
      },
      {
        nodeCount: 2,
        solver: 'parallel',
        mode: costed ? 'primal-dual' : undefined,
        maximumAugmentations: 0,
        maximumCostPhases: costed ? 1 : undefined,
        maximumRelabels: costed ? undefined : 1
      }
    );
    expect(result.flow).toEqual(Uint32Array.of(1, 1));
    expect(result.status.iterationCount[0]).toBe(1);
    expect(result.status.converged[0]).toBe(1);
  });
}

it('network:G4 different shortest hop counts require separate primal-dual phases', () => {
  const input = {
    edges: Uint32Array.of(0, 2, 0, 1, 1, 2),
    capacities: Uint32Array.of(1, 1, 1),
    costs: Uint32Array.of(2, 1, 1),
    parameters: Uint32Array.of(0, 2, 2)
  };
  const options = {
    nodeCount: 3,
    solver: 'parallel' as const,
    mode: 'primal-dual' as const,
    maximumCostPhases: 1
  };
  const limited = computeNetworkFlowOnCPU(input, options);
  expect(limited.flow).toEqual(Uint32Array.of(1, 0, 0));
  expect(limited.status.iterationCount[0]).toBe(1);
  expect(limited.status.converged[0]).toBe(0);
  const complete = computeNetworkFlowOnCPU(input, {...options, maximumCostPhases: 2});
  expect(complete.flow).toEqual(Uint32Array.of(1, 1, 1));
  expect(complete.status.iterationCount[0]).toBe(2);
  expect(complete.status.converged[0]).toBe(1);
});

it('network:G4 relaxation rounds bind parallel searches including the unchanged sweep', () => {
  const input = {
    edges: Uint32Array.of(0, 1),
    capacities: Uint32Array.of(1),
    parameters: Uint32Array.of(0, 1, 1)
  };
  const limited = computeNetworkFlowOnCPU(input, {
    nodeCount: 2,
    solver: 'parallel',
    maximumRelabels: 1,
    maximumRelaxationRounds: 1
  });
  expect(limited.flow).toEqual(Uint32Array.of(0));
  expect(limited.status.converged[0]).toBe(0);
  const complete = computeNetworkFlowOnCPU(input, {
    nodeCount: 2,
    solver: 'parallel',
    maximumRelabels: 1,
    maximumRelaxationRounds: 2
  });
  expect(complete.flow).toEqual(Uint32Array.of(1));
  expect(complete.status.converged[0]).toBe(1);
});
