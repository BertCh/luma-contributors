// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeNetworkReachabilityOnCPU} from '../../../src/gpu-network/network-reachability/cpu-network-reachability-operation';
import {computeNetworkServiceAreasOnCPU} from '../../../src/gpu-network/network-analysis/cpu-network-service-areas-operation';
import {computeNetworkStructureOnCPU} from '../../../src/gpu-network/network-structure/cpu-network-structure-operation';
import {CPUNetworkServiceAreasOperation} from '../../../src/gpu-network/network-analysis/cpu-network-service-areas-operation';
import {getOracleCPUVectors} from '../alternative-routes/output-oracle-harness';
import {
  GPUNetworkServiceAreasOperation,
  type NetworkServiceAreasOperationProps
} from '../../../src/gpu-network/network-analysis/gpu-network-service-areas-operation';
import {GPUNetworkServiceAreas} from '../../../src/gpu-network/network-analysis/gpu-network-service-areas';
import {
  GPUCommandGraph,
  createTransientView,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('R8 unassigned service-area nodes cannot certify one another as parents', async () => {
  // Only seed 0 and its immediate neighbor receive a label in one hop.
  // Equal NONE labels at nodes 2 and 3 do not designate a shared facility.
  const operation = new CPUNetworkServiceAreasOperation({
    nodeCount: 4,
    facilityCapacity: 1,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    facilities: 'facilities',
    maximumIterations: 5,
    maximumLabelIterations: 1,
    maximumTieIterations: 4,
    localIterations: 1,
    output: {
      assignments: 'assignments',
      predecessors: 'parents',
      facilityNodeCounts: 'counts',
      status: {converged: 'converged', iterationLimitReached: 'limited'}
    }
  });
  const result = await operation.execute({
    inputs: getOracleCPUVectors({
      offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2, 3, 3)},
      neighbors: {format: 'uint32', values: Uint32Array.of(1, 2, 3)},
      weights: {format: 'float32', values: Float32Array.of(1, 1, 1)},
      facilities: {format: 'uint32', values: Uint32Array.of(0)}
    })
  });
  expect(Array.from(result['assignments'] as Uint32Array)).toEqual([0, 0, 0xffffffff, 0xffffffff]);
  expect(Array.from(result['parents'] as Uint32Array)).toEqual([
    0xffffffff, 0, 0xffffffff, 0xffffffff
  ]);
  expect(Array.from(result['counts'] as Uint32Array)).toEqual([2]);
  expect([...(result['converged'] as Uint32Array), ...(result['limited'] as Uint32Array)]).toEqual([
    0, 1
  ]);
});

it('R1 service-area execution and repeated lowering retain captured bindings and budgets', async () => {
  const props: NetworkServiceAreasOperationProps = {
    nodeCount: 3,
    facilityCapacity: 1,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    facilities: 'facilities',
    activeIterations: 'active',
    maximumIterations: 1,
    localIterations: 1,
    maximumLabelIterations: 4,
    maximumTieIterations: 2,
    output: {
      assignments: 'assignments',
      costs: 'costs',
      facilityNodeCounts: 'counts',
      status: {converged: 'converged', iterationLimitReached: 'limited'}
    }
  };
  const cpu = new CPUNetworkServiceAreasOperation(props);
  const gpu = new GPUNetworkServiceAreasOperation(props);
  const inputs = getOracleCPUVectors({
    offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2, 2)},
    neighbors: {format: 'uint32', values: Uint32Array.of(1, 2)},
    weights: {format: 'float32', values: Float32Array.of(1, 1)},
    facilities: {format: 'uint32', values: Uint32Array.of(0)},
    activeIterations: {format: 'uint32', values: Uint32Array.of(0)}
  });
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const views: Record<string, GraphDataView> = {};
  for (const [name, vector] of Object.entries(inputs))
    views[name] = createTransientView(graph, name, vector.format, vector.length);
  for (const [name, descriptor] of Object.entries(gpu.outputs))
    views[name] = createTransientView(graph, name, descriptor.format, descriptor.length);
  for (let replay = 0; replay < 2; replay++) {
    const result = await cpu.execute({inputs});
    // A bound zero is always seed-only, even after deleting the caller's binding.
    expect.soft(Array.from(result['costs'] as Float32Array)).toEqual([0, Infinity, Infinity]);
    expect.soft(Array.from(result['counts'] as Uint32Array)).toEqual([1]);
    expect
      .soft([...(result['converged'] as Uint32Array), ...(result['limited'] as Uint32Array)])
      .toEqual([0, 1]);
    const contributor = gpu.createContributor(
      <Format extends GPUGraphVectorFormat>(name: string) => views[name] as GraphDataView<Format>
    );
    if (!(contributor instanceof GPUNetworkServiceAreas))
      throw new Error('Expected service-area contributor');
    expect.soft(contributor.maximumIterations).toBe(1);
    expect.soft(contributor.props.activeIterations === views['activeIterations']).toBe(true);
    expect(contributor.maximumLabelIterations).toBe(4);
    expect(contributor.maximumTieIterations).toBe(2);
    expect(contributor.localIterations).toBe(1);
    delete props.activeIterations;
    props.maximumIterations = 4;
    props.maximumLabelIterations = 1;
    props.maximumTieIterations = 1;
    props.localIterations = 2;
    props.topology.nodePositions = 'later-positions';
  }
});

for (const computePredecessors of [true, false])
  it(`R9 disabled predecessor phase performs no tie search (${computePredecessors})`, () => {
    // All four labels settle; a single tie hop reaches only node 1. Omitting
    // that phase leaves all parents uncomputed and no pending tie work to certify.
    const result = computeNetworkServiceAreasOnCPU(
      {
        topology: {
          offsets: Uint32Array.of(0, 1, 2, 3, 3),
          neighbors: Uint32Array.of(1, 2, 3),
          weights: Float32Array.of(0, 0, 0)
        },
        facilities: Uint32Array.of(0)
      },
      {
        maximumIterations: 5,
        maximumLabelIterations: 5,
        maximumTieIterations: 1,
        localIterations: 1,
        computePredecessors
      }
    );
    expect(Array.from(result.predecessors)).toEqual(
      computePredecessors
        ? [0xffffffff, 0, 0xffffffff, 0xffffffff]
        : [0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]
    );
    expect(Array.from(result.facilityNodeCounts)).toEqual([4]);
    expect([result.converged[0], result.iterationLimitReached[0]]).toEqual(
      computePredecessors ? [0, 1] : [1, 0]
    );
  });

for (const maximumIterations of [268435455, 268435456])
  it(`R11 valid large round budget exhausts a three-node graph (${maximumIterations})`, async () => {
    // Three unit-chain hops suffice, including examination of the terminal node.
    // A larger allowance cannot create any further strict nonnegative improvement.
    const result = computeNetworkServiceAreasOnCPU(
      {
        topology: {
          offsets: Uint32Array.of(0, 1, 2, 2),
          neighbors: Uint32Array.of(1, 2),
          weights: Float32Array.of(1, 1)
        },
        facilities: Uint32Array.of(0)
      },
      {maximumIterations}
    );
    expect(Array.from(result.costs)).toEqual([0, 1, 2]);
    expect(Array.from(result.facilityNodeCounts)).toEqual([3]);
    expect([result.converged[0], result.iterationLimitReached[0]]).toEqual([1, 0]);
    const operation = new CPUNetworkServiceAreasOperation({
      nodeCount: 3,
      facilityCapacity: 1,
      topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      facilities: 'facilities',
      maximumIterations,
      output: {
        assignments: 'assignments',
        costs: 'costs',
        facilityNodeCounts: 'counts',
        status: {converged: 'converged', iterationLimitReached: 'limited'}
      }
    });
    const rows = await operation.execute({
      inputs: getOracleCPUVectors({
        offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2, 2)},
        neighbors: {format: 'uint32', values: Uint32Array.of(1, 2)},
        weights: {format: 'float32', values: Float32Array.of(1, 1)},
        facilities: {format: 'uint32', values: Uint32Array.of(0)}
      })
    });
    expect(Array.from(rows['costs'] as Float32Array)).toEqual([0, 1, 2]);
    expect(Array.from(rows['counts'] as Uint32Array)).toEqual([3]);
    expect([...(rows['converged'] as Uint32Array), ...(rows['limited'] as Uint32Array)]).toEqual([
      1, 0
    ]);
  });

for (const activeIterations of [0, 1, 4])
  it(`R1 service-area operation binds active cost rounds (${activeIterations})`, async () => {
    // Each one-hop round reaches one more node of 0->1->2. At zero rounds
    // only the facility seed is reached; four rounds also certify the empty frontier.
    const operation = new CPUNetworkServiceAreasOperation({
      nodeCount: 3,
      facilityCapacity: 1,
      topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      facilities: 'facilities',
      activeIterations: 'active',
      maximumIterations: 4,
      maximumLabelIterations: 4,
      maximumTieIterations: 4,
      localIterations: 1,
      output: {
        assignments: 'assignments',
        costs: 'costs',
        facilityNodeCounts: 'counts',
        status: {converged: 'converged', iterationLimitReached: 'limited'}
      }
    });
    const result = await operation.execute({
      inputs: getOracleCPUVectors({
        offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2, 2)},
        neighbors: {format: 'uint32', values: Uint32Array.of(1, 2)},
        weights: {format: 'float32', values: Float32Array.of(1, 1)},
        facilities: {format: 'uint32', values: Uint32Array.of(0)},
        activeIterations: {format: 'uint32', values: Uint32Array.of(activeIterations)}
      })
    });
    expect(Array.from(result['costs'] as Float32Array)).toEqual(
      activeIterations === 0
        ? [0, Infinity, Infinity]
        : activeIterations === 1
          ? [0, 1, Infinity]
          : [0, 1, 2]
    );
    expect(Array.from(result['counts'] as Uint32Array)).toEqual([
      activeIterations === 4 ? 3 : activeIterations + 1
    ]);
    expect([
      ...(result['converged'] as Uint32Array),
      ...(result['limited'] as Uint32Array)
    ]).toEqual(activeIterations === 4 ? [1, 0] : [0, 1]);
  });

for (const maximumTieIterations of [1, 2])
  it(`R2 pending cycle tie work differs from unresolved predecessors (${maximumTieIterations})`, () => {
    // At one tie hop node 1 has parent 0 but its outgoing edge 1->0 still
    // awaits examination. Hop two consumes that item without changing a parent.
    const result = computeNetworkReachabilityOnCPU(
      {
        topology: {
          offsets: Uint32Array.of(0, 1, 2),
          neighbors: Uint32Array.of(1, 0),
          weights: Float32Array.of(0, 0)
        },
        startNodes: Uint32Array.of(0)
      },
      {maximumIterations: 4, localIterations: 1, maximumTieIterations}
    );
    expect(Array.from(result.predecessors)).toEqual([0xffffffff, 0]);
    expect([
      result.unresolvedCount[0],
      result.converged[0],
      result.iterationLimitReached[0]
    ]).toEqual(maximumTieIterations === 1 ? [0, 0, 1] : [0, 1, 0]);
  });

it('R1 service-area label and tie budgets bind their own pending frontiers', () => {
  const topology = {
    offsets: Uint32Array.of(0, 1, 2, 2),
    neighbors: Uint32Array.of(1, 2),
    weights: Float32Array.of(1, 1)
  };
  const labelLimited = computeNetworkServiceAreasOnCPU(
    {topology, facilities: Uint32Array.of(0)},
    {maximumIterations: 4, maximumLabelIterations: 1, localIterations: 1}
  );
  expect(Array.from(labelLimited.costs)).toEqual([0, 1, 2]);
  expect(Array.from(labelLimited.assignments)).toEqual([0, 0, 0xffffffff]);
  expect(Array.from(labelLimited.facilityNodeCounts)).toEqual([2]);
  expect([labelLimited.converged[0], labelLimited.iterationLimitReached[0]]).toEqual([0, 1]);
  for (const maximumTieIterations of [1, 2]) {
    const cycle = computeNetworkServiceAreasOnCPU(
      {
        topology: {
          offsets: Uint32Array.of(0, 1, 2),
          neighbors: Uint32Array.of(1, 0),
          weights: Float32Array.of(0, 0)
        },
        facilities: Uint32Array.of(0)
      },
      {maximumIterations: 4, maximumLabelIterations: 4, maximumTieIterations, localIterations: 1}
    );
    expect(Array.from(cycle.predecessors)).toEqual([0xffffffff, 0]);
    expect([cycle.converged[0], cycle.iterationLimitReached[0]]).toEqual(
      maximumTieIterations === 1 ? [0, 1] : [1, 0]
    );
  }
});

for (const scene of ['round-limit', 'tie-limit', 'complete', 'zero-active'] as const)
  it(`derives reachable unresolved and stopping counts from a zero-weight chain (${scene})`, () => {
    // Three-node zero-cost chain: one local hop leaves node 2 unreached;
    // after costs settle, one tie sweep certifies node 1 but leaves node 2's
    // predecessor unresolved. Four tie sweeps settle both predecessors.
    const result = computeNetworkReachabilityOnCPU(
      {
        topology: {
          offsets: Uint32Array.of(0, 1, 2, 2),
          neighbors: Uint32Array.of(1, 2),
          weights: Float32Array.of(0, 0)
        },
        startNodes: Uint32Array.of(0),
        activeIterations: Uint32Array.of(scene === 'zero-active' ? 0 : 4)
      },
      {
        maximumIterations: scene === 'round-limit' ? 1 : 4,
        localIterations: 1,
        maximumTieIterations: scene === 'tie-limit' ? 1 : 4
      }
    );
    expect(Array.from(result.unresolvedCount)).toEqual([scene === 'tie-limit' ? 1 : 0]);
    expect(Array.from(result.converged)).toEqual([scene === 'complete' ? 1 : 0]);
    expect(Array.from(result.iterationLimitReached)).toEqual([scene === 'complete' ? 0 : 1]);
    if (scene === 'round-limit' || scene === 'zero-active')
      expect(Array.from(result.iterationCount)).toEqual([scene === 'round-limit' ? 1 : 0]);
  });

it('derives exact service and structure completion from a finite three-node traversal', () => {
  const topology = {
    offsets: Uint32Array.of(0, 1, 2, 2),
    neighbors: Uint32Array.of(1, 2),
    weights: Float32Array.of(1, 1)
  };
  const service = computeNetworkServiceAreasOnCPU({topology, facilities: Uint32Array.of(0)});
  expect(Array.from(service.facilityNodeCounts)).toEqual([3]);
  expect([service.converged[0], service.iterationLimitReached[0]]).toEqual([1, 0]);
  const structure = computeNetworkStructureOnCPU(topology);
  expect(Array.from(structure.strongComponents)).toEqual([0, 1, 2]);
  expect([
    structure.strongComponentsStatus.converged[0],
    structure.strongComponentsStatus.iterationCount[0],
    structure.strongComponentsStatus.iterationLimitReached[0]
  ]).toEqual([1, 1, 0]);
});
