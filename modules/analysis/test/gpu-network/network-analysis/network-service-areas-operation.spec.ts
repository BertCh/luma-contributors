// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUNetworkServiceAreasOperation} from '../../../src/gpu-network/network-analysis/cpu-network-service-areas-operation';
import {
  GPUNetworkServiceAreasOperation,
  type NetworkServiceAreasOperationProps
} from '../../../src/gpu-network/network-analysis/gpu-network-service-areas-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  getOverlappingRoutingTopology,
  getRoutingTopology
} from '../network-reachability/network-routing-operation-test-utils';

for (const empty of [false, true])
  it(`service area twins compare all outputs (inactive facilities: ${empty})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const topology = getRoutingTopology();
    const props: NetworkServiceAreasOperationProps = {
      nodeCount: 9,
      facilityCapacity: 4,
      topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      facilities: 'facilities',
      facilityCosts: 'facilityCosts',
      facilityCount: 'facilityCount',
      costLimit: 'cutoff',
      maximumIterations: 8,
      maximumLabelIterations: 8,
      maximumTieIterations: 8,
      localIterations: 4,
      output: {
        assignments: 'assignments',
        costs: 'costs',
        predecessors: 'parents',
        facilityNodeCounts: 'counts',
        facilityCostSums: 'sums',
        status: {converged: 'converged', iterationLimitReached: 'limitReached'}
      }
    };
    const inputs = {
      offsets: {format: 'uint32' as const, values: topology.offsets},
      neighbors: {format: 'uint32' as const, values: topology.neighbors},
      weights: {format: 'float32' as const, values: topology.weights},
      facilities: {format: 'uint32' as const, values: Uint32Array.of(7, 0, 4, 99)},
      facilityCosts: {format: 'float32' as const, values: Float32Array.of(2, 0.5, 1.75, 0)},
      facilityCount: {format: 'uint32' as const, values: Uint32Array.of(empty ? 0 : 3)},
      cutoff: {format: 'float32' as const, values: Float32Array.of(6)}
    };
    const outputs = [
      'assignments',
      'costs',
      'parents',
      'counts',
      'sums',
      'converged',
      'limitReached'
    ];
    const gpuRows = await runOperationGraph(device, {
      inputs,
      operations: [new GPUNetworkServiceAreasOperation(props)],
      outputs
    });
    const cpuRows = await runOperationGraph(device, {
      inputs,
      operations: [new CPUNetworkServiceAreasOperation(props)],
      outputs
    });
    // Exact binary-fraction arithmetic, including aggregation: zero tolerance for all outputs.
    for (const output of outputs)
      for (let index = 0; index < cpuRows[output].length; index++)
        expect(gpuRows[output][index] === cpuRows[output][index]).toBe(true);
    expect(Array.from(cpuRows['counts'])).toEqual(empty ? [0, 0, 0, 0] : [4, 4, 0, 0]);
  });

it('reuses tight slot flags across overlapping CSR rows on both backends', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const topology = getOverlappingRoutingTopology();
  const props: NetworkServiceAreasOperationProps = {
    nodeCount: 3,
    facilityCapacity: 2,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    facilities: 'facilities',
    facilityCosts: 'facilityCosts',
    maximumIterations: 8,
    maximumLabelIterations: 8,
    maximumTieIterations: 8,
    localIterations: 4,
    output: {
      assignments: 'assignments',
      costs: 'costs',
      predecessors: 'parents',
      facilityNodeCounts: 'counts',
      facilityCostSums: 'sums',
      status: {converged: 'converged', iterationLimitReached: 'limitReached'}
    }
  };
  const inputs = {
    offsets: {format: 'uint32' as const, values: topology.offsets},
    neighbors: {format: 'uint32' as const, values: topology.neighbors},
    weights: {format: 'float32' as const, values: topology.weights},
    facilities: {format: 'uint32' as const, values: Uint32Array.of(0, 2)},
    facilityCosts: {format: 'float32' as const, values: Float32Array.of(1, 0)}
  };
  // Row 2 marks slot 0 tight; row 0 then propagates facility 0 over that flag.
  // Neither edge into node 1 is both tight and within its final facility label.
  const expected: Record<string, number[]> = {
    assignments: [0, 0, 1],
    costs: [1, 0, 0],
    parents: [0xffffffff, 0xffffffff, 0xffffffff],
    counts: [2, 1],
    sums: [1, 0],
    converged: [1],
    limitReached: [0]
  };
  const outputs = Object.keys(expected);
  const gpuRows = await runOperationGraph(device, {
    inputs,
    operations: [new GPUNetworkServiceAreasOperation(props)],
    outputs
  });
  const cpuRows = await runOperationGraph(device, {
    inputs,
    operations: [new CPUNetworkServiceAreasOperation(props)],
    outputs
  });
  // All values are exact integers; numerical equality ignores any zero sign.
  for (const output of outputs) {
    expect(gpuRows[output].length).toBe(expected[output].length);
    expect(cpuRows[output].length).toBe(expected[output].length);
    for (let index = 0; index < expected[output].length; index++) {
      expect(gpuRows[output][index] === expected[output][index]).toBe(true);
      expect(cpuRows[output][index] === expected[output][index]).toBe(true);
    }
  }
});
