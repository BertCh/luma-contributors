// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUNetworkCostTableOperation} from '../../../src/gpu-network/network-cost-table/cpu-network-cost-table-operation';
import {
  GPUNetworkCostTableOperation,
  type NetworkCostTableOperationProps
} from '../../../src/gpu-network/network-cost-table/gpu-network-cost-table-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  enumerateRoutingCosts,
  getRoutingTopology
} from '../network-reachability/network-routing-operation-test-utils';

for (const overflow of [false, true])
  it(`cost table twins compare every output (overflow: ${overflow})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const topology = getRoutingTopology();
    const requiredCount = Array.from({length: 9}, (_, source) =>
      enumerateRoutingCosts(topology, [{node: source, cost: 0}], 5)
    ).reduce((count, distances) => count + distances.filter(Number.isFinite).length, 0);
    const props: NetworkCostTableOperationProps = {
      nodeCount: 9,
      capacity: overflow ? 3 : requiredCount,
      topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      costLimit: 'cutoff',
      laneCount: 2,
      maximumIterations: 8,
      localIterations: 4,
      output: {
        offsets: 'outputOffsets',
        targets: 'targets',
        costs: 'costs',
        status: {
          overflow: 'overflow',
          requiredCount: 'requiredCount',
          converged: 'converged',
          iterationLimitReached: 'limitReached'
        }
      }
    };
    const inputs = {
      offsets: {format: 'uint32' as const, values: topology.offsets},
      neighbors: {format: 'uint32' as const, values: topology.neighbors},
      weights: {format: 'float32' as const, values: topology.weights},
      cutoff: {format: 'float32' as const, values: Float32Array.of(5)}
    };
    const outputs = [
      'outputOffsets',
      'targets',
      'costs',
      'overflow',
      'requiredCount',
      'converged',
      'limitReached'
    ];
    const gpuRows = await runOperationGraph(device, {
      inputs,
      operations: [new GPUNetworkCostTableOperation(props)],
      outputs
    });
    const cpuRows = await runOperationGraph(device, {
      inputs,
      operations: [new CPUNetworkCostTableOperation(props)],
      outputs
    });
    // Binary fractions are exact at every path addition; CSR indices and all status words are exact.
    for (const output of outputs)
      for (let index = 0; index < cpuRows[output].length; index++)
        expect(gpuRows[output][index] === cpuRows[output][index]).toBe(true);
    expect(cpuRows['limitReached'][0]).toBe(0);
    expect(cpuRows['overflow'][0]).toBe(Number(overflow));
    expect(cpuRows['requiredCount'][0]).toBe(requiredCount);
  });
