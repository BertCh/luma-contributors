// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUNetworkCostMatrixOperation} from '../../../src/gpu-network/network-accessibility/cpu-network-cost-matrix-operation';
import {
  GPUNetworkCostMatrixOperation,
  type NetworkCostMatrixOperationProps
} from '../../../src/gpu-network/network-accessibility/gpu-network-cost-matrix-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {getRoutingTopology} from '../network-reachability/network-routing-operation-test-utils';

for (const empty of [false, true])
  it(`cost matrix twins compare every output (no valid starts: ${empty})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const topology = getRoutingTopology();
    const props: NetworkCostMatrixOperationProps = {
      nodeCount: 9,
      rowCount: 3,
      targetNodes: 'targets',
      targetCount: 6,
      topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      startNodes: 'starts',
      startCosts: 'startCosts',
      startRows: 'startRows',
      costLimit: 'cutoff',
      laneCount: 2,
      maximumIterations: 8,
      localIterations: 4,
      output: {
        costs: 'costs',
        status: {converged: 'converged', iterationLimitReached: 'limitReached'}
      }
    };
    const inputs = {
      offsets: {format: 'uint32' as const, values: topology.offsets},
      neighbors: {format: 'uint32' as const, values: topology.neighbors},
      weights: {format: 'float32' as const, values: topology.weights},
      starts: {
        format: 'uint32' as const,
        values: empty ? Uint32Array.of(99, 99, 99, 99, 99) : Uint32Array.of(0, 7, 4, 2, 99)
      },
      startCosts: {format: 'float32' as const, values: Float32Array.of(0.5, 2, 3, 0.75, 0)},
      startRows: {format: 'uint32' as const, values: Uint32Array.of(0, 0, 1, 1, 2)},
      cutoff: {format: 'float32' as const, values: Float32Array.of(6)},
      targets: {format: 'uint32' as const, values: Uint32Array.of(5, 1, 2, 8, 5, 99)}
    };
    const outputs = ['costs', 'converged', 'limitReached'];
    const gpuRows = await runOperationGraph(device, {
      inputs,
      operations: [new GPUNetworkCostMatrixOperation(props)],
      outputs
    });
    const cpuRows = await runOperationGraph(device, {
      inputs,
      operations: [new CPUNetworkCostMatrixOperation(props)],
      outputs
    });
    // Binary-fraction costs are exactly representable throughout this fixture: zero tolerance.
    for (const output of outputs)
      for (let index = 0; index < cpuRows[output].length; index++)
        expect(gpuRows[output][index] === cpuRows[output][index]).toBe(true);
    expect(cpuRows['costs'].some(Number.isFinite)).toBe(!empty);
  });
