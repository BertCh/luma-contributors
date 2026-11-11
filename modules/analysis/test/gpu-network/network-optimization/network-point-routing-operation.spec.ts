// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUNetworkPointRoutingOperation} from '../../../src/gpu-network/network-optimization/cpu-network-point-routing-operation';
import {
  GPUNetworkPointRoutingOperation,
  type NetworkPointRoutingOperationProps
} from '../../../src/gpu-network/network-optimization/gpu-network-point-routing-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {getPointRoutingInput} from './network-point-routing-operation-test-utils';

for (const invalid of [false, true])
  it(`point routing twins compare every output (invalid endpoints: ${invalid})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = getPointRoutingInput();
    const props: NetworkPointRoutingOperationProps = {
      edgeCount: 6,
      queryCount: 5,
      turns: 'turns',
      costs: 'edgeCosts',
      queries: 'queries',
      fractions: 'fractions',
      maximumRelaxations: 8,
      output: {costs: 'costs', status: 'status', predecessorEdges: 'parents'}
    };
    const inputs = {
      edgeCosts: {format: 'float32' as const, values: input.costs},
      turns: {format: 'uint32x2' as const, values: input.turns},
      queries: {
        format: 'uint32x2' as const,
        values: invalid ? new Uint32Array(10).fill(99) : input.queries
      },
      fractions: {format: 'float32x2' as const, values: input.fractions}
    };
    const outputs = ['costs', 'status', 'parents'];
    const gpuRows = await runOperationGraph(device, {
      inputs,
      operations: [new GPUNetworkPointRoutingOperation(props)],
      outputs
    });
    const cpuRows = await runOperationGraph(device, {
      inputs,
      operations: [new CPUNetworkPointRoutingOperation(props)],
      outputs
    });
    // Integer costs and binary fractions are exact throughout. Numeric equality ignores zero sign.
    for (const output of outputs)
      for (let index = 0; index < cpuRows[output].length; index++)
        expect(gpuRows[output][index] === cpuRows[output][index]).toBe(true);
    expect(Array.from(cpuRows['costs'])).toEqual(
      invalid ? Array(5).fill(Infinity) : [11, 12, 4, 19, Infinity]
    );
  });

it('checks the documented status difference when a full-edge float32 sum overflows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const maximumFloat32 = new Float32Array(Uint32Array.of(0x7f7fffff).buffer)[0];
  const props: NetworkPointRoutingOperationProps = {
    edgeCount: 2,
    queryCount: 1,
    turns: 'turns',
    costs: 'edgeCosts',
    queries: 'queries',
    fractions: 'fractions',
    maximumRelaxations: 8,
    output: {costs: 'costs', status: 'status', predecessorEdges: 'parents'}
  };
  const inputs = {
    edgeCosts: {
      format: 'float32' as const,
      values: Float32Array.of(maximumFloat32, maximumFloat32)
    },
    turns: {format: 'uint32x2' as const, values: Uint32Array.of(0, 1)},
    queries: {format: 'uint32x2' as const, values: Uint32Array.of(0, 1)},
    fractions: {format: 'float32x2' as const, values: Float32Array.of(0, 0)}
  };
  const outputs = ['costs', 'status', 'parents'];
  const gpuRows = await runOperationGraph(device, {
    inputs,
    operations: [new GPUNetworkPointRoutingOperation(props)],
    outputs
  });
  const cpuRows = await runOperationGraph(device, {
    inputs,
    operations: [new CPUNetworkPointRoutingOperation(props)],
    outputs
  });
  // These exact backend-specific expectations test the kernel prediction: the GPU returns
  // before publishing its finite partial cost M, whereas the float64 twin completes the search.
  const gpuExpected: Record<string, number[]> = {
    costs: [Infinity],
    status: [1],
    parents: [0xffffffff, 0xffffffff]
  };
  const cpuExpected: Record<string, number[]> = {
    costs: [2 ** 128 - 2 ** 104],
    status: [0],
    parents: [0xffffffff, 0]
  };
  for (const output of outputs) {
    expect(gpuRows[output].length).toBe(gpuExpected[output].length);
    expect(cpuRows[output].length).toBe(cpuExpected[output].length);
    for (let index = 0; index < gpuExpected[output].length; index++) {
      expect(gpuRows[output][index] === gpuExpected[output][index]).toBe(true);
      expect(cpuRows[output][index] === cpuExpected[output][index]).toBe(true);
    }
  }
});
