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

for (const mode of ['reference', 'scalable'] as const) {
  it(`point routing twins charge complete per-turn penalties on partial destinations (${mode})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: NetworkPointRoutingOperationProps = {
      mode,
      edgeCount: 3,
      queryCount: 2,
      turns: 'turns',
      turnPenalties: 'penalties',
      costs: 'baseCosts',
      queries: 'queries',
      fractions: 'fractions',
      maximumRelaxations: 8,
      output: {costs: 'costs', status: 'status', predecessorEdges: 'parents'}
    };
    for (const Operation of [GPUNetworkPointRoutingOperation, CPUNetworkPointRoutingOperation]) {
      const rows = await runOperationGraph(device, {
        inputs: {
          turns: {format: 'uint32x2', values: Uint32Array.of(0, 2, 1, 2)},
          penalties: {format: 'float32', values: Float32Array.of(7, 1)},
          baseCosts: {format: 'float32', values: Float32Array.of(2, 2, 10)},
          queries: {format: 'uint32x2', values: Uint32Array.of(0, 2, 1, 2)},
          fractions: {format: 'float32x2', values: Float32Array.of(0.5, 0.25, 0.5, 0.25)}
        },
        operations: [new Operation(props)],
        outputs: ['costs', 'status', 'parents']
      });
      expect(rows['costs']).toEqual(Float32Array.of(10.5, 4.5));
      expect(rows['status']).toEqual(Uint32Array.of(0, 0));
      expect(rows['parents']).toEqual(
        Uint32Array.of(0xffffffff, 0xffffffff, 0, 0xffffffff, 0xffffffff, 1)
      );
    }
  });
}

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

it('lowers explicit reference mode with maximumRelaxations 2048 and preserves expected same-edge results', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const input = getPointRoutingInput();
  const rows = await runOperationGraph(device, {
    inputs: {
      turns: {format: 'uint32x2', values: input.turns},
      edgeCosts: {format: 'float32', values: input.costs},
      queries: {format: 'uint32x2', values: input.queries},
      fractions: {format: 'float32x2', values: input.fractions}
    },
    operations: [
      new GPUNetworkPointRoutingOperation({
        edgeCount: 6,
        queryCount: 5,
        turns: 'turns',
        costs: 'edgeCosts',
        queries: 'queries',
        fractions: 'fractions',
        mode: 'reference',
        maximumRelaxations: 2048,
        output: {costs: 'costs', status: 'status', predecessorEdges: 'parents'}
      })
    ],
    outputs: ['costs', 'status', 'parents']
  });
  expect(Array.from(rows['costs'])).toEqual([11, 12, 4, 19, Infinity]);
  expect(Array.from(rows['status'])).toEqual([0, 0, 0, 0, 0]);
  expect(rows['parents'][6]).toBe(2);
  expect(rows['parents'][12]).toBe(0xffffffff);
});
