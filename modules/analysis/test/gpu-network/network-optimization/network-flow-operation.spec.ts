// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUNetworkFlowOperation} from '../../../src/gpu-network/network-optimization/cpu-network-flow-operation';
import {GPU_NETWORK_FLOW_WORD} from '../../../src/gpu-network/network-optimization/gpu-network-flow';
import {
  GPUNetworkFlowOperation,
  type NetworkFlowOperationProps
} from '../../../src/gpu-network/network-optimization/gpu-network-flow-operation';
import {runOperationGraph, type OperationGraphRows} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const scene of [
  'reroute',
  'finite-request',
  'zero-capacity',
  'invalid',
  'large-cost',
  'uncosted'
] as const) {
  it(`flow twins compare every output (${scene})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const large = scene === 'large-cost';
    const props: NetworkFlowOperationProps = {
      id: 'flow',
      nodeCount: large ? 2 : 6,
      edgeCount: large ? 1 : 8,
      edges: 'edges',
      capacities: 'capacities',
      costs: scene === 'uncosted' ? undefined : 'costs',
      parameters: 'parameters',
      maximumAugmentations: 8,
      output: {flow: 'flow', cut: 'cut', summary: 'summary', exactCost: 'exact'}
    };
    const inputs = {
      edges: {
        format: 'uint32x2' as const,
        values: large
          ? Uint32Array.of(0, 1)
          : Uint32Array.of(
              0,
              scene === 'invalid' ? 99 : 1,
              0,
              2,
              1,
              3,
              1,
              4,
              2,
              3,
              2,
              4,
              3,
              5,
              4,
              5
            )
      },
      capacities: {
        format: 'uint32' as const,
        values: large
          ? Uint32Array.of(0xffffff)
          : new Uint32Array(8).fill(scene === 'zero-capacity' ? 0 : 1)
      },
      costs: {
        format: 'uint32' as const,
        values: large ? Uint32Array.of(32767) : Uint32Array.of(0, 0, 1, 2, 2, 100, 0, 0)
      },
      parameters: {
        format: 'uint32' as const,
        values: Uint32Array.of(0, large ? 1 : 5, scene === 'finite-request' ? 1 : 0xffffffff)
      }
    };
    const outputs = ['flow', 'cut', 'summary', 'exact'];
    const results: Record<string, Record<string, OperationGraphRows>> = {};
    for (const location of ['gpu', 'cpu'])
      results[location] = await runOperationGraph(device, {
        inputs,
        operations: [
          location === 'gpu'
            ? new GPUNetworkFlowOperation(props)
            : new CPUNetworkFlowOperation(props)
        ],
        outputs
      });
    for (const port of outputs) {
      expect(results['gpu'][port].length).toBe(results['cpu'][port].length);
      for (let row = 0; row < results['cpu'][port].length; row++) {
        const expected = results['cpu'][port][row];
        const actual = results['gpu'][port][row];
        if (port === 'summary' && row === GPU_NETWORK_FLOW_WORD.cost) {
          // Only the packed float32 objective permits rounding differences (relative 2e-5).
          const expectedCost = new Float32Array(Uint32Array.of(expected).buffer)[0];
          const actualCost = new Float32Array(Uint32Array.of(actual).buffer)[0];
          expect(Math.abs(actualCost - expectedCost)).toBeLessThanOrEqual(
            2e-5 * Math.abs(expectedCost)
          );
        } else expect(actual).toBe(expected);
      }
    }
    if (scene === 'reroute')
      expect(Array.from(results['cpu']['flow'])).toEqual([1, 1, 0, 1, 1, 0, 1, 1]);
    if (scene === 'large-cost')
      expect(BigInt(results['cpu']['exact'][0]) + (BigInt(results['cpu']['exact'][1]) << 32n)).toBe(
        0xffffffn * 32767n
      );
  });
}
