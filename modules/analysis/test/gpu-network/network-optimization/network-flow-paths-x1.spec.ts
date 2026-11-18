// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUNetworkFlowOperation} from '../../../src/gpu-network/network-optimization/cpu-network-flow-operation';
import {GPUNetworkFlowOperation} from '../../../src/gpu-network/network-optimization/gpu-network-flow-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  getEmptyFlowExpectedRows,
  getEmptyFlowOperationProps
} from './network-flow-paths-x1.test-utils';

for (const solver of ['reference', 'parallel'] as const)
  it(`network-recipes:X7 ${solver} operations execute a zero-edge unit request with paths`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const expected = getEmptyFlowExpectedRows();
    for (const Operation of [GPUNetworkFlowOperation, CPUNetworkFlowOperation]) {
      const rows = await runOperationGraph(device, {
        inputs: {
          edges: {format: 'uint32x2', values: new Uint32Array(0)},
          capacities: {format: 'uint32', values: new Uint32Array(0)},
          parameters: {format: 'uint32', values: Uint32Array.of(0, 1, 1)}
        },
        operations: [new Operation(getEmptyFlowOperationProps(solver))],
        outputs: Object.keys(expected)
      });
      for (const [name, values] of Object.entries(expected))
        expect(rows[name], name).toEqual(values);
    }
  });
