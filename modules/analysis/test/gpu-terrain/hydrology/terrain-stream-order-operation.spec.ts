// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUTerrainStreamOrderOperation} from '../../../src/gpu-terrain/hydrology/cpu-terrain-stream-order-operation';
import {
  GPUTerrainStreamOrderOperation,
  type TerrainStreamOrderOperationProps
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-stream-order-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const testCase of ['tree', 'no-streams', 'unknown', 'cycle'] as const) {
  it(`stream-order operations agree on ${testCase}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: TerrainStreamOrderOperationProps = {
      grid: {width: 3, height: 2, pixelInterpretation: 'area', affine: [1, 0, 0, 0, -1, 0]},
      flowDirections: 'directions',
      streams: 'streams',
      output: {
        streamOrder: 'order',
        status: {
          converged: 'converged',
          iterationLimitReached: 'iterationLimitReached',
          iterationCount: 'iterationCount'
        }
      }
    };
    // 0 and 2 merge at 4; the remaining cells have no stream membership.
    const inputs = {
      directions: {
        format: 'uint32' as const,
        values: new Uint32Array(testCase === 'cycle' ? [1, 16, 0, 0, 0, 0] : [2, 0, 8, 0, 0, 0])
      },
      streams: {
        format: 'uint32' as const,
        values: new Uint32Array(
          testCase === 'no-streams'
            ? [0, 0, 0, 0, 0, 0]
            : testCase === 'cycle'
              ? [1, 1, 0, 0, 0, 0]
              : [testCase === 'unknown' ? 0xffffffff : 1, 0, 1, 0, 1, 0]
        )
      }
    };
    const outputs = ['order', 'converged', 'iterationLimitReached', 'iterationCount'];
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUTerrainStreamOrderOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUTerrainStreamOrderOperation(props)],
      outputs
    });
    for (const name of ['order', 'converged', 'iterationLimitReached'])
      expect(Array.from(gpu[name])).toEqual(Array.from(cpu[name]));
    // CPU executes zero GPU rounds; tour ranking uses at most ceil(log2(2 * cells)) + 1.
    expect(cpu['iterationCount'][0]).toBe(0);
    expect(Math.abs(gpu['iterationCount'][0] - cpu['iterationCount'][0])).toBeLessThanOrEqual(
      Math.ceil(Math.log2(12)) + 1
    );
  });
}
