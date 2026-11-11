// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUTerrainWatershedsOperation} from '../../../src/gpu-terrain/hydrology/cpu-terrain-watersheds-operation';
import {
  GPUTerrainWatershedsOperation,
  type TerrainWatershedsOperationProps
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-watersheds-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const testCase of ['basins', 'pour-points', 'unresolved'] as const) {
  it(`watershed operations agree on ${testCase}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: TerrainWatershedsOperationProps = {
      grid: {width: 5, height: 1, pixelInterpretation: 'area', affine: [1, 0, 0, 0, -1, 0]},
      flowDirections: 'directions',
      pourPoints: testCase === 'pour-points' ? 'points' : undefined,
      output: {
        labels: 'labels',
        status: {
          converged: 'converged',
          iterationLimitReached: 'iterationLimitReached',
          iterationCount: 'iterationCount'
        }
      }
    };
    const inputs = {
      directions: {
        format: 'uint32' as const,
        values: new Uint32Array(
          testCase === 'unresolved' ? [1, 0xfffffffe, 0xffffffff, 1, 0] : [1, 1, 0, 16, 16]
        )
      },
      ...(props.pourPoints
        ? {points: {format: 'uint32' as const, values: new Uint32Array([2, 1, 1, 99])}}
        : {})
    };
    const outputs = ['labels', 'converged', 'iterationLimitReached', 'iterationCount'];
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUTerrainWatershedsOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUTerrainWatershedsOperation(props)],
      outputs
    });
    for (const name of ['labels', 'converged', 'iterationLimitReached'])
      expect(Array.from(gpu[name])).toEqual(Array.from(cpu[name]));
    // Iteration counts have different documented meanings: CPU runs no GPU rounds; the GPU's
    // in-place schedule can take any count within its budget. The allowed count difference is 32.
    expect(cpu['iterationCount'][0]).toBe(0);
    expect(gpu['iterationCount'][0]).toBeGreaterThanOrEqual(1);
    expect(Math.abs(gpu['iterationCount'][0] - cpu['iterationCount'][0])).toBeLessThanOrEqual(32);
  });
}
