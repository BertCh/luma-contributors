// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUBrownianBridgeUtilizationOperation} from '../../../src/gpu-spatial-analysis/trajectory-home-range/cpu-brownian-bridge-utilization-operation';
import {
  GPUBrownianBridgeUtilizationOperation,
  type BrownianBridgeUtilizationOperationProps
} from '../../../src/gpu-spatial-analysis/trajectory-home-range/gpu-brownian-bridge-utilization-operation';
import {getGPUBrownianBridgeUtilizationParameterValues} from '../../../src/gpu-spatial-analysis/trajectory-home-range/gpu-brownian-bridge-utilization';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const scenario of ['moving', 'zero-variance', 'no-duration', 'words'] as const) {
  it(`bridge operations compare every density cell: ${scenario}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: BrownianBridgeUtilizationOperationProps = {
      positions: 'positions',
      timestamps: 'timestamps',
      trackOffsets: 'trackOffsets',
      motionVariances: 'motionVariances',
      locationErrors: 'locationErrors',
      parameters: 'parameters',
      rowCount: 5,
      gridSize: [4, 3],
      subStepCount: 15,
      timestampFormat: scenario === 'words' ? 'uint32x2' : 'float32',
      output: {density: 'density'}
    };
    const inputs = {
      positions: {
        format: 'float32x2' as const,
        values: new Float32Array([0, 0, 1, 0, 2, 1, -1, 2, 0, 1])
      },
      timestamps: {
        format: props.timestampFormat!,
        values:
          scenario === 'words'
            ? new Uint32Array([0xfffffffe, 7, 0xffffffff, 7, 1, 8, 20, 8, 23, 8])
            : new Float32Array(scenario === 'no-duration' ? [0, 0, 0, 10, 10] : [0, 1, 3, 10, 13])
      },
      trackOffsets: {format: 'uint32' as const, values: new Uint32Array([0, 3, 3, 5])},
      motionVariances: {
        format: 'float32' as const,
        values: new Float32Array(scenario === 'zero-variance' ? [0, 0, 0, 0, 0] : [1, 2, 0, 3, 0])
      },
      locationErrors: {
        format: 'float32' as const,
        values: new Float32Array(5).fill(scenario === 'zero-variance' ? 0 : 0.5)
      },
      parameters: {
        format: 'float32' as const,
        values: getGPUBrownianBridgeUtilizationParameterValues({extent: [-2, -2, 3, 3]})
      }
    };
    const outputs = ['density'];
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUBrownianBridgeUtilizationOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUBrownianBridgeUtilizationOperation(props)],
      outputs
    });
    expect(gpu['density'].length).toBe(12);
    expect(cpu['density'].length).toBe(12);
    for (let cell = 0; cell < 12; cell++) {
      // Absolute 1e-7 plus relative 2e-5 covers f32 accumulation and Gaussian evaluation.
      expect(Math.abs(gpu['density'][cell] - cpu['density'][cell])).toBeLessThanOrEqual(
        1e-7 + 2e-5 * Math.abs(cpu['density'][cell])
      );
    }
  });
}

it('bridge review regression compares the stationary density: clamped-offset', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: BrownianBridgeUtilizationOperationProps = {
    positions: 'positions',
    timestamps: 'timestamps',
    trackOffsets: 'trackOffsets',
    motionVariances: 'motionVariances',
    locationErrors: 'locationErrors',
    parameters: 'parameters',
    rowCount: 2,
    gridSize: [1, 1],
    subStepCount: 15,
    output: {density: 'density'}
  };
  const inputs = {
    positions: {format: 'float32x2' as const, values: new Float32Array(4)},
    timestamps: {format: 'float32' as const, values: new Float32Array([0, 1])},
    trackOffsets: {
      format: 'uint32' as const,
      values: new Uint32Array([0, 3])
    },
    motionVariances: {
      format: 'float32' as const,
      values: new Float32Array([2, 2])
    },
    locationErrors: {format: 'float32' as const, values: new Float32Array([1, 1])},
    parameters: {format: 'float32' as const, values: new Float32Array([-1, -1, 1, 1])}
  };
  const outputs = ['density'];
  const gpu = await runOperationGraph(device, {
    inputs,
    operations: [new GPUBrownianBridgeUtilizationOperation(props)],
    outputs
  });
  const cpu = await runOperationGraph(device, {
    inputs,
    operations: [new CPUBrownianBridgeUtilizationOperation(props)],
    outputs
  });
  const expected = Math.fround(1 / (2 * Math.PI));
  expect(Array.from(cpu['density'])).toEqual([expected]);
  expect(gpu['density'].length).toBe(1);
  // Use the ordinary density tolerance, absolute 1e-7 plus relative 2e-5.
  expect(Math.abs(gpu['density'][0] - cpu['density'][0])).toBeLessThanOrEqual(
    1e-7 + 2e-5 * Math.abs(expected)
  );
});
