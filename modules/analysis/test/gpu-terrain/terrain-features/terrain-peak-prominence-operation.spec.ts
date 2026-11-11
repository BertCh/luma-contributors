// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUTerrainPeakProminenceOperation} from '../../../src/gpu-terrain/terrain-features/cpu-terrain-peak-prominence-operation';
import {
  GPUTerrainPeakProminenceOperation,
  type TerrainPeakProminenceOperationProps
} from '../../../src/gpu-terrain/terrain-features/gpu-terrain-peak-prominence-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('both peak-prominence operations reject uint32 nodata -1 on the one-cell band [10]', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: TerrainPeakProminenceOperationProps = {
    id: 'native-nodata',
    grid: {width: 1, height: 1, pixelInterpretation: 'area', affine: [1, 0, 0, 0, -1, 0]},
    elevation: {
      id: 'elevation',
      format: 'uint32',
      storage: {kind: 'buffer', values: 'elevation'},
      noDataValue: -1
    },
    output: {prominence: 'prominence'}
  };
  for (const Operation of [GPUTerrainPeakProminenceOperation, CPUTerrainPeakProminenceOperation]) {
    await expect(async () =>
      runOperationGraph(device, {
        inputs: {elevation: {format: 'uint32', values: new Uint32Array([10])}},
        operations: [new Operation(props)],
        outputs: ['prominence']
      })
    ).rejects.toThrow(/nodata.*uint32/i);
  }
});

for (const testCase of [
  'peaks',
  'plateau',
  'masked',
  'integer-calibration',
  'negative-zero'
] as const) {
  it(`peak-prominence operations agree on ${testCase}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const integer = testCase === 'integer-calibration';
    const cellCount = testCase === 'negative-zero' ? 2 : 5;
    const props: TerrainPeakProminenceOperationProps = {
      grid: {width: cellCount, height: 1, pixelInterpretation: 'area', affine: [1, 0, 0, 0, -1, 0]},
      elevation: {
        id: 'elevation',
        format: integer ? 'sint32' : 'float32',
        storage: {kind: 'buffer', values: 'elevation'},
        validity: 'validity',
        scale: integer ? 2 : undefined,
        offset: testCase === 'negative-zero' ? -0 : integer ? 1 : undefined
      },
      output: {prominence: 'prominence', keySaddle: 'saddle', parentPeak: 'parent'}
    };
    const inputs = {
      elevation: {
        format: props.elevation.format,
        values: integer
          ? new Int32Array([10, 2, 8, 3, 9])
          : new Float32Array(
              testCase === 'negative-zero'
                ? [-0, -0]
                : testCase === 'plateau'
                  ? [5, 5, 5, 5, 5]
                  : [10, 2, 8, 3, 9]
            )
      },
      validity: {
        format: 'uint32' as const,
        values:
          testCase === 'masked'
            ? new Uint32Array([1, 0, 1, 0, 1])
            : new Uint32Array(cellCount).fill(1)
      }
    };
    const outputs = ['prominence', 'saddle', 'parent'];
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUTerrainPeakProminenceOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUTerrainPeakProminenceOperation(props)],
      outputs
    });
    expect(Array.from(gpu['parent'])).toEqual(Array.from(cpu['parent']));
    if (testCase === 'negative-zero') {
      for (const rows of [gpu, cpu]) {
        expect(rows['saddle'][0]).toBe(-Infinity);
        // Compare numerically: the test GPU can publish -0 despite source normalization.
        expect(rows['saddle'][1] === 0).toBe(true);
        expect(Array.from(rows['parent'])).toEqual([0xffffffff, 0]);
      }
      expect(Object.is(cpu['saddle'][1], -0)).toBe(false);
    }
    // Integer input heights/calibration are exactly representable here. Compare infinities and
    // NaNs explicitly, and allow 1e-6 absolute for every finite float output.
    for (const name of ['prominence', 'saddle'])
      for (let cell = 0; cell < cellCount; cell++) {
        const expected = cpu[name][cell];
        if (Number.isNaN(expected)) expect(gpu[name][cell]).toBeNaN();
        else if (!Number.isFinite(expected)) expect(gpu[name][cell]).toBe(expected);
        else expect(Math.abs(gpu[name][cell] - expected)).toBeLessThanOrEqual(1e-6);
      }
  });
}
