// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUSolarPositionOperation} from '../../../src/gpu-terrain/terrain-illumination/cpu-solar-position-operation';
import {
  GPUSolarPositionOperation,
  type SolarPositionOperationProps
} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-position-operation';
import {getGPUSolarPositionParameterValues} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-position';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const refraction of [true, false]) {
  for (const testCase of ['ordinary', 'invalid', 'seam'] as const) {
    it(`solar operations agree, refraction=${refraction}, case=${testCase}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const props: SolarPositionOperationProps = {
        positions: 'positions',
        parameters: 'parameters',
        rowCount: testCase === 'seam' ? 1 : 3,
        refraction,
        output: {azimuth: 'azimuth', altitude: 'altitude', daylight: 'daylight'}
      };
      const inputs = {
        positions: {
          format: 'float32x2' as const,
          values:
            testCase === 'invalid'
              ? new Float32Array([NaN, 0, 0, 91, Infinity, 0])
              : testCase === 'seam'
                ? new Float32Array([0.48121488094329834, -30])
                : new Float32Array([-105, 40, 0, 50, 120, -30])
        },
        parameters: {
          format: 'float32' as const,
          values: getGPUSolarPositionParameterValues({
            timestamp: Date.UTC(2024, 5, 21, testCase === 'seam' ? 12 : 18)
          })
        }
      };
      const outputs = ['azimuth', 'altitude', 'daylight'];
      const gpu = await runOperationGraph(device, {
        inputs,
        operations: [new GPUSolarPositionOperation(props)],
        outputs
      });
      const cpu = await runOperationGraph(device, {
        inputs,
        operations: [new CPUSolarPositionOperation(props)],
        outputs
      });
      expect(Array.from(gpu['daylight'])).toEqual(Array.from(cpu['daylight']));
      // Range is a separate contract from circular error: 360 and 0 have zero circular error.
      for (const rows of [gpu, cpu])
        for (const azimuth of rows['azimuth']) {
          if (Number.isNaN(azimuth)) continue;
          expect(azimuth).toBeGreaterThanOrEqual(0);
          expect(azimuth).toBeLessThan(360);
        }
      if (testCase === 'seam') expect(cpu['azimuth'][0]).toBe(0);
      // Away from zenith/nadir and the daylight threshold, allow 0.03° orbital f32 error.
      for (const name of ['azimuth', 'altitude']) {
        for (let row = 0; row < props.rowCount; row++) {
          if (Number.isNaN(cpu[name][row])) expect(gpu[name][row]).toBeNaN();
          else {
            const difference = Math.abs(gpu[name][row] - cpu[name][row]);
            expect(
              name === 'azimuth' ? Math.min(difference, 360 - difference) : difference
            ).toBeLessThan(0.03);
          }
        }
      }
    });
  }
}
