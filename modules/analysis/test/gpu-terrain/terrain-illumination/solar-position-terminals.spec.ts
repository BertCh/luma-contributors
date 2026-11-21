// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {CPUSolarPositionOperation} from '../../../src/gpu-terrain/terrain-illumination/cpu-solar-position-operation';
import {
  GPUSolarPositionOperation,
  type SolarPositionOperationProps
} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-position-operation';
import {getGPUSolarPositionParameterValues} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-position';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {expectSolarAngle} from '../terrain-features/terrain-terminal-assertions';

const SELECTIONS: SolarPositionOperationProps['output'][] = [
  {azimuth: 'azimuth'},
  {altitude: 'altitude'},
  {daylight: 'daylight'},
  {azimuth: 'azimuth', altitude: 'altitude', daylight: 'daylight'}
];

for (const Operation of [CPUSolarPositionOperation, GPUSolarPositionOperation]) {
  for (const output of SELECTIONS) {
    it(`TSC-03 ${Operation.name} writes selected terminals: ${Object.keys(output).join(',')}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rows = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: Float32Array.of(-105.1786, 39.742476)},
          parameters: {
            format: 'float32',
            values: getGPUSolarPositionParameterValues({
              timestamp: Date.UTC(2003, 9, 17, 19, 30, 30)
            })
          }
        },
        operations: [
          new Operation({positions: 'positions', parameters: 'parameters', rowCount: 1, output})
        ],
        outputs: Object.values(output)
      });
      // NREL/TP-560-34302 Appendix A: azimuth 194.34024°, zenith 50.11162°.
      // NOAA and standard refraction differ from SPA; allow 0.05° including GPU arithmetic.
      for (const [name, expected] of [
        ['azimuth', 194.34024],
        ['altitude', 39.88838]
      ] as const) {
        if (output[name] === undefined) continue;
        expectSolarAngle(rows[name], expected);
      }
      if (output.daylight !== undefined) expect(Array.from(rows['daylight'])).toEqual([1]);
    });
  }
}
