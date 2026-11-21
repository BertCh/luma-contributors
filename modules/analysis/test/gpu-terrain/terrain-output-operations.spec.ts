// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {runOperationGraph} from '../utils/mixed-execution-test-utils';
import {getTerrainOperationOracles} from './terrain-output-operation-fixtures';

for (const fixture of getTerrainOperationOracles()) {
  for (const backend of ['gpu', 'cpu'] as const) {
    it(`terrain operation oracle: ${fixture.name}/${backend}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const expected = {...fixture.expected, ...(backend === 'gpu' ? fixture.gpuExpected : {})};
      const result = await runOperationGraph(device, {
        inputs: fixture.inputs,
        operations: [fixture[backend]()],
        outputs: Object.keys(expected)
      });
      for (const [port, values] of Object.entries(expected)) {
        expect(result[port].length, port).toBe(values.length);
        for (const [index, value] of values.entries()) {
          const actual = result[port][index];
          if (Number.isNaN(value)) expect(actual, `${port}[${index}]`).toBeNaN();
          else if (!Number.isFinite(value) || result[port] instanceof Uint32Array)
            expect(actual, port).toBe(value);
          else {
            expect(Number.isFinite(actual), port).toBe(true);
            if (port === 'azimuth') {
              expect(actual, 'azimuth lower bound').toBeGreaterThanOrEqual(0);
              expect(actual, 'azimuth upper bound').toBeLessThan(360);
            }
            const difference = Math.abs(actual - value);
            const error =
              port === 'azimuth' ? Math.min(difference, Math.abs(360 - difference)) : difference;
            // Closed-form arithmetic: float32 relative 2e-5, absolute floor 1e-6;
            // solar angles instead use the external fixture's row-specific conditioning bound.
            const tolerance =
              fixture.angleBounds && (port === 'azimuth' || port === 'altitude')
                ? fixture.angleBounds[index]
                : 2e-5 * Math.max(0.05, Math.abs(value));
            expect(error, `${port}[${index}]`).toBeLessThanOrEqual(tolerance);
          }
        }
      }
    });
  }
}
