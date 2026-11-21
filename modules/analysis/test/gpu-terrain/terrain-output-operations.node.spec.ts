// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {getGPUVectorFormatInfo} from '@luma.gl/graph/gpu-data';
import {expect, it} from 'vitest';
import {getTerrainOperationOracles} from './terrain-output-operation-fixtures';

for (const fixture of getTerrainOperationOracles()) {
  it(`CPU operation independent oracle: ${fixture.name}`, async () => {
    const operation = fixture.cpu();
    const inputs: Record<string, GPUCPUVector> = {};
    for (const [port, name] of Object.entries(operation.inputs)) {
      const input = fixture.inputs[name];
      const length = input.values.length / getGPUVectorFormatInfo(input.format).components;
      inputs[port] = {
        format: input.format,
        length,
        chunks: [{values: input.values, length}],
        pack: () => input.values
      };
    }
    const result = await operation.execute({inputs});
    for (const [port, expected] of Object.entries(fixture.expected)) {
      const values = result[port];
      if (
        !(
          values instanceof Float32Array ||
          values instanceof Uint32Array ||
          values instanceof Int32Array
        )
      )
        throw new Error(`${port} must be packed`);
      expect(values.length, port).toBe(expected.length);
      for (const [index, value] of expected.entries()) {
        if (Number.isNaN(value)) expect(values[index], port).toBeNaN();
        else if (!Number.isFinite(value) || values instanceof Uint32Array)
          expect(values[index], port).toBe(value);
        else {
          expect(Number.isFinite(values[index]), port).toBe(true);
          if (port === 'azimuth') {
            expect(values[index], 'azimuth lower bound').toBeGreaterThanOrEqual(0);
            expect(values[index], 'azimuth upper bound').toBeLessThan(360);
          }
          const difference = Math.abs(values[index] - value);
          const error =
            port === 'azimuth' ? Math.min(difference, Math.abs(360 - difference)) : difference;
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
