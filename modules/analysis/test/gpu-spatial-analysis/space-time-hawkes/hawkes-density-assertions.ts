// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {TWO_EVENT_DENSITY} from './two-event-density-fixture';

/** Check the independent closed form through CPU and GPU public result containers. */
export function expectHawkesDensityOutputs(
  actual: Record<string, unknown>,
  tolerance = 2e-7
): void {
  for (const [port, expected] of Object.entries(TWO_EVENT_DENSITY)) {
    const values = actual[port];
    if (
      !(Array.isArray(values) || values instanceof Float32Array || values instanceof Uint32Array)
    ) {
      throw new Error(`${port} must be a numeric array`);
    }
    expect(values.length, port).toBe(expected.length);
    expected.forEach((value, index) => {
      expect(Number.isFinite(values[index]), port).toBe(true);
      const floatPort = [
        'estimates',
        'logLikelihood',
        'fittedLogLikelihood',
        'backgroundProbability'
      ].includes(port);
      const allowance = floatPort ? tolerance * Math.max(1, Math.abs(value)) : 0;
      expect(Math.abs(values[index] - value), port).toBeLessThanOrEqual(allowance);
    });
  }
}
