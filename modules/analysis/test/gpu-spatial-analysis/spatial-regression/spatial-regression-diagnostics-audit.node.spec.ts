// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeSpatialRegressionDiagnosticsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-regression-diagnostics-operation';
import fixtures from './spatial-regression-diagnostics-audit-fixtures.json';
import {expectRegressionRowsClose} from './regression-operation-test-utils';

for (const fixture of fixtures) {
  it(`diagnostics independent annihilator oracle: ${fixture.name}`, () => {
    const result = computeSpatialRegressionDiagnosticsOnCPU(
      {
        predictors: Float32Array.from(fixture.predictors),
        response: Float32Array.from(fixture.response),
        residuals: Float32Array.from(fixture.residuals),
        mask: Uint32Array.from(fixture.mask),
        weights: {
          offsets: Uint32Array.from(fixture.weights.offsets),
          neighbors: Uint32Array.from(fixture.weights.neighbors),
          weights: Float32Array.from(fixture.weights.weights)
        }
      },
      fixture
    );
    expect(Array.from(result.status)).toEqual([0]);
    for (const port of ['tests', 'summary'] as const) {
      const expected = Float32Array.from(fixture.expected[port]);
      expectRegressionRowsClose(result[port], expected, 3e-6);
      console.info(
        `${fixture.name} ${port}: ${Math.max(...Array.from(expected, (value, row) => Math.abs(value - result[port][row])))}`
      );
    }
  });
}
