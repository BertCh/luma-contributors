// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeSpatialErrorGMOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-error-gm-operation';
import fixtures from './spatial-error-gm-audit-fixtures.json';
import {expectRegressionRowsClose} from './regression-operation-test-utils';

for (const fixture of fixtures) {
  it(`GM independent dense oracle covers every output: ${fixture.name}`, () => {
    const result = computeSpatialErrorGMOnCPU(
      {
        predictors: Float32Array.from(fixture.predictors),
        response: Float32Array.from(fixture.response),
        weights: {
          offsets: Uint32Array.from(fixture.weights.offsets),
          neighbors: Uint32Array.from(fixture.weights.neighbors),
          weights: Float32Array.from(fixture.weights.weights)
        },
        traceParameters: Uint32Array.of(fixture.seed)
      },
      {
        predictorCount: 2,
        estimator: fixture.estimator as 'homoskedastic' | 'kelejian-prucha',
        traceMode: fixture.mode as 'exact' | 'stochastic',
        traceProbeCount: fixture.probeCount
      }
    );
    expect(Array.from(result.status)).toEqual(fixture.expected.status);
    for (const port of ['table', 'summary', 'residuals'] as const) {
      const expected = Float32Array.from(fixture.expected[port], value => value ?? NaN);
      // Final float32 rounding plus SciPy scalar minimization error; the CPU uses float64.
      expectRegressionRowsClose(result[port], expected, 2e-6);
      const maximumDifference = Math.max(
        ...Array.from(expected, (value, row) =>
          Number.isFinite(value) ? Math.abs(result[port][row] - value) : 0
        )
      );
      console.info(`${fixture.name} ${port} maximum absolute difference: ${maximumDifference}`);
    }
  });
}
