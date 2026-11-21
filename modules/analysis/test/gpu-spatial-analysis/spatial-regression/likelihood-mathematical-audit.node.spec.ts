// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeSpatialLagMaximumLikelihoodOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-lag-maximum-likelihood-operation';
import {computeSpatialErrorMaximumLikelihoodOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-error-maximum-likelihood-operation';
import {fixture, getLikelihoodAuditInput} from './likelihood-mathematical-audit-test-utils';
import {expectRegressionRowsClose} from './regression-operation-test-utils';

for (const transformed of [false, true]) {
  for (const kind of ['lag', 'error'] as const) {
    it(`audits every ${kind} ML output against dense likelihood and full information: transformed=${transformed}`, () => {
      const expected = fixture.expected[transformed ? (`${kind}Transformed` as const) : kind];
      const compute =
        kind === 'lag'
          ? computeSpatialLagMaximumLikelihoodOnCPU
          : computeSpatialErrorMaximumLikelihoodOnCPU;
      const actual: Readonly<Record<string, Float32Array | Uint32Array>> = compute(
        getLikelihoodAuditInput(transformed),
        {
          predictorCount: 2,
          maximumPower: 64,
          traceMode: 'exact'
        }
      );
      const differences: Record<string, number> = {};
      for (const port of Object.keys(expected) as (keyof typeof expected)[]) {
        const values = actual[port];
        const reference = Float32Array.from(expected[port], value => value ?? NaN);
        expectRegressionRowsClose(values, reference, 3e-6);
        differences[port] = Math.max(
          0,
          ...reference.map((value, row) =>
            Number.isNaN(value) ? 0 : Math.abs(values[row] - value)
          )
        );
      }
      expect(actual['status'][0]).toBe(0);
      expect(actual['converged'][0]).toBe(1);
      console.info(
        `${kind} transformed=${transformed} maximum absolute oracle differences`,
        differences
      );
    });
  }
}
