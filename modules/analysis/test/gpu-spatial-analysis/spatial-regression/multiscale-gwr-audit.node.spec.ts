// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeMultiscaleGeographicallyWeightedRegressionOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-multiscale-geographically-weighted-regression-operation';
import {
  createMultiscaleAuditFixture,
  createMultiscaleOverflowFixture
} from './multiscale-gwr-audit-fixtures';

it('R6 CPU MGWR rejects an unrepresentable published residual variance', () => {
  const result = computeMultiscaleGeographicallyWeightedRegressionOnCPU(
    createMultiscaleOverflowFixture(),
    {predictorCount: 1, inference: {}}
  );
  expect(Array.from(result.coefficients)).toEqual(Array(16).fill(0));
  expect(Array.from(result.fitted)).toEqual(Array(8).fill(0));
  expect(Array.from(result.inference!.summary)).toEqual([2, 2, Infinity, 6]);
  expect(Array.from(result.inference!.status)).toEqual([2]);
  expect(result.inference!.standardErrors.every(Number.isNaN)).toBe(true);
  expect(result.inference!.effectiveParameters.every(Number.isNaN)).toBe(true);
});

for (const kernel of ['gaussian', 'bisquare'] as const) {
  for (const varianceConvention of [undefined, 'residual-df', 'smoother-df'] as const) {
    it(`MGWR ${kernel}/${varianceConvention ?? 'conditional'} matches the simultaneous block oracle on every output and rescaled copy`, () => {
      const maximumErrors: Record<string, number> = {};
      for (const rescaled of [false, true]) {
        const {input, expected} = createMultiscaleAuditFixture(
          kernel,
          rescaled,
          varianceConvention
        );
        const result = computeMultiscaleGeographicallyWeightedRegressionOnCPU(input, {
          predictorCount: 1,
          kernel,
          maximumIterations: 4,
          inference: varianceConvention ? {varianceConvention} : undefined
        });
        const actual = {
          ...result,
          inferenceStandardErrors: result.inference?.standardErrors,
          inferenceEffectiveParameters: result.inference?.effectiveParameters,
          inferenceSummary: result.inference?.summary,
          inferenceStatus: result.inference?.status
        };
        for (const [name, values] of Object.entries(expected)) {
          const output = actual[name as keyof typeof actual] as Float32Array | Uint32Array;
          expect(output.length, name).toBe(values.length);
          values.forEach((value, index) => {
            if (Number.isNaN(value)) expect(output[index], `${name}[${index}]`).toBeNaN();
            else {
              expect(Number.isFinite(output[index]), `${name}[${index}]`).toBe(true);
              const difference = Math.abs(output[index] - value);
              maximumErrors[name] = Math.max(maximumErrors[name] ?? 0, difference);
              expect(difference, `${name}[${index}]`).toBeLessThanOrEqual(
                name === 'codes' ||
                  name === 'iterationCount' ||
                  name === 'inferenceStatus' ||
                  name === 'componentSearchIterationLimitReached' ||
                  name === 'selectedBandwidths'
                  ? 0
                  : 1e-6
              );
            }
          });
        }
      }
      console.info(
        `${kernel}/${varianceConvention ?? 'conditional'} maximum absolute differences`,
        maximumErrors
      );
    });
  }
}
