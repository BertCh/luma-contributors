// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getGPUSpatialLagMaximumLikelihoodModelResult,
  getGPUSpatialErrorMaximumLikelihoodModelResult
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-model-result';
import {getGPUSpatialErrorGMModelResult} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-error-gm-model-result';
import {getGPUSpatialTwoStageLeastSquaresModelResult} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-two-stage-least-squares-model-result';

for (const reader of [
  getGPUSpatialLagMaximumLikelihoodModelResult,
  getGPUSpatialErrorMaximumLikelihoodModelResult,
  getGPUSpatialErrorGMModelResult,
  getGPUSpatialTwoStageLeastSquaresModelResult
]) {
  for (const predictorNames of [undefined, ['income']]) {
    it(`${reader.name} ignores spare capacity with predictor names ${Boolean(predictorNames)}`, () => {
      const table = Float32Array.of(10, 1, 10, 0, 20, 2, 10, 0, 0.5, 0.1, 5, 0, 99, 98, 97, 96);
      const readback = {table, summary: new Float32Array(32).fill(10), status: Uint32Array.of(0)};
      const result = reader(readback, {predictorCount: 1, predictorNames});
      const exact = reader({...readback, table: table.subarray(0, 12)}, {predictorNames});
      expect(result.coefficients.map(coefficient => coefficient.name)).toEqual([
        'CONSTANT',
        predictorNames?.[0] ?? 'x0',
        reader === getGPUSpatialErrorMaximumLikelihoodModelResult ||
        reader === getGPUSpatialErrorGMModelResult
          ? 'lambda'
          : 'W_y'
      ]);
      expect(result).toEqual(exact);
      // Independent dimensions: n=10, intercept + one predictor + spatial parameter => k=3.
      // Nonzero spare rows must not alter model degrees of freedom or add coefficient estimates.
      expect(result.parameterCount).toBe(3);
      expect(result.degreesOfFreedom).toBe(7);
      expect(result.coefficients.map(coefficient => coefficient.estimate)).toEqual([10, 20, 0.5]);
    });
  }
  it(`${reader.name} rejects an invalid fitted layout`, () => {
    const readback = {
      table: new Float32Array(12),
      summary: new Float32Array(32),
      status: Uint32Array.of(0)
    };
    for (const predictorCount of [0, -1, 1.5, NaN, Infinity, 2]) {
      expect(() => reader(readback, {predictorCount})).toThrow(/predictorCount/);
    }
    expect(
      reader({...readback, table: new Float32Array(13)}, {predictorCount: 1}).coefficients
    ).toHaveLength(3);
    expect(() => reader({...readback, table: new Float32Array(13)})).toThrow(/predictorCount/);
  });
}
